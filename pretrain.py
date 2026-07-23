# ruff: noqa: E402  # XLA flags must be set before importing JAX.
import xla_flags

xla_flags.set()

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Union

import grain
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import rich
import tyro
import wandb
import yaml
from grain.checkpoint import CheckpointRestore as GrainRestore
from grain.checkpoint import CheckpointSave as GrainSave
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from orbax.checkpoint.args import StandardRestore, StandardSave
from tqdm import tqdm, trange
from tyro.conf import subcommand

from data import FineWeb
from opt import EMA, Adam
from transformer import Transformer
from util import astype, config_to_dict, get_max_flops_sec, laxmean, tree_to_dict

ModelSize = Union[
    Annotated[
        Transformer,
        subcommand("nano", default=Transformer(D=768, L=12, M=4 * 768, H=12, K=64)),
    ],
    Annotated[
        Transformer,
        subcommand(
            "olmo150m", default=Transformer(D=1024, L=12, M=4 * 1024, H=16, K=64)
        ),
    ],
]


@dataclass
class TrainConfig:
    model: ModelSize
    dataset: FineWeb
    batch_size: int
    log_dir: str
    tokens: int
    opt: Adam
    eval_batch_size: int
    save_checkpoint: bool = False
    n_eval: int = 20
    max_eval_tokens: int = 100_000_000
    load_checkpoint: str | None = None
    ghost_batch_size: int = -1
    ema: EMA = EMA(())
    seed: int = 0
    tok_per_step: int = field(init=False)
    steps: int = field(init=False)
    flops_per_seq: int = field(init=False)

    def __post_init__(self):
        self.model = replace(self.model, V=self.dataset.vocab_size)
        self.tok_per_step = self.batch_size * self.dataset.seq_len
        self.steps = (self.tokens + self.tok_per_step - 1) // self.tok_per_step
        self.flops_per_seq = self.model.flops_per_seq(self.dataset.seq_len)
        self.opt.schedule = replace(self.opt.schedule, steps=self.steps)


def main(cfg: TrainConfig):
    print(f"Running for {cfg.steps} steps, {cfg.steps * cfg.tok_per_step} tokens.")
    cfg_dict = config_to_dict(cfg)
    rich.print(cfg_dict)
    log_path = Path(cfg.log_dir)
    wandb.init(
        project="transformer_base",
        config=cfg_dict,
        dir=log_path / "wandb",
        entity="harvardml",
    )
    assert wandb.run is not None
    checkpoint_dir = log_path / "checkpoints" / wandb.run.id
    print("Saving checkpoints to:", checkpoint_dir)
    if cfg.save_checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (checkpoint_dir / "config.yaml").open("w") as f:
            yaml.safe_dump(cfg_dict, f)

    print("Creating Mesh...")
    mesh = jax.make_mesh((jax.device_count(),), ("data",), (AxisType.Explicit,))
    print(mesh)
    jax.set_mesh(mesh)

    ds = cfg.dataset
    train_ds = ds.build("train", cfg.batch_size, cfg.seed, shuffle=True, repeat=True)
    eval_ds = ds.build("eval", cfg.eval_batch_size, seed=0, shuffle=False, repeat=False)
    eval_bytes = 0
    for eval_batches, (_, n_bytes) in enumerate(eval_ds, start=1):
        eval_bytes += int(n_bytes.sum())
        eval_tokens = eval_batches * cfg.eval_batch_size * cfg.dataset.seq_len
        if eval_tokens >= cfg.max_eval_tokens:
            break
    eval_tokens = eval_batches * cfg.eval_batch_size * cfg.dataset.seq_len
    bpb_factor = eval_tokens / (np.log(2) * eval_bytes)
    wandb.config.update({"bpb_factor": bpb_factor})

    sharding = jax.NamedSharding(mesh, P("data"))
    train_ds = grain.experimental.device_put(train_ds, (sharding, None))
    eval_ds = grain.experimental.device_put(eval_ds, (sharding, None))
    ds_iter = iter(train_ds)

    model, opt, ema = cfg.model, cfg.opt, cfg.ema
    params = astype(model.init(jr.key(cfg.seed)), jnp.float32)
    opt_state = opt.init(params, model.lrs)
    ema_state = ema.init(params)
    nu_ema_state = ema.init(opt_state.nu)

    start_step = 0
    if cfg.load_checkpoint is not None:
        ckpt_dir = Path(cfg.load_checkpoint)
        start_step = int(ckpt_dir.name)
        has_nu_ema = (ckpt_dir / "nu_ema").exists()
        with ocp.CheckpointManager(ckpt_dir.parent) as loader:
            restore_args = dict(
                params=StandardRestore(params),
                opt=StandardRestore(opt_state),
                ema=StandardRestore(ema_state),
                ds=GrainRestore(ds_iter),
            )
            if has_nu_ema:
                restore_args["nu_ema"] = StandardRestore(nu_ema_state)
            ckpt = loader.restore(start_step, args=ocp.args.Composite(**restore_args))
            params = astype(ckpt["params"], jnp.float32)
            opt_state = ckpt["opt"]
            ema_state = astype(ckpt["ema"], jnp.float32)
            if has_nu_ema:
                nu_ema_state = ckpt["nu_ema"]
            else:
                nu_ema_state = ema.init(opt_state.nu)._replace(
                    count=int(np.asarray(opt_state.count))
                )
            del ckpt
        print(f"Loaded checkpoint from step {start_step}")

    @jax.jit
    def loss_fn(p, batch):
        x, y = batch
        logits = model(p, x).astype(jnp.float32)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step_fn(params, opt_state, ema_state, nu_ema_state, batch):
        def batch_grad_fn(b):
            p_cast = astype(params, model.dtype)
            return jax.value_and_grad(loss_fn)(p_cast, b)

        L, g = laxmean(batch_grad_fn, batch, batch_size=cfg.ghost_batch_size)
        g = astype(g, jnp.float32)

        dp, opt_state, opt_log = opt.update(g, opt_state)

        next_params = optax.apply_updates(params, dp)
        ema_state = ema.update(next_params, ema_state)
        nu_ema_state = ema.update(opt_state.nu, nu_ema_state)
        nu_ema_log = {
            f"ema_{pct}": tree_to_dict(jax.tree.map(jnp.mean, nu_ema_state.ema[pct]))
            for pct in ema.pct
        }
        out_dict = dict(loss=L, opt={**opt_log, "nu_ema": nu_ema_log})
        return next_params, opt_state, ema_state, nu_ema_state, out_dict

    @jax.jit
    def eval_loss_fn(params, batch):
        params = astype(params, model.dtype)
        return loss_fn(params, batch)

    def eval_all(params, ema_state):
        results = {}
        all_params = {0: params} | ema_state.ema
        for label, p in tqdm(all_params.items(), desc="eval"):
            eval_iter = iter(eval_ds)
            batch, _ = next(eval_iter)
            total = eval_loss_fn(p, batch)
            for _ in range(eval_batches - 1):
                batch, _ = next(eval_iter)
                total += eval_loss_fn(p, batch)
            results[f"eval/ema_{label}"] = total / eval_batches
        return jax.device_get(tree_to_dict(results, sep="/"))

    print("Compiling...")
    compile_batch, _ = next(iter(train_ds))
    jax.block_until_ready(
        step_fn(params, opt_state, ema_state, nu_ema_state, compile_batch)
    )
    print("Data Sharding:", jax.typeof(compile_batch[0]))
    del compile_batch

    print("Running...")
    buffer = []
    last_flush_time = time.perf_counter()

    eval_steps = set()
    if cfg.n_eval > 0:
        eval_steps = set(map(int, np.linspace(0, cfg.steps, cfg.n_eval)))
        print(f"Evaluating at steps: {sorted(eval_steps)}")

    def flush_buffer():
        nonlocal last_flush_time
        if not buffer:
            return

        chunk_size = 64
        rows = list(buffer)
        buffer.clear()

        host_rows = []
        last_step = None
        for start in range(0, len(rows), chunk_size):
            host_rows.extend(jax.device_get(rows[start : start + chunk_size]))

        now = time.perf_counter()
        elapsed_time = now - last_flush_time
        last_flush_time = now

        for row in host_rows:
            aux = dict(tokens=row["step"] * cfg.tok_per_step, step_bytes=row["n_bytes"])
            wandb.log(tree_to_dict(row | aux, sep="/"), step=row["step"])
            last_step = row["step"]

        tok_sec = cfg.tok_per_step * len(host_rows) / elapsed_time
        total_flops = cfg.flops_per_seq * cfg.batch_size * len(host_rows)
        max_flops_sec = get_max_flops_sec(cfg.model.dtype)
        mfu = total_flops / (elapsed_time * max_flops_sec)
        wandb.log({"Mtok_sec": tok_sec / 1e6, "MFU": mfu}, step=last_step)

    if cfg.save_checkpoint:
        checkpointer = ocp.CheckpointManager(
            checkpoint_dir, item_names=("params", "opt", "ema", "nu_ema", "ds")
        )

        def save_checkpoint(step, params, opt_state, ema_state, nu_ema_state, ds_iter):
            save_args = dict(
                params=StandardSave(astype(params, model.dtype)),
                opt=StandardSave(opt_state),
                ema=StandardSave(astype(ema_state, model.dtype)),
                nu_ema=StandardSave(nu_ema_state),
                ds=GrainSave(ds_iter),
            )
            checkpointer.save(step=step, args=ocp.args.Composite(**save_args))

    pbar = trange(start_step, cfg.steps, total=cfg.steps, initial=start_step)
    for i in pbar:
        if i in eval_steps:
            flush_buffer()
            wandb.log(eval_all(params, ema_state), step=i)
            if cfg.save_checkpoint:
                save_checkpoint(i, params, opt_state, ema_state, nu_ema_state, ds_iter)
            last_flush_time = time.perf_counter()

        batch, n_bytes = next(ds_iter)
        params, opt_state, ema_state, nu_ema_state, out_dict = step_fn(
            params, opt_state, ema_state, nu_ema_state, batch
        )
        buffer.append(out_dict | {"step": i, "n_bytes": n_bytes.sum()})
        if time.perf_counter() - last_flush_time >= 60:
            flush_buffer()

    flush_buffer()
    eval_dict = eval_all(params, ema_state)
    eval_dict |= {"step": cfg.steps, "tokens": cfg.steps * cfg.tok_per_step}
    wandb.log(eval_dict, step=cfg.steps)
    if cfg.save_checkpoint:
        save_checkpoint(cfg.steps, params, opt_state, ema_state, nu_ema_state, ds_iter)
        checkpointer.wait_until_finished()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
