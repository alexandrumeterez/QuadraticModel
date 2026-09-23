import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import xla_flags

xla_flags.set()

import time
import os
from dataclasses import replace

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
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from orbax.checkpoint.args import StandardRestore
from tqdm import tqdm, trange

from harvest import reap
from opt import Adam
from pretrain import TrainConfig
from util import (
    astype,
    config_to_dict,
    dict_to_config,
    get_max_flops_sec,
    laxmean,
    tree_to_dict,
)

REST_G2_CSV_NAME = "sample_g2_over_nu_rest_below_top1pct_any_weight.csv"
RUN_FRACTION_OF_TOTAL_STEPS = 0.1
LIN_LOSS_ABORT_THRESHOLD = 1e10


class CheckpointOffsetCsvReader:
    def __init__(self, csv_path: Path, max_offset: int | None = None):
        if not csv_path.exists():
            raise FileNotFoundError(f"g2 rest-sample CSV not found: {csv_path}")

        self.csv_path = csv_path
        self.max_offset = max_offset
        self.file = csv_path.open("r")
        header_line = self.file.readline()
        if not header_line:
            self.file.close()
            raise ValueError(f"empty g2 rest-sample CSV: {csv_path}")
        header = header_line.rstrip("\n").split(",")
        try:
            self.offset_col = header.index("checkpoint_offset_sample_index")
        except ValueError as exc:
            self.file.close()
            raise ValueError(
                "Expected checkpoint_offset_sample_index column in" f" {csv_path}"
            ) from exc
        self.prev = -1
        self.line_no = 1
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        for line in self.file:
            self.line_no += 1
            if not line.strip():
                continue
            parts = line.split(",", self.offset_col + 1)
            if len(parts) <= self.offset_col:
                raise ValueError(f"Malformed row {self.line_no} in {self.csv_path}")
            offset = int(parts[self.offset_col])
            if offset <= self.prev:
                raise ValueError(
                    "checkpoint_offset_sample_index must be strictly increasing:"
                    f" row {self.line_no} has {offset} after {self.prev}"
                    f" in {self.csv_path}"
                )
            if self.max_offset is not None and offset >= self.max_offset:
                raise ValueError(
                    "checkpoint_offset_sample_index exceeds the remaining"
                    " post-checkpoint training stream:"
                    f" row {self.line_no} has {offset}, max allowed offset is"
                    f" {self.max_offset - 1} in {self.csv_path}"
                )
            self.prev = offset
            return offset
        self.close()
        raise StopIteration

    def close(self):
        if not self.closed:
            self.file.close()
            self.closed = True


class OffsetFilteredBatchIterator:
    def __init__(
        self,
        source_iter,
        accepted_offsets,
        batch_size: int,
        sharding,
    ):
        self.source_iter = source_iter
        self.accepted_offsets = iter(accepted_offsets)
        self.next_accepted_offset = next(self.accepted_offsets, None)
        self.batch_size = batch_size
        self.sharding = sharding
        self.next_offset = 0
        self.pending_batches = []
        self.pending_n_bytes = []
        self.pending_offsets = []
        self.pending_count = 0
        self.carry_batch = None
        self.carry_n_bytes = None
        self.carry_offsets = None
        self.last_offsets = None

    def __iter__(self):
        return self

    def close(self):
        close = getattr(self.accepted_offsets, "close", None)
        if close is not None:
            close()

    def _take_chunk(self, batch, n_bytes, offsets):
        chunk_size = int(jax.tree.leaves(batch)[0].shape[0])
        offset = 0
        while offset < chunk_size and self.pending_count < self.batch_size:
            need = self.batch_size - self.pending_count
            take = min(need, chunk_size - offset)
            sl = slice(offset, offset + take)
            self.pending_batches.append(jax.tree.map(lambda x: x[sl], batch))
            self.pending_n_bytes.append(n_bytes[sl])
            self.pending_offsets.append(offsets[sl])
            self.pending_count += take
            offset += take

        if offset < chunk_size:
            sl = slice(offset, None)
            self.carry_batch = jax.tree.map(lambda x: x[sl], batch)
            self.carry_n_bytes = n_bytes[sl]
            self.carry_offsets = offsets[sl]

    def _append_accepted_from_source_batch(self, batch, n_bytes):
        source_batch_size = int(jax.tree.leaves(batch)[0].shape[0])
        start_offset = self.next_offset
        stop_offset = start_offset + source_batch_size
        self.next_offset = stop_offset

        while (
            self.next_accepted_offset is not None
            and self.next_accepted_offset < start_offset
        ):
            self.next_accepted_offset = next(self.accepted_offsets, None)

        source_indices = []
        while (
            self.next_accepted_offset is not None
            and self.next_accepted_offset < stop_offset
        ):
            source_indices.append(self.next_accepted_offset - start_offset)
            self.next_accepted_offset = next(self.accepted_offsets, None)

        if not source_indices:
            return

        source_indices = np.asarray(source_indices, dtype=np.int64)
        batch = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), batch)
        n_bytes = np.asarray(jax.device_get(n_bytes))
        accepted_batch = jax.tree.map(lambda x: x[source_indices], batch)
        accepted_n_bytes = n_bytes[source_indices]
        accepted_offsets = start_offset + source_indices
        self._take_chunk(accepted_batch, accepted_n_bytes, accepted_offsets)

    def _pop_pending_batch(self):
        batch = jax.tree.map(
            lambda *xs: np.concatenate(xs, axis=0), *self.pending_batches
        )
        n_bytes = np.concatenate(self.pending_n_bytes, axis=0)
        offsets = np.concatenate(self.pending_offsets, axis=0)
        self.pending_batches = []
        self.pending_n_bytes = []
        self.pending_offsets = []
        self.pending_count = 0
        self.last_offsets = offsets
        return jax.device_put((batch, n_bytes), (self.sharding, None))

    def __next__(self):
        if self.carry_batch is not None:
            carry_batch = self.carry_batch
            carry_n_bytes = self.carry_n_bytes
            carry_offsets = self.carry_offsets
            self.carry_batch = None
            self.carry_n_bytes = None
            self.carry_offsets = None
            self._take_chunk(carry_batch, carry_n_bytes, carry_offsets)

        while self.pending_count < self.batch_size:
            if self.next_accepted_offset is None:
                if self.pending_count:
                    self.pending_batches = []
                    self.pending_n_bytes = []
                    self.pending_offsets = []
                    self.pending_count = 0
                raise StopIteration
            batch, n_bytes = next(self.source_iter)
            self._append_accepted_from_source_batch(batch, n_bytes)

        return self._pop_pending_batch()


def main(
    checkpoint_dir: Path,
    ema: float | None = None,
    transform: Literal["none", "quad", "prox"] = "none",
    train: Literal["full", "head"] = "full",
    optimizer: Literal["adam", "stale_adam", "frozen_adam", "sgdm"] = "adam",
    clip_threshold: float = float("inf"),
    lr_mult: float = 1.0,
    bsz_mult: float = 1.0,
    n_eval: int | None = 10,
    max_eval_tokens: int | None = 10_000_000,
    rest_csv_rel: str | None = None,
):
    if ema == 0:
        ema = None
    clip_threshold = float("inf")
    if clip_threshold <= 0:
        raise ValueError("clip_threshold must be positive")
    if lr_mult <= 0:
        raise ValueError("lr_mult must be positive")
    if bsz_mult <= 0:
        raise ValueError("bsz_mult must be positive")
    if n_eval is not None and n_eval < 0:
        raise ValueError("n_eval must be non-negative")
    if max_eval_tokens is not None and max_eval_tokens <= 0:
        raise ValueError("max_eval_tokens must be positive")

    def scale_int(name, value):
        scaled = max(1, int(round(value * bsz_mult)))
        if scaled <= 0:
            raise ValueError(f"{name} must remain positive after scaling")
        return scaled

    def make_checkpoint_schedule(schedule):
        ckpt_opt_cfg = cfg_dict["opt"]
        ckpt_sched_cfg = ckpt_opt_cfg["schedule"]
        ckpt_schedule = replace(
            schedule,
            warmup_pct=ckpt_sched_cfg["warmup_pct"],
            init_value=ckpt_sched_cfg["init_value"],
            peak_value=ckpt_sched_cfg["peak_value"],
            end_value=ckpt_sched_cfg["end_value"],
            steps=ckpt_sched_cfg["steps"],
        )
        return ckpt_schedule, float(ckpt_opt_cfg["lr"])

    def ema_label(value):
        return f"{value:g}".replace(".", "p")

    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_step = int(checkpoint_dir.name)
    checkpoint_run_id = checkpoint_dir.parent.name
    if rest_csv_rel is None:
        rest_csv_rel = os.environ.get("REST_SAMPLE_CSV_REL") or os.environ.get(
            "REST_G2_CSV_REL"
        )
    if rest_csv_rel is None:
        g2_split_dir = "sample_g2_over_nu_splits"
        if ema is not None:
            g2_split_dir = f"sample_g2_over_nu_ema_{ema_label(ema)}_splits"
        rest_csv_rel = f"{g2_split_dir}/{REST_G2_CSV_NAME}"
    rest_csv_path = checkpoint_dir / rest_csv_rel

    config_path = checkpoint_dir.parent / "config.yaml"
    with config_path.open("r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = dict_to_config(cfg_dict, TrainConfig)
    if n_eval is not None:
        cfg = replace(cfg, n_eval=n_eval)
    if max_eval_tokens is not None:
        cfg = replace(cfg, max_eval_tokens=max_eval_tokens)
    base_batch_size = cfg.batch_size
    base_lr = cfg.opt.lr
    checkpoint_total_train_steps = cfg.steps
    batch_size = scale_int("batch_size", cfg.batch_size)
    ghost_batch_size = cfg.ghost_batch_size
    stream_batch_size = base_batch_size
    cfg = replace(
        cfg,
        batch_size=batch_size,
        ghost_batch_size=ghost_batch_size,
        opt=replace(cfg.opt, lr=cfg.opt.lr * lr_mult),
        load_checkpoint=str(checkpoint_dir),
        save_checkpoint=False,
    )
    model = cfg.model
    lin_model = replace(
        model, dtype=jnp.float32, flash_attention=False, grad_checkpoint=False
    )

    print(f"Loading from checkpoint: {checkpoint_dir}")
    cfg_log = config_to_dict(cfg) | dict(
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_run_id=checkpoint_run_id,
        checkpoint_step=checkpoint_step,
        ema=ema,
        transform=transform,
        train=train,
        optimizer=optimizer,
        clip_threshold=clip_threshold,
        lr_mult=lr_mult,
        bsz_mult=bsz_mult,
        base_batch_size=base_batch_size,
        stream_batch_size=stream_batch_size,
        base_lr=base_lr,
        checkpoint_total_train_steps=checkpoint_total_train_steps,
        n_eval=cfg.n_eval,
        max_eval_tokens=cfg.max_eval_tokens,
        sample_rest_filter=True,
        sample_rest_csv=str(rest_csv_path),
        sample_rest_csv_rel=rest_csv_rel,
        g2_rest_filter=True,
        g2_rest_csv=str(rest_csv_path),
        eval_model_flash_attention=model.flash_attention,
        eval_lin_model_flash_attention=lin_model.flash_attention,
    )
    rich.print(cfg_log)

    log_path = Path(cfg.log_dir)
    wandb.init(
        project="transformer_base",
        config=cfg_log,
        dir=log_path / "wandb",
        group=os.environ.get("WANDB_RUN_GROUP", "eos_linearization_full_js"),
        entity="harvardml",
    )

    print("Creating Mesh...")
    mesh = jax.make_mesh((jax.device_count(),), ("data",), (AxisType.Explicit,))
    print(mesh)
    jax.set_mesh(mesh)

    ds = cfg.dataset
    # Match the checkpointed training input pipeline for GrainRestore. The
    # checkpoint-offset guard below prevents consuming repeated samples.
    train_ds = ds.build("train", stream_batch_size, cfg.seed, shuffle=True, repeat=True)
    eval_ds = ds.build("eval", cfg.eval_batch_size, seed=0, shuffle=False, repeat=False)
    eval_bytes = 0
    for eval_batches, (_, n_bytes) in enumerate(eval_ds, start=1):
        eval_bytes += int(n_bytes.sum())
        eval_tokens = eval_batches * cfg.eval_batch_size * cfg.dataset.seq_len
        if eval_tokens >= cfg.max_eval_tokens:
            break
    eval_tokens = eval_batches * cfg.eval_batch_size * cfg.dataset.seq_len
    bpb_factor = eval_tokens / (np.log(2) * eval_bytes)
    wandb.config.update(
        {
            "eval_batches": eval_batches,
            "eval_tokens": eval_tokens,
            "bpb_factor": bpb_factor,
        }
    )

    sharding = jax.NamedSharding(mesh, P("data"))
    train_ds = grain.experimental.device_put(train_ds, (sharding, None))
    eval_ds = grain.experimental.device_put(eval_ds, (sharding, None))
    ds_iter = iter(train_ds)

    opt = cfg.opt
    assert isinstance(
        opt, Adam
    ), "linearization_full currently requires Adam checkpoints"
    checkpoint_schedule, checkpoint_lr = make_checkpoint_schedule(opt.schedule)
    ema_cfg = cfg.ema
    nu_ema = ema_cfg
    params = astype(model.init(jr.key(cfg.seed)), jnp.float32)
    opt_state = opt.init(params, model.lrs)
    ema_state = ema_cfg.init(params)
    nu_ema_state = nu_ema.init(opt_state.nu)

    has_nu_ema = (checkpoint_dir / "nu_ema").exists()
    ckpt_step = checkpoint_step
    with ocp.CheckpointManager(checkpoint_dir.parent) as loader:
        restore_args = dict(
            params=StandardRestore(params),
            opt=StandardRestore(opt_state),
            ema=StandardRestore(ema_state),
            ds=GrainRestore(ds_iter),
        )
        if has_nu_ema:
            restore_args["nu_ema"] = StandardRestore(nu_ema_state)
        ckpt = loader.restore(ckpt_step, args=ocp.args.Composite(**restore_args))
        params = astype(ckpt["params"], jnp.float32)
        opt_state = ckpt["opt"]
        ema_state = astype(ckpt["ema"], jnp.float32)
        if has_nu_ema:
            nu_ema_state = ckpt["nu_ema"]
        else:
            nu_ema_state = nu_ema.init(opt_state.nu)._replace(
                count=int(np.asarray(opt_state.count))
            )
        if ema is not None:
            assert has_nu_ema, f"Expected nu_ema in checkpoint: {checkpoint_dir}"
            assert ema in ema_cfg.pct, f"{ema} not in {ema_cfg.pct}"
            params = astype(ema_state.ema[ema], jnp.float32)
            opt_state = opt_state._replace(
                nu=jax.tree.map(
                    lambda src, n: astype(src, n.dtype),
                    nu_ema_state.ema[ema],
                    opt_state.nu,
                )
            )
        del ckpt

    start_step = int(np.asarray(opt_state.count))
    if start_step != ckpt_step:
        print(
            f"Checkpoint dir step {ckpt_step} restored with optimizer step {start_step}"
        )
    print(f"Loaded checkpoint from step {start_step}")
    remaining_train_source_samples = max(
        0, (checkpoint_total_train_steps - start_step) * stream_batch_size
    )
    accepted_offsets = CheckpointOffsetCsvReader(
        rest_csv_path,
        max_offset=remaining_train_source_samples,
    )
    print(
        "Streaming rest-sample filter from"
        f" {rest_csv_path} with max checkpoint offset"
        f" {remaining_train_source_samples}"
    )
    checkpoint_base_lr = float(
        np.asarray(checkpoint_schedule(start_step) * checkpoint_lr)
    )
    fixed_base_lr = checkpoint_base_lr * lr_mult
    planned_run_steps = max(1, int(np.ceil(RUN_FRACTION_OF_TOTAL_STEPS * cfg.steps)))
    end_step = start_step + planned_run_steps
    run_tokens = planned_run_steps * cfg.tok_per_step
    print(
        f"Running for {planned_run_steps} steps, {run_tokens} tokens, through step {end_step}."
    )
    wandb.config.update(
        {
            "checkpoint_base_lr": checkpoint_base_lr,
            "fixed_base_lr": fixed_base_lr,
            "fixed_lr_step": int(np.asarray(opt_state.count)),
            "start_step": start_step,
            "total_train_steps": cfg.steps,
            "checkpoint_total_train_steps": checkpoint_total_train_steps,
            "run_fraction_of_total_steps": RUN_FRACTION_OF_TOTAL_STEPS,
            "lin_loss_abort_threshold": LIN_LOSS_ABORT_THRESHOLD,
            "run_steps": planned_run_steps,
            "end_step": end_step,
            "run_tokens": run_tokens,
            "stream_batch_size": stream_batch_size,
            "remaining_train_source_samples": remaining_train_source_samples,
            "sample_rest_filter": True,
            "sample_rest_csv": str(rest_csv_path),
            "sample_rest_csv_rel": rest_csv_rel,
            "sample_rest_filter_mode": "streaming",
            "sample_rest_max_checkpoint_offset": remaining_train_source_samples,
            "g2_rest_filter": True,
            "g2_rest_csv": str(rest_csv_path),
            "g2_rest_filter_mode": "streaming",
            "g2_rest_max_checkpoint_offset": remaining_train_source_samples,
        }
    )
    print(
        "Using constant learning rate from"
        f" optimizer step {int(np.asarray(opt_state.count))}:"
        f" checkpoint_base_lr={checkpoint_base_lr:.8g}"
        f" lr_mult={lr_mult:.8g}"
        f" base_lr={fixed_base_lr:.8g}"
    )
    p0 = params

    keystr = lambda key: jax.tree_util.keystr(key, simple=True, separator=".")
    train_mask = jax.tree.map_with_path(
        lambda path, _: jnp.asarray(train == "full" or keystr(path) == "head"),
        params,
    )

    def zero_like_tree(tree):
        return jax.tree.map(lambda _: jnp.array(0.0, dtype=jnp.float32), tree)

    def zero_out(tree):
        return jax.tree.map(
            lambda x, m: jnp.where(m, x, jnp.zeros_like(x)), tree, train_mask
        )

    def keep_old(new_tree, old_tree):
        return jax.tree.map(
            lambda new, old, m: jnp.where(m, new, old), new_tree, old_tree, train_mask
        )

    def criterion(logits, y):
        logits = logits.astype(jnp.float32).reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    def criterion_quad(f, f0, y):
        df = f - f0
        crit = lambda z: criterion(z, y)

        def jvp_fn(z):
            L0, dL = jax.jvp(crit, (z,), (df,))
            return dL, L0

        dL, d2L, L0 = jax.jvp(jvp_fn, (f0,), (df,), has_aux=True)
        return L0 + dL + 0.5 * d2L

    @jax.jit
    def nonlinear_loss_fn(p, batch):
        x, y = batch
        return criterion(model(astype(p, model.dtype), x), y)

    @jax.jit
    def lin_apply_fn(p, p0, x):
        p0_cast = astype(p0, lin_model.dtype)
        dp = jax.tree.map(lambda pi, p0i: pi - p0i, p, p0)
        dp = astype(dp, lin_model.dtype)
        apply_fn = lambda pp: lin_model(pp, x).astype(jnp.float32)
        f0, df = jax.jvp(apply_fn, (p0_cast,), (dp,))
        return f0 + df, f0

    @jax.jit
    def lin_loss_fn(p, p0, batch):
        if transform == "none":
            return nonlinear_loss_fn(p, batch)
        x, y = batch
        f, f0 = lin_apply_fn(p, p0, x)
        if transform == "quad":
            return criterion_quad(f, f0, y)
        return criterion(f, y)

    @jax.jit
    def eval_losses_fn(params, p0, batch):
        params = astype(params, jnp.float32)
        return nonlinear_loss_fn(params, batch), lin_loss_fn(params, p0, batch)

    def eval_all(params, ema_state):
        results = {}
        all_params = {0: params}
        for label, p in tqdm(all_params.items(), desc="eval"):
            eval_iter = iter(eval_ds)
            batch, _ = next(eval_iter)
            loss_totals, lin_loss_totals = eval_losses_fn(p, p0, batch)
            for _ in range(eval_batches - 1):
                batch, _ = next(eval_iter)
                loss, lin_loss = eval_losses_fn(p, p0, batch)
                loss_totals = loss_totals + loss
                lin_loss_totals = lin_loss_totals + lin_loss
            loss = loss_totals / eval_batches
            lin_loss = lin_loss_totals / eval_batches
            results[f"eval/ema_{label}"] = loss
            results[f"eval/loss/ema_{label}"] = loss
            results[f"eval/lin_loss/ema_{label}"] = lin_loss
        return jax.device_get(tree_to_dict(results, sep="/"))

    def beta2_from_count(count):
        m = jnp.maximum(20.0, opt.b2_pct * count)
        return 1.0 - 1.0 / m

    def unit_nu_like(state):
        return jax.tree.map(lambda x: jnp.ones_like(x, dtype=jnp.float32), state.nu)

    def get_stale_nu_hat(state):
        return jax.lax.cond(
            state.count == 0,
            lambda _: unit_nu_like(state),
            lambda _: astype(opt.get_nu_hat(state), jnp.float32),
            operand=None,
        )

    def log_tree(tree):
        return tree_to_dict(
            jax.tree.map(lambda x: jnp.mean(x.astype(jnp.float32)), tree)
        )

    def zero_log(tree):
        return tree_to_dict(zero_like_tree(tree))

    def scale_grads(grads):
        return jax.tree.map(lambda g, lr: g * lr.pre, grads, model.lrs)

    def precond_from_nu(nu_hat):
        return jax.tree.map(
            lambda lr, n: (lr.pre * lr.post) / (jnp.sqrt(n) + opt.eps),
            model.lrs,
            nu_hat,
        )

    def clip_grads(grads, nu_tree):
        grads = zero_out(grads)
        g_pre = scale_grads(grads)
        eps = 1e-8
        criterion_sq = jax.tree.map(
            lambda g, n, m: jnp.where(
                m,
                jnp.mean(
                    (g.astype(jnp.float32) ** 2)
                    / ((jnp.sqrt(n.astype(jnp.float32)) + eps) ** 2)
                ),
                0.0,
            ),
            g_pre,
            nu_tree,
            train_mask,
        )
        criterion = jax.tree.map(jnp.sqrt, criterion_sq)
        exceed = jax.tree.map(lambda c: c > clip_threshold, criterion)
        exceed_leaves = jax.tree.leaves(exceed)
        should_skip = (
            jnp.any(jnp.stack(exceed_leaves)) if exceed_leaves else jnp.array(False)
        )
        clip_log = dict(
            clipped=should_skip.astype(jnp.int32),
            skipped=should_skip.astype(jnp.int32),
            criterion=tree_to_dict(criterion),
        )
        return grads, clip_log, should_skip

    def nu_ema_log(state):
        return {
            f"ema_{pct}": tree_to_dict(jax.tree.map(jnp.mean, state.ema[pct]))
            for pct in nu_ema.pct
        }

    def zero_opt_log(beta2, nu_ema_state):
        return dict(
            base_lr=jnp.array(0.0, dtype=jnp.float32),
            beta2=beta2,
            mu=zero_log(opt_state.mu),
            nu=zero_log(opt_state.nu),
            eff_lr=zero_log(opt_state.nu),
            dL=zero_log(opt_state.mu),
            nu_ema=nu_ema_log(nu_ema_state),
        )

    def adam_update(grads, state):
        grads = astype(grads, state.mu)
        grads_pre = scale_grads(grads)
        count = state.count + 1
        base_lr = jnp.asarray(fixed_base_lr, dtype=state.log_prod_b2.dtype)
        beta2_t = beta2_from_count(count)
        next_log_prod_b2 = state.log_prod_b2 + jnp.log(beta2_t).astype(
            state.log_prod_b2.dtype
        )

        mu = jax.tree.map(
            lambda m, g: m + (g - m) * (1 - opt.b1),
            state.mu,
            grads_pre,
        )
        next_nu = jax.tree.map(
            lambda n, g: n + (g**2 - n) * (1 - beta2_t),
            state.nu,
            grads_pre,
        )
        log_prod_b1 = count * jnp.log(opt.b1)
        mu_hat = opt.bias_correct(mu, log_prod_b1)
        nu_hat = opt.bias_correct(next_nu, next_log_prod_b2)
        dp = jax.tree.map(
            lambda m, n, lr: -base_lr * m / (jnp.sqrt(n) + opt.eps) * lr.post,
            mu_hat,
            nu_hat,
            model.lrs,
        )
        eff_lr = precond_from_nu(nu_hat)
        dL = jax.tree.map(lambda g, d: -jnp.vdot(g, d), grads, dp)
        opt_log = dict(
            base_lr=base_lr,
            beta2=beta2_t,
            mu=log_tree(mu_hat),
            nu=log_tree(nu_hat),
            eff_lr=log_tree(eff_lr),
            dL=tree_to_dict(dL),
        )
        next_state = state._replace(
            mu=mu, nu=next_nu, count=count, log_prod_b2=next_log_prod_b2
        )
        return dp, next_state, opt_log

    def custom_update(grads, state):
        grads_pre = scale_grads(grads)
        count = state.count + 1
        base_lr = jnp.asarray(fixed_base_lr, dtype=state.log_prod_b2.dtype)
        beta2_t = beta2_from_count(count)
        next_log_prod_b2 = state.log_prod_b2 + jnp.log(beta2_t).astype(
            state.log_prod_b2.dtype
        )

        mu = jax.tree.map(lambda m, g: m + (g - m) * (1 - opt.b1), state.mu, grads_pre)
        log_prod_b1 = count * jnp.log(opt.b1)
        mu_hat = opt.bias_correct(mu, log_prod_b1)
        stale_nu_hat = get_stale_nu_hat(state)
        next_nu = jax.tree.map(
            lambda n, g: n + (g**2 - n) * (1 - beta2_t), state.nu, grads_pre
        )
        next_nu_hat = opt.bias_correct(next_nu, next_log_prod_b2)

        if optimizer == "stale_adam":
            used_nu_hat = stale_nu_hat
            next_state = state._replace(
                mu=mu, nu=next_nu, count=count, log_prod_b2=next_log_prod_b2
            )
        elif optimizer == "frozen_adam":
            used_nu_hat = next_nu_hat
            next_state = state._replace(mu=mu, count=count)
        else:
            used_nu_hat = stale_nu_hat
            next_state = state._replace(mu=mu, count=count)

        dp = jax.tree.map(
            lambda m, n, lr: -base_lr * m / (jnp.sqrt(n) + opt.eps) * lr.post,
            mu_hat,
            used_nu_hat,
            model.lrs,
        )
        eff_lr = precond_from_nu(used_nu_hat)
        dL = jax.tree.map(lambda g, d: -jnp.vdot(g, d), grads, dp)
        opt_log = dict(
            base_lr=base_lr,
            beta2=beta2_t,
            mu=log_tree(mu_hat),
            nu=log_tree(used_nu_hat),
            eff_lr=log_tree(eff_lr),
            dL=tree_to_dict(dL),
        )
        return dp, next_state, opt_log

    @jax.jit
    def step_fn(params, opt_state, ema_state, nu_ema_state, p0, batch):
        def batch_grad_fn(b):
            p_model = astype(params, model.dtype)
            loss, aux = reap(nonlinear_loss_fn)(p_model, b)
            lin_loss, grads = jax.value_and_grad(lin_loss_fn)(params, p0, b)
            aux_sq = jax.tree.map(lambda x: jnp.mean(x.astype(jnp.float32) ** 2), aux)
            return (loss, lin_loss, grads), aux_sq

        (loss, lin_loss, grads), aux_sq = laxmean(
            batch_grad_fn, batch, batch_size=cfg.ghost_batch_size
        )
        grads = astype(grads, jnp.float32)
        aux = jax.tree.map(jnp.sqrt, aux_sq)

        grads, clip_log, should_skip = clip_grads(grads, get_stale_nu_hat(opt_state))
        beta2_t = beta2_from_count(opt_state.count + 1)

        def skip_step(_):
            out_dict = dict(
                loss=loss,
                lin_loss=lin_loss,
                opt=zero_opt_log(beta2_t, nu_ema_state),
                clip=clip_log,
                aux=aux,
            )
            return params, opt_state, ema_state, nu_ema_state, out_dict

        def do_step(_):
            if optimizer == "adam":
                dp, next_opt_state, opt_log = adam_update(grads, opt_state)
                next_opt_state = next_opt_state._replace(
                    mu=keep_old(next_opt_state.mu, opt_state.mu),
                    nu=keep_old(next_opt_state.nu, opt_state.nu),
                )
            else:
                dp, next_opt_state, opt_log = custom_update(grads, opt_state)
                next_opt_state = next_opt_state._replace(
                    mu=keep_old(next_opt_state.mu, opt_state.mu),
                    nu=keep_old(next_opt_state.nu, opt_state.nu),
                )

            dp = zero_out(dp)
            next_params = optax.apply_updates(params, dp)
            next_params = keep_old(next_params, params)
            next_ema_state = ema_cfg.update(next_params, ema_state)
            next_nu_ema_state = nu_ema.update(next_opt_state.nu, nu_ema_state)
            out_dict = dict(
                loss=loss,
                lin_loss=lin_loss,
                opt=dict(**opt_log, nu_ema=nu_ema_log(next_nu_ema_state)),
                clip=clip_log,
                aux=aux,
            )
            return (
                next_params,
                next_opt_state,
                next_ema_state,
                next_nu_ema_state,
                out_dict,
            )

        return jax.lax.cond(should_skip, skip_step, do_step, operand=None)

    train_iter = OffsetFilteredBatchIterator(
        ds_iter,
        accepted_offsets,
        batch_size=cfg.batch_size,
        sharding=sharding,
    )

    print("Compiling...")
    try:
        first_train_batch, first_train_n_bytes = next(train_iter)
    except StopIteration as exc:
        raise ValueError(
            "rest-sample filter ended before one full linearization batch could"
            f" be built: csv={rest_csv_path} batch_size={cfg.batch_size}"
        ) from exc
    pending_train_batch = (first_train_batch, first_train_n_bytes)
    jax.block_until_ready(
        step_fn(params, opt_state, ema_state, nu_ema_state, p0, first_train_batch)
    )
    first_eval_batch, _ = next(iter(eval_ds))
    jax.block_until_ready(eval_losses_fn(params, p0, first_eval_batch))
    print("Data Sharding:", jax.typeof(first_train_batch[0]))

    print("Running...")
    buffer = []
    last_flush_time = time.perf_counter()
    clipped_steps = 0
    run_steps = 0
    aborted_for_percentage_clipped = False
    aborted_for_loss = False
    aborted_for_lin_loss = False
    eval_steps = set()
    if cfg.n_eval > 0:
        eval_steps = set(
            int(x) for x in np.linspace(start_step, end_step, cfg.n_eval + 2)[1:-1]
        )
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

    pbar = trange(start_step, end_step, total=end_step, initial=start_step)
    for i in pbar:
        if i in eval_steps:
            flush_buffer()
            wandb.log(eval_all(params, ema_state), step=i)
            last_flush_time = time.perf_counter()
        try:
            if pending_train_batch is None:
                batch, n_bytes = next(train_iter)
            else:
                batch, n_bytes = pending_train_batch
                pending_train_batch = None
        except StopIteration:
            print(
                "Stopping early because the rest-sample filter ended before"
                f" another full batch could be built at step {i}."
            )
            break
        params, opt_state, ema_state, nu_ema_state, out_dict = step_fn(
            params, opt_state, ema_state, nu_ema_state, p0, batch
        )
        host_scalars = jax.device_get(
            dict(
                clipped=out_dict["clip"]["clipped"],
                loss=out_dict["loss"],
                lin_loss=out_dict["lin_loss"],
            )
        )
        clipped = int(np.asarray(host_scalars["clipped"]))
        loss = float(np.asarray(host_scalars["loss"]))
        lin_loss = float(np.asarray(host_scalars["lin_loss"]))
        clipped_steps += clipped
        run_steps += 1
        percentage_clipped = np.float32(clipped_steps / run_steps)
        clip_log = dict(out_dict["clip"], percentage_clipped=percentage_clipped)
        buffer.append(
            out_dict
            | {
                "step": i,
                "n_bytes": n_bytes.sum(),
                "clip": clip_log,
            }
        )
        if lin_loss > LIN_LOSS_ABORT_THRESHOLD:
            aborted_for_lin_loss = True
            flush_buffer()
            wandb.log(
                {
                    "run/aborted_for_lin_loss": 1,
                    "run/lin_loss_abort": lin_loss,
                    "run/loss_at_lin_loss_abort": loss,
                    "run/lin_loss_abort_threshold": LIN_LOSS_ABORT_THRESHOLD,
                },
                step=i,
            )
            print(
                f"Stopping early at step {i}: lin_loss={lin_loss:.6g}"
                f" exceeded {LIN_LOSS_ABORT_THRESHOLD:.6g}"
            )
            break
        # if run_steps > 10 and percentage_clipped > 0.5:
        #     aborted_for_percentage_clipped = True
        #     flush_buffer()
        #     wandb.log(
        #         {
        #             "run/aborted_for_percentage_clipped": 1,
        #             "run/percentage_clipped_abort": percentage_clipped,
        #         },
        #         step=i,
        #     )
        #     print(
        #         f"Stopping early at step {i}: clip/percentage_clipped={percentage_clipped:.3f} > 0.5"
        #     )
        #     break
        if time.perf_counter() - last_flush_time >= 60:
            flush_buffer()

    flush_buffer()
    if run_steps > 0:
        wandb.summary["clip/clipped_steps"] = clipped_steps
        wandb.summary["clip/percentage_clipped"] = clipped_steps / run_steps
    wandb.summary["run/actual_steps"] = run_steps
    close_train_iter = getattr(train_iter, "close", None)
    if close_train_iter is not None:
        close_train_iter()
    wandb.summary["run/aborted_for_loss"] = int(aborted_for_loss)
    wandb.summary["run/aborted_for_lin_loss"] = int(aborted_for_lin_loss)
    wandb.summary["run/aborted_for_percentage_clipped"] = int(
        aborted_for_percentage_clipped
    )
    if aborted_for_loss or aborted_for_lin_loss or aborted_for_percentage_clipped:
        wandb.finish()
        return
    final_eval_step = start_step + run_steps
    eval_dict = eval_all(params, ema_state)
    eval_dict |= {
        "step": final_eval_step,
        "tokens": final_eval_step * cfg.tok_per_step,
    }
    wandb.log(eval_dict, step=final_eval_step)
    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)
