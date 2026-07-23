# ruff: noqa: E402  # Set the import path and XLA flags before importing JAX.
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xla_flags

xla_flags.set()

import grain
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
import yaml
from grain.checkpoint import CheckpointRestore as GrainRestore
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from orbax.checkpoint.args import StandardRestore
from tqdm.auto import tqdm

from opt import Adam
from pretrain import TrainConfig
from util import astype, dict_to_config


def main(
    checkpoint_dir: Path,
    output_file: Path | None = None,
    num_samples: int = -1,
    num_tokens: int = 400_000_000,
    num_probes: int = 10,
    seed: int | None = None,
    probe_seed: int = 0,
    flush_every: int = 20,
    ema: float | None = 0.04,
    include_adam_precond: bool = True,
    allow_past_training_end: bool = False,
    repeat_dataset: bool = True,
):
    """Estimate per-sequence Gauss--Newton Frobenius norms with Hutchinson probes."""
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoint_step = int(checkpoint_dir.name)

    config_path = checkpoint_dir.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    dataset_state_dir = checkpoint_dir / "ds"
    if not any(dataset_state_dir.glob("process_*-of-*.json")):
        raise FileNotFoundError(
            "Dataset iterator state not found under "
            f"{dataset_state_dir}. It is required to reconstruct the exact "
            "post-checkpoint training stream."
        )
    with config_path.open("r") as f:
        cfg = dict_to_config(yaml.safe_load(f), TrainConfig)

    if seed is None:
        seed = cfg.seed
    if ema == 0:
        ema = None
    if num_samples < -1:
        raise ValueError("num_samples must be -1 or nonnegative")
    if num_tokens < -1:
        raise ValueError("num_tokens must be -1 or nonnegative")
    if num_probes <= 0:
        raise ValueError("num_probes must be positive")
    if flush_every <= 0:
        raise ValueError("flush_every must be positive")
    model = cfg.model
    if model.flash_attention:
        print("Disabling flash attention: Hessian-vector products require JVP support.")
        model = replace(model, flash_attention=False)
    opt = cfg.opt
    if not isinstance(opt, Adam):
        raise TypeError("sample_hessian_frob2 currently requires Adam checkpoints")

    print("Creating Mesh...")
    mesh = jax.make_mesh((jax.device_count(),), ("data",), (AxisType.Explicit,))
    print(mesh)
    jax.set_mesh(mesh)

    print(f"Building train dataset with checkpoint batch size {cfg.batch_size}...")
    dataset = cfg.dataset.build(
        "train",
        batch_size=cfg.batch_size,
        seed=seed,
        shuffle=True,
        repeat=repeat_dataset,
    )
    sharding = jax.NamedSharding(mesh, P("data"))
    dataset = grain.experimental.device_put(dataset, (sharding, None))
    ds_iter = iter(dataset)

    print(f"Loading checkpoint: {checkpoint_dir}")
    params = astype(model.init(jr.key(cfg.seed)), jnp.float32)
    opt_state = opt.init(params, model.lrs)
    ema_cfg = cfg.ema
    ema_state = ema_cfg.init(params)
    nu_ema_state = ema_cfg.init(opt_state.nu)
    with ocp.CheckpointManager(checkpoint_dir.parent) as loader:
        restore_args = dict(
            params=StandardRestore(params),
            opt=StandardRestore(opt_state),
            ds=GrainRestore(ds_iter),
        )
        if ema is not None:
            if not (checkpoint_dir / "ema").exists():
                raise FileNotFoundError(f"EMA state not found in checkpoint: {checkpoint_dir}")
            if not (checkpoint_dir / "nu_ema").exists():
                raise FileNotFoundError(f"nu_ema state not found in checkpoint: {checkpoint_dir}")
            restore_args["ema"] = StandardRestore(ema_state)
            restore_args["nu_ema"] = StandardRestore(nu_ema_state)

        restored = loader.restore(
            checkpoint_step,
            args=ocp.args.Composite(**restore_args),
        )
    params = astype(restored["params"], jnp.float32)
    opt_state = restored["opt"]

    if ema is not None:
        ema_keys = tuple(ema_cfg.pct)
        ema_key = next((pct for pct in ema_keys if abs(float(pct) - ema) < 1e-12), None)
        if ema_key is None:
            raise ValueError(f"EMA {ema} is not available in checkpoint config: {ema_keys}")
        ema_state = astype(restored["ema"], jnp.float32)
        nu_ema_state = restored["nu_ema"]
        params = astype(ema_state.ema[ema_key], jnp.float32)
        opt_state = opt_state._replace(
            nu=jax.tree.map(
                lambda src, n: astype(src, n.dtype),
                nu_ema_state.ema[ema_key],
                opt_state.nu,
            )
        )
    del restored

    start_step = int(np.asarray(opt_state.count))
    if start_step != checkpoint_step:
        print(
            f"Checkpoint dir step {checkpoint_step} restored with optimizer step"
            f" {start_step}"
        )
    if ema is None:
        print(f"Using last-iterate params and nu from optimizer step {start_step}")
    else:
        print(
            f"Using EMA params and EMA nu from optimizer step {start_step}:"
            f" ema={ema:g}"
        )

    remaining_train_batches = max(0, cfg.steps - start_step)
    remaining_train_samples = remaining_train_batches * cfg.batch_size
    if num_samples == -1:
        target_samples = (
            remaining_train_samples
            if num_tokens == -1
            else (num_tokens + cfg.dataset.seq_len - 1) // cfg.dataset.seq_len
        )
    else:
        target_samples = num_samples
    if not allow_past_training_end:
        target_samples = min(target_samples, remaining_train_samples)
    start_sample_index = start_step * cfg.batch_size
    print(
        "Restored training stream:"
        f" start_sample_index={start_sample_index}"
        f" remaining_train_batches={remaining_train_batches}"
        f" remaining_train_samples={remaining_train_samples}"
        f" target_samples={target_samples}"
        f" target_tokens={target_samples * cfg.dataset.seq_len}"
        f" num_probes={num_probes}"
        f" preconditioned={include_adam_precond}"
        f" allow_past_training_end={allow_past_training_end}"
        f" repeat_dataset={repeat_dataset}"
    )

    def unit_nu_like(state):
        return jax.tree.map(lambda x: jnp.ones_like(x, dtype=jnp.float32), state.nu)

    def get_nu_hat(state):
        return jax.lax.cond(
            state.count == 0,
            lambda _: unit_nu_like(state),
            lambda _: astype(opt.get_nu_hat(state), jnp.float32),
            operand=None,
        )

    nu_hat = get_nu_hat(opt_state)

    flat_template, treedef = jax.tree_util.tree_flatten(astype(params, model.dtype))
    n_leaves = len(flat_template)

    def tree_sum_squares(tree):
        leaves = jax.tree.leaves(tree)
        total = jnp.array(0.0, dtype=jnp.float32)
        for leaf in leaves:
            total = total + jnp.sum(leaf.astype(jnp.float32) ** 2)
        return total

    def make_probe(key, like_tree):
        leaves, _ = jax.tree_util.tree_flatten(like_tree)
        keys = jr.split(key, n_leaves)
        probe_leaves = [
            jr.rademacher(k, shape=leaf.shape, dtype=leaf.dtype)
            for k, leaf in zip(keys, leaves)
        ]
        return jax.tree_util.tree_unflatten(treedef, probe_leaves)

    @jax.jit
    def score_fn(p, nu, batch, key):
        x, y = batch
        p_model = astype(p, model.dtype)

        def apply_logits(pp):
            return model(pp, x).astype(jnp.float32)

        logits0, jvp_fn = jax.linearize(apply_logits, p_model)
        q = jax.nn.softmax(logits0, axis=-1)
        token_count = jnp.asarray(y.size, dtype=jnp.float32)

        adam_precond = jax.tree.map(
            lambda n, lr: jnp.sqrt(
                lr.pre * lr.post / (jnp.sqrt(n.astype(jnp.float32)) + opt.eps)
            ),
            nu,
            model.lrs,
        )

        def precond_tree(tree):
            return jax.tree.map(
                lambda x, m: (x.astype(jnp.float32) * m).astype(x.dtype),
                tree,
                adam_precond,
            )

        def selected_hvp_norm2(probe):
            left_probe = precond_tree(probe) if include_adam_precond else probe
            jvp_logits = jvp_fn(left_probe)
            jvp_logits = jvp_logits.astype(jnp.float32)
            centered = jvp_logits - jnp.sum(q * jvp_logits, axis=-1, keepdims=True)
            cotangent = q * centered / token_count
            hvp = astype(jax.linear_transpose(jvp_fn, left_probe)(cotangent)[0], jnp.float32)
            if include_adam_precond:
                hvp = jax.tree.map(lambda h, m: h * m, hvp, adam_precond)
            return tree_sum_squares(hvp)

        probe_keys = jr.split(key, num_probes)

        def one_probe(probe_key):
            probe = make_probe(probe_key, p_model)
            return selected_hvp_norm2(probe)

        def scan_probe(_, probe_key):
            return None, one_probe(probe_key)

        _, values = jax.lax.scan(scan_probe, None, probe_keys)
        logits_flat = logits0.reshape(-1, logits0.shape[-1])
        y_flat = y.reshape(-1)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits_flat, y_flat)
        target_probs = jnp.take_along_axis(
            q.reshape(-1, q.shape[-1]),
            y_flat[:, None],
            axis=-1,
        )[:, 0]
        entropy = -jnp.sum(q * jnp.log(jnp.maximum(q, 1e-30)), axis=-1)
        max_prob = jnp.max(q, axis=-1)

        value_mean = jnp.mean(values)
        value_std = jnp.std(values)
        value_stderr = value_std / jnp.sqrt(jnp.asarray(num_probes, dtype=jnp.float32))

        return dict(
            h_frob2_est=value_mean,
            h_frob2_probe_std=value_std,
            h_frob2_probe_stderr=value_stderr,
            loss=jnp.mean(losses),
            logit_entropy_mean=jnp.mean(entropy),
            max_prob_mean=jnp.mean(max_prob),
            target_prob_mean=jnp.mean(target_probs),
            token_count=token_count,
        )

    if output_file is not None:
        output_path = output_file
    else:
        ema_suffix = ""
        if ema is not None:
            ema_suffix = f"_ema_{ema:g}".replace(".", "p")
        metric_prefix = "hessian_frob2" if include_adam_precond else "hessian_frob2_raw"
        output_path = checkpoint_dir / f"{metric_prefix}{ema_suffix}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing per-sample Hessian estimates to {output_path}")
    with output_path.open("w", newline="") as f:
        writer = None
        scored_samples = 0
        pbar = tqdm(desc="samples", unit="sample", total=target_samples)
        while scored_samples < target_samples:
            batch, _ = next(ds_iter)
            batch_size = len(jax.tree.leaves(batch)[0])
            for batch_idx in range(batch_size):
                if scored_samples >= target_samples:
                    break
                sample_batch = jax.tree.map(lambda x: x[batch_idx : batch_idx + 1], batch)
                sample_key = jr.fold_in(jr.key(probe_seed), start_sample_index + scored_samples)
                scores = jax.device_get(score_fn(params, nu_hat, sample_batch, sample_key))
                row = {
                    "sample_index": start_sample_index + scored_samples,
                    "checkpoint_offset_sample_index": scored_samples,
                    "checkpoint_step": start_step,
                    "num_probes": num_probes,
                    "probe_seed": probe_seed,
                    "ema": 0.0 if ema is None else float(ema),
                    "preconditioned": int(include_adam_precond),
                }
                row.update({k: float(np.asarray(v)) for k, v in scores.items()})
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                scored_samples += 1
                if scored_samples % flush_every == 0:
                    f.flush()
                pbar.update(1)
        pbar.close()

    print(f"Saved {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
