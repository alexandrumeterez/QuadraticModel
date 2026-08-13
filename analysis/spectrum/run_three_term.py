import os
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import rich
import tyro
import yaml
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jax.flatten_util import ravel_pytree
from jax.sharding import AxisType, reshard
from jax.sharding import PartitionSpec as P
from grain.checkpoint import CheckpointRestore as GrainRestore
from orbax.checkpoint import checkpoint_utils
from tqdm.auto import tqdm

from pretrain import TrainConfig
from util import DType, astype, dict_to_config, laxmean


REST_CSV_REL = "hessian_frob2_ema0p04_splits/filtered_data.csv"


def main(
    ckpt: Path,
    cfg: Path | None = None,
    m: int = 300,
    batch_size: int = 8,
    n_tokens: int = 10_000_000,
    ema: float | None = 0.04,
    dtype: DType | None = None,
    curvature: Literal["gn", "hessian"] = "gn",
    preconditioner: Literal["adam", "sgd"] = "adam",
    v0: str = "g",
):
    cfg_path = cfg or ckpt.parent / "config.yaml"
    cfg_dict = yaml.safe_load(cfg_path.open())
    if dtype is not None:
        cfg_dict["model"]["dtype"] = str(jnp.dtype(dtype))
    cfg_dict["model"].update(
        grad_checkpoint=False,
        scan_unroll=True,
        flash_attention=False,
    )
    cfg = dict_to_config(cfg_dict, TrainConfig)
    dtype = jnp.dtype(dtype or cfg.model.dtype)

    jobid = os.environ["SLURM_ARRAY_JOB_ID"]
    taskid = os.environ["SLURM_ARRAY_TASK_ID"]
    output_path = (
        ckpt
        / "spectrum_three_term"
        / f"{curvature}_{preconditioner}_ntok{n_tokens}_m{m}_{dtype}"
        / f"{v0}_{jobid}_{taskid}.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    print(f"output_path={output_path}", flush=True)
    rich.print(cfg)

    n_devices = jax.device_count()
    batch_size *= n_devices
    mesh = jax.make_mesh((n_devices,), ("d",), (AxisType.Explicit,))
    print(mesh)
    jax.set_mesh(mesh)
    data_sharding = jax.NamedSharding(mesh, P("d"))

    data_batch_size = batch_size * (n_tokens // cfg.dataset.seq_len // batch_size)
    rest_csv_path = ckpt / REST_CSV_REL
    train_ds = cfg.dataset.build(
        "train",
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=True,
        repeat=True,
    )
    train_iter = iter(train_ds)

    def load_ckpt():
        p_spec = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32),
            jax.eval_shape(cfg.model.init, jr.key(cfg.seed)),
        )

        def restore(item):
            return ocp.args.PyTreeRestore(
                item=item,
                restore_args=checkpoint_utils.construct_restore_args(item),
                partial_restore=True,
            )

        restore_args = dict(ds=GrainRestore(train_iter))
        if ema is None:
            restore_args["params"] = restore(p_spec)
        else:
            restore_args["ema"] = restore({"ema": {ema: p_spec}})
        if preconditioner == "adam":
            if ema is None:
                restore_args["opt"] = restore({"log_prod_b2": 0.0, "nu": p_spec})
            else:
                restore_args["nu_ema"] = restore({"ema": {ema: p_spec}})
                restore_args["opt"] = restore({"log_prod_b2": 0.0})

        with ocp.CheckpointManager(ckpt.parent) as loader:
            restored = loader.restore(
                int(ckpt.name), args=ocp.args.Composite(**restore_args)
            )
            params = astype(
                restored["params"] if ema is None else restored["ema"]["ema"][ema],
                jnp.float32,
            )
            if preconditioner == "adam":
                nu = (
                    restored["opt"]["nu"]
                    if ema is None
                    else restored["nu_ema"]["ema"][ema]
                )
                nu = astype(nu, jnp.float32)
                nu = cfg.opt.bias_correct(nu, restored["opt"]["log_prod_b2"])

        if preconditioner == "adam":
            precond_lrs = jax.tree.map(
                lambda lr_i, nu_i: lr_i.total / (jnp.sqrt(nu_i) + cfg.opt.eps),
                cfg.model.lrs,
                nu,
            )
            precond_lrs = astype(precond_lrs, jnp.float32)
        else:
            precond_lrs = jax.tree.map(
                lambda lr_i, p_i: jnp.full_like(p_i, lr_i.total, dtype=jnp.float32),
                cfg.model.lrs,
                params,
            )

        p0, unravel = ravel_pytree(params)
        return p0, unravel, ravel_pytree(precond_lrs)[0]

    p0, unravel, precond_lrs = load_ckpt()

    def offsets():
        with rest_csv_path.open() as f:
            header = f.readline().rstrip("\n").split(",")
            i = header.index("checkpoint_offset_sample_index")
            for line in f:
                yield int(line.split(",", i + 1)[i])

    keep = offsets()
    next_keep = next(keep)
    next_offset = 0
    xs = []
    ns = []
    count = 0
    load_pbar = tqdm(total=data_batch_size, desc="filtered samples", unit="seq")
    while count < data_batch_size:
        batch_i, n_i = next(train_iter)
        source_bs = int(jax.tree.leaves(batch_i)[0].shape[0])
        start = next_offset
        stop = start + source_bs
        next_offset = stop
        while next_keep < start:
            next_keep = next(keep)
        idx = []
        while next_keep < stop and count + len(idx) < data_batch_size:
            idx.append(next_keep - start)
            if count + len(idx) < data_batch_size:
                next_keep = next(keep)
        if not idx:
            continue
        idx = np.asarray(idx[: data_batch_size - count], dtype=np.int64)
        batch_i = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), batch_i)
        n_i = np.asarray(jax.device_get(n_i))
        xs.append(jax.tree.map(lambda x: x[idx], batch_i))
        ns.append(n_i[idx])
        count += len(idx)
        load_pbar.update(len(idx))
    load_pbar.close()
    batch = jax.tree.map(lambda *ys: np.concatenate(ys, axis=0), *xs)
    n_bytes = np.concatenate(ns, axis=0)
    batch, _ = jax.device_put((batch, n_bytes), (data_sharding, None))

    print(
        f"Loaded p0/precond_lrs n={len(p0)} curvature={curvature} preconditioner={preconditioner} "
        f"vector_gib={len(p0) * 4 / n_devices / 1024**3:.2f}",
        flush=True,
    )
    precond_sqrt = jnp.sqrt(precond_lrs)
    precond_sqrt.block_until_ready()
    del precond_lrs
    pbar = tqdm(total=m, desc="matvec", unit="call")

    def criterion(pred, y):
        pred = pred.astype(jnp.float32).reshape(-1, pred.shape[-1])
        y = y.reshape(-1)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y).mean()

    def loss_fn(p, batch):
        x, y = batch
        return criterion(apply_fn(p, x), y)

    def apply_fn(p, x):
        return cfg.model(astype(unravel(p), dtype), x)

    def raw_matvec(p, v, batch):
        x, y = batch
        v = v * precond_sqrt
        if curvature == "hessian":
            Hv = jax.jvp(
                jax.grad(lambda p: criterion(apply_fn(p, x), y)), (p,), (v,)
            )[1]
            return Hv * precond_sqrt
        f, jvp_fn = jax.linearize(lambda p: apply_fn(p, x), p)
        Jv = jvp_fn(v)
        HJv = jax.jvp(jax.grad(lambda f: criterion(f, y)), (f,), (Jv,))[1]
        Gv = jax.linear_transpose(jvp_fn, v)(HJv)[0]
        return Gv * precond_sqrt

    def matvec(p, v, batch):
        v = reshard(v, P())
        out = laxmean(lambda b: raw_matvec(p, v, batch=b), batch, batch_size=batch_size)
        jax.debug.callback(lambda: pbar.update(1))
        return reshard(out, P())

    if v0 == "g":

        @jax.jit
        def grad_fn(p, batch):
            g = laxmean(lambda b: jax.grad(loss_fn)(p, b), batch, batch_size=batch_size)
            return g * precond_sqrt

        v = grad_fn(p0, batch=batch)
    else:
        v = jr.normal(jr.key(int(v0.removeprefix("r"))), (len(p0),), dtype=jnp.float32)
    v.block_until_ready()

    def mm(x, y, *, P=None):
        return jnp.dot(x, y, precision=lax.Precision.HIGHEST, out_sharding=P)

    norm = lambda x: jnp.sqrt(mm(x, x, P=P()))

    @jax.jit
    def lanczos_step(p, carry, batch):
        v_prev, v, beta_prev = carry
        w = matvec(p, v, batch)
        w = reshard(w, P("d")) - beta_prev * v_prev
        alpha = mm(v, w, P=P())
        w = w - alpha * v
        beta = norm(w)
        return (v, w / beta, beta), (alpha, beta)

    alphas = np.empty((m,), dtype=np.float32)
    betas = np.empty((m,), dtype=np.float32)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    carry = jnp.zeros_like(v), reshard(v / norm(v), P("d")), jnp.array(0.0, jnp.float32)

    for j in range(m):
        carry, (alpha, beta) = lanczos_step(p0, carry, batch)
        alpha, beta = jax.device_get((alpha, beta))
        alphas[j] = alpha
        betas[j] = beta
        with tmp_path.open("wb") as f:
            np.savez(f, alphas=alphas[: j + 1], betas=betas[: j + 1])
        os.replace(tmp_path, output_path)
    pbar.close()
    print(f"Saved {output_path}", flush=True)


tyro.cli(main)
