import json
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
from orbax.checkpoint import checkpoint_utils
from tqdm.auto import tqdm

from pretrain import TrainConfig
from util import astype, dict_to_config, laxmean, tree_to_dict


REST_CSV_REL = "hessian_frob2_ema0p04_splits/filtered_data.csv"
OUTPUT_ROOT = Path("spectrum_results")


def main(
    ckpt: Path,
    cfg: Path | None = None,
    m: int = 300,
    batch_size: int = 8,
    n_tokens: int = 10_000_000,
    ema: float | None = 0.04,
    dtype: str | None = None,
    curvature: Literal["gn", "hessian"] = "gn",
    preconditioner: Literal["adam", "sgd"] = "adam",
    v0: str = "g",
    output_root: Path = OUTPUT_ROOT,
):
    cfg_path = cfg or ckpt.parent / "config.yaml"
    cfg_dict = yaml.safe_load(cfg_path.open())
    if dtype is not None:
        cfg_dict["model"]["dtype"] = dtype
    cfg_dict["model"].update(
        grad_checkpoint=False,
        scan_unroll=True,
        flash_attention=False,
    )
    n_local = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    jax.distributed.initialize(
        cluster_detection_method="slurm",
        local_device_ids=list(range(n_local)),
    )
    root = jax.process_index() == 0

    cfg = dict_to_config(cfg_dict, TrainConfig)
    dtype = jnp.dtype(dtype or cfg.model.dtype)
    rest_csv_path = ckpt / REST_CSV_REL
    train_ds = cfg.dataset.build(
        "train",
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=True,
        repeat=True,
    )
    train_iter = iter(train_ds)
    train_iter.set_state(json.loads((ckpt / "ds/process_0-of-1.json").read_text()))
    train_iter.start_prefetch()

    jobid = os.environ["SLURM_ARRAY_JOB_ID"]
    taskid = os.environ["SLURM_ARRAY_TASK_ID"]
    ema_name = "RAW" if ema is None else f"EMA{ema:g}".replace(".", "p")
    output_path = (
        output_root
        / ckpt.parent.name
        / ckpt.name
        / f"{curvature}_{preconditioner}_{ema_name}_ntok{n_tokens}_m{m}_{dtype}"
        / f"{v0}_{jobid}_{taskid}_{preconditioner}.npz"
    )
    if root:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(output_path)
        print(f"output_path={output_path}", flush=True)
        rich.print(cfg)

    n_devices = jax.device_count()
    batch_size *= n_devices
    mesh = jax.make_mesh((n_devices,), ("d",), (AxisType.Explicit,))
    if root:
        print(mesh, flush=True)
        print(
            f"process_count={jax.process_count()} local_device_count={jax.local_device_count()} device_count={n_devices}",
            flush=True,
        )
    jax.set_mesh(mesh)
    data_sharding = jax.NamedSharding(mesh, P("d"))

    data_batch_size = batch_size * (n_tokens // cfg.dataset.seq_len // batch_size)

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

        restore_args = {}
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
    load_pbar = tqdm(total=data_batch_size, desc="filtered samples", unit="seq") if root else None
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
        if root:
            load_pbar.update(len(idx))
    if root:
        load_pbar.close()
    batch = jax.tree.map(lambda *ys: np.concatenate(ys, axis=0), *xs)
    n_bytes = np.concatenate(ns, axis=0)
    batch, _ = jax.device_put((batch, n_bytes), (data_sharding, None))

    if root:
        print(
            f"Loaded p0/precond_lrs n={len(p0)} curvature={curvature} preconditioner={preconditioner} "
            f"shard_basis_gib={len(p0) * m * 4 / n_devices / 1024**3:.2f}",
            flush=True,
        )
    precond_sqrt = jnp.sqrt(precond_lrs)
    precond_sqrt.block_until_ready()
    del precond_lrs
    pbar = tqdm(total=m, desc="matvec", unit="call") if root else None

    def criterion(pred, y):
        pred = pred.astype(jnp.float32).reshape(-1, pred.shape[-1])
        y = y.reshape(-1)
        return optax.softmax_cross_entropy_with_integer_labels(pred, y).mean()

    def loss_fn(p, batch):
        x, y = batch
        x = reshard(x, P("d"))
        y = reshard(y, P("d"))
        return criterion(apply_fn(p, x), y)

    def apply_fn(p, x):
        return cfg.model(astype(unravel(p), dtype), x)

    def raw_matvec(p, v, batch):
        x, y = batch
        x = reshard(x, P("d"))
        y = reshard(y, P("d"))
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
        return reshard(out, P())

    def mm(x, y, *, P=None):
        return jnp.dot(x, y, precision=lax.Precision.HIGHEST, out_sharding=P)

    norm = lambda x: jnp.sqrt(mm(x, x, P=P()))

    save_kwargs = {}
    if v0 == "g":

        @jax.jit
        def grad_fn(p, batch):
            g = laxmean(lambda b: jax.grad(loss_fn)(p, b), batch, batch_size=batch_size)
            return g * precond_sqrt

        v = grad_fn(p0, batch=batch)
    else:
        v = jr.normal(
            jr.key(int(v0.removeprefix("r"))),
            (len(p0),),
            dtype=jnp.float32,
            out_sharding=P("d"),
        )
    v.block_until_ready()
    if v0 == "g":
        g_norm = np.asarray(jax.device_get(norm(v)), dtype=np.float32)
        save_kwargs["g_norm"] = g_norm
        if root:
            print(f"g_norm={g_norm:.8e}", flush=True)

    @jax.jit
    def lanczos_init(v):
        V = jnp.zeros((m, len(p0)), dtype=jnp.float32, out_sharding=P(None, "d"))
        v = reshard(v / norm(v), P("d"))
        v = reshard(v[None, :], P(None, "d"))
        return lax.dynamic_update_slice(V, v, (0, 0))

    @jax.jit(donate_argnames=("V",))
    def lanczos_step(p, V, beta_prev, j, batch):
        def reorth(w, end):
            def proj(i, w):
                v = V[i]
                return w - v * mm(v, w, P=P())

            w = lax.fori_loop(0, end, proj, w)
            return lax.fori_loop(0, end, proj, w)

        v = reshard(V[j], P())
        w = matvec(p, v, batch)
        Aw_norm = norm(w)
        alpha = mm(v, w, P=P())
        w = reshard(w, P("d"))
        w = w - alpha * V[j]
        w = lax.cond(j > 0, lambda: w - beta_prev * V[j - 1], lambda: w)
        w = reorth(w, j + 1)
        beta = norm(w)
        tol = 100.0 * jnp.finfo(jnp.float32).eps * jnp.maximum(Aw_norm, 1.0)
        v_next, beta = lax.cond(
            beta > tol,
            lambda: (w / beta, beta),
            lambda: (jnp.zeros_like(w), jnp.zeros_like(beta)),
        )

        def set_next():
            v = reshard(v_next[None, :], P(None, "d"))
            return lax.dynamic_update_slice(V, v, (j + 1, 0))

        V = lax.cond(j + 1 == m, lambda: V, set_next)
        return V, alpha, beta

    @jax.jit
    def lanczos_finish(V, alphas, betas):
        betas = betas[:-1]
        T = jnp.diag(alphas) + jnp.diag(betas, 1) + jnp.diag(betas, -1)
        _, U = jnp.linalg.eigh(T)
        U = U[:, ::-1]

        def get_mass(u):
            def add_col(j, evec):
                return evec + u[j] * V[j]

            evec = jnp.zeros((len(p0),), dtype=jnp.float32, out_sharding=P("d"))
            evec = lax.fori_loop(0, m, add_col, evec)
            return jax.tree.map(lambda x: jnp.sum(x**2), unravel(reshard(evec, P())))

        masses = lax.map(get_mass, U.T)
        return tree_to_dict(masses)

    V = lanczos_init(v)
    beta_prev = jnp.array(0.0, dtype=jnp.float32)
    alphas = np.empty((m,), dtype=np.float32)
    betas = np.empty((m,), dtype=np.float32)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    for j in range(m):
        V, alpha, beta_prev = lanczos_step(p0, V, beta_prev, j, batch)
        alphas[j], betas[j] = jax.device_get((alpha, beta_prev))
        if root:
            with tmp_path.open("wb") as f:
                np.savez(f, alphas=alphas[: j + 1], betas=betas[: j + 1], **save_kwargs)
            os.replace(tmp_path, output_path)
            pbar.update(1)
    masses = lanczos_finish(V, alphas, betas)
    if root:
        pbar.close()
        np.savez(
            output_path,
            alphas=alphas,
            betas=betas,
            **save_kwargs,
            **{f"masses.{mk}": np.asarray(mv) for mk, mv in masses.items()},
        )
        print(f"Saved {output_path}", flush=True)


tyro.cli(main)
