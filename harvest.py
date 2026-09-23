# Lightweight sow/reap utilities for capturing named intermediate values.
# Example: y = sow(x * 2, name="double"); out, saved = reap(f)(x)
# `saved["double"]` contains the captured intermediate.
# Reduced capture: y = sow(x, name="x", reduce=jax.numpy.linalg.norm)
# reaps `saved["x"] = ||x||` while the function still computes on the full `x`.
# Grad: g, saved = reap(jax.grad(f))(x); `saved["x_grad"]` captures `||cotangent||`.
from functools import wraps
from typing import Any, Callable, cast

import jax
import jax.core as jcore
import jax.extend as jex
import jax.lax as lax
from jax._src import core as internal_core
from jax._src.sharding_impls import UNSPECIFIED
from jax.interpreters import ad, batching, mlir

# ── Primitive ─────────────────────────────────────────────────────────────────

sow_p = jex.core.Primitive("sow")
sow_p.def_impl(lambda val, *, name, tag, reduce_jaxpr: val)
sow_p.def_abstract_eval(lambda aval, *, name, tag, reduce_jaxpr: aval)
mlir.register_lowering(
    sow_p, cast(mlir.LoweringRule, lambda ctx, *args, **params: args)
)
batching.primitive_batchers[sow_p] = lambda args, dims, *, name, tag, reduce_jaxpr: (
    sow_p.bind(args[0], name=name, tag=tag, reduce_jaxpr=reduce_jaxpr),
    dims[0],
)
ad.primitive_jvps[sow_p] = lambda p, t, *, name, tag, reduce_jaxpr: (
    sow_p.bind(p[0], name=name, tag=tag, reduce_jaxpr=reduce_jaxpr),
    sow_p.bind(t[0], name=f"{name}_jvp", tag=tag, reduce_jaxpr=reduce_jaxpr),
)
ad.primitive_transposes[sow_p] = lambda ct, val, *, name, tag, reduce_jaxpr: [
    sow_p.bind(
        ct,
        name=name.removesuffix("_jvp") + "_grad",
        tag=tag,
        reduce_jaxpr=reduce_jaxpr,
    )
]

# ── Public API ────────────────────────────────────────────────────────────────


_LEAF_TREEDEF = jax.tree.structure(0)


def _trace_reduce(
    reduce: Callable[[Any], Any] | None, value: Any, name: str
) -> jex.core.ClosedJaxpr | None:
    if reduce is None:
        return None
    if not callable(reduce):
        raise TypeError(
            f"sow(..., name={name!r}, reduce=...) must be a callable unary reducer."
        )

    try:
        closed, out_shape = jax.make_jaxpr(reduce, return_shape=True)(value)
    except Exception as err:
        raise TypeError(
            f"sow(..., name={name!r}, reduce=...) must be a pure JAX-traceable unary function."
        ) from err

    flat_out, treedef = jax.tree.flatten(out_shape)
    if treedef != _LEAF_TREEDEF or len(flat_out) != 1 or len(closed.out_avals) != 1:
        raise TypeError(
            f"sow(..., name={name!r}, reduce=...) must return a single array/scalar value."
        )
    if closed.effects:
        raise TypeError(f"sow(..., name={name!r}, reduce=...) must be effect-free.")
    return closed


def sow(value, name, tag="default", reduce=None):
    reduce_jaxpr = _trace_reduce(reduce, value, name)
    return sow_p.bind(value, name=name, tag=tag, reduce_jaxpr=reduce_jaxpr)


def reap(f, tag: str = "default"):
    def _rewrite_jaxpr(closed):
        new_eqns, extra = [], []
        for eqn in closed.jaxpr.eqns:
            if eqn.primitive is sow_p and eqn.params["tag"] == tag:
                new_eqns.append(eqn)
                reduce_jaxpr = eqn.params.get("reduce_jaxpr")
                if reduce_jaxpr is None:
                    extra.append((eqn.params["name"], eqn.outvars[0]))
                else:
                    reduced = jex.core.Var(reduce_jaxpr.out_avals[0])
                    new_eqns.append(
                        jcore.new_jaxpr_eqn(
                            [eqn.outvars[0]],
                            [reduced],
                            internal_core.closed_call_p,
                            {"call_jaxpr": reduce_jaxpr},
                            reduce_jaxpr.effects,
                            source_info=eqn.source_info,
                            ctx=eqn.ctx,
                        )
                    )
                    extra.append((eqn.params["name"], reduced))
                continue

            params = dict(eqn.params)
            sub = []
            for k, v in eqn.params.items():
                if k == "reduce_jaxpr":
                    continue
                if isinstance(v, jex.core.ClosedJaxpr):
                    params[k], body_extra = _rewrite_jaxpr(v)
                    sub.extend(body_extra)

            if sub and eqn.primitive is lax.scan_p:
                L = eqn.params["length"]
                _stack = lambda v: jex.core.Var(jex.core.unmapped_aval(L, 0, v.aval))
                sub = [(n, _stack(v)) for n, v in sub]

            if sub:
                n_extra = len(sub)
                if "out_shardings" in params:
                    params["out_shardings"] += (UNSPECIFIED,) * n_extra
                if "out_layouts" in params:
                    params["out_layouts"] += (None,) * n_extra

            new_eqns.append(
                eqn.replace(
                    params=params, outvars=list(eqn.outvars) + [v for _, v in sub]
                )
            )
            extra.extend(sub)

        new_jaxpr = closed.jaxpr.replace(
            eqns=new_eqns, outvars=list(closed.jaxpr.outvars) + [v for _, v in extra]
        )
        return jex.core.ClosedJaxpr(new_jaxpr, closed.consts), extra

    @wraps(f)
    def wrapper(*args, **kwargs):
        closed, out_shape = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)
        new_closed, extra = _rewrite_jaxpr(closed)
        flat_inputs = jax.tree.leaves((args, kwargs))
        all_outs = jex.core.jaxpr_as_fun(new_closed)(*flat_inputs)
        flat_out_shape, out_treedef = jax.tree.flatten(out_shape)
        n = len(flat_out_shape)
        f_out = jax.tree.unflatten(out_treedef, all_outs[:n])
        aux = {name: val for (name, _), val in zip(extra, all_outs[n:])}
        return f_out, aux

    return wrapper
