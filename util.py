from dataclasses import is_dataclass
from functools import cached_property, partial
from types import UnionType
from typing import Annotated, Any, Callable, Union, get_args, get_origin, get_type_hints

import jax
import numpy as np
import tyro
from jax import numpy as jnp
from jax.tree_util import register_pytree_node
from jaxtyping import PyTree


def _is_dtype(x):
    try:
        jnp.dtype(x)
        return True
    except Exception:
        return False


DType = Annotated[
    Any,
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="DType",
        instance_from_str=lambda x: jnp.dtype(x[0]),
        is_instance=_is_dtype,
        str_from_instance=lambda instance: [str(jnp.dtype(instance))],
    ),
]


def astype(x: PyTree, target: Any) -> PyTree:
    """Cast a pytree to a dtype or to the dtypes of another pytree."""
    if _is_dtype(target):
        dtype = jnp.dtype(target)
        return jax.tree.map(lambda a: jnp.astype(a, dtype), x)
    return jax.tree.map(lambda a, b: a.astype(b.dtype), x, target)


def tree_to_dict(pytree, sep="."):
    return {
        jax.tree_util.keystr(k, simple=True, separator=sep): v
        for k, v in jax.tree.leaves_with_path(pytree)
    }


def get_max_flops_sec(dtype=jax.numpy.bfloat16):
    devices = jax.devices()
    device_kind = devices[0].device_kind.lower()
    is_fp32 = dtype == jax.numpy.float32

    if "h200" in device_kind or "h100" in device_kind:
        flops_per_device = 67e12 if is_fp32 else 990e12
    elif "a100" in device_kind:
        flops_per_device = 19.5e12 if is_fp32 else 312e12
    else:
        return np.nan

    return flops_per_device * len(devices)


def laxmean(f: Callable, xs: PyTree, batch_size: int):
    n = len(jax.tree.leaves(xs)[0])

    if batch_size <= 0 or n <= batch_size:
        return f(xs)

    assert n % batch_size == 0
    n_batch = n // batch_size

    def _reshape(x):
        return x.reshape(batch_size, n_batch, *x.shape[1:]).swapaxes(0, 1)

    xs = jax.tree.map(_reshape, xs)
    x0 = jax.tree.map(lambda x: x[0], xs)
    out_shape = jax.eval_shape(f, x0)
    init = jax.tree.map(jnp.zeros_like, out_shape)

    def step_fn(carry, b):
        carry = jax.tree.map(lambda a, x: a + x / n_batch, carry, f(b))
        return carry, None

    return jax.lax.scan(step_fn, init, xs)[0]


def dict_to_config(data: Any, cls: Any) -> Any:
    if get_origin(cls) in {Union, UnionType}:
        if not isinstance(data, dict):
            return data
        class_name = data.get("class")
        if class_name is None:
            return data
        for typ in get_args(cls):
            if is_dataclass(typ) and typ.__name__ == class_name:
                payload = dict(data)
                payload.pop("class", None)
                return dict_to_config(payload, typ)
        return data

    if not is_dataclass(cls) or not isinstance(data, dict):
        return data

    hints = get_type_hints(cls)
    init_fields = {f.name for f in cls.__dataclass_fields__.values() if f.init}
    resolved = {}
    for name, typ in hints.items():
        if name not in data or name not in init_fields:
            continue
        value = data[name]
        if is_dataclass(typ):
            value = dict_to_config(value, typ)
        elif get_origin(typ) in {Union, UnionType} and isinstance(value, dict):
            value = dict_to_config(value, typ)
        resolved[name] = value

    return cls(**resolved)


class saved_property(cached_property):
    """Like cached_property, but included in config_to_dict output."""


def config_to_dict(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {"class": type(obj).__name__}
        for f in obj.__dataclass_fields__:
            result[f] = config_to_dict(getattr(obj, f))
        for name in vars(type(obj)):
            if isinstance(getattr(type(obj), name), saved_property):
                result[name] = config_to_dict(getattr(obj, name))
        return result
    elif isinstance(obj, np.dtype):
        return str(jnp.dtype(obj))
    elif isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [config_to_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        values = [config_to_dict(v) for v in obj]
        if hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*values)
        return tuple(values)
    return obj


class RNG:
    def __init__(self, *, seed=None, key=None):
        if seed is not None:
            assert key is None
            self.key = jax.random.PRNGKey(seed)
        else:
            assert key is not None
            self.key = key

    def __call__(self, n_keys=1):
        if n_keys > 1:
            return jax.random.split(self(), n_keys)
        else:
            key, self.key = jax.random.split(self.key)
            return key

    def fork(self, n_forks=1):
        return RNG(key=self(n_forks))

    def __getattr__(self, name):
        return partial(getattr(jax.random, name), self())


register_pytree_node(
    RNG,
    lambda rng: ((rng.key,), None),
    lambda _, c: RNG(key=c[0]),
)
