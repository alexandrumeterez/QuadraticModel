import jax
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array


def apply_rope(
    x: Array, *, freq: float, pos: Array | None = None, dtype: jnp.dtype | None = None
) -> Array:
    dtype = dtype or x.dtype
    out = x.astype(dtype)
    T, _, K = out.shape[-3:]
    assert K % 2 == 0, f"apply_rope expects an even head dimension, got {K}"
    half_K = K // 2
    idx = jnp.arange(half_K)[None, None, :]
    base = freq ** (-idx / half_K)
    if pos is None:
        pos = jnp.arange(T)
    pos = pos.astype(dtype)[..., None, None]
    theta_cis = jnp.exp(1j * (pos * base))
    out = out[..., :half_K] + 1j * out[..., half_K:]
    out *= theta_cis
    return jnp.concatenate([out.real, out.imag], axis=-1, dtype=x.dtype)


def rmsnorm(x: Array, eps: float, dtype: jnp.dtype | None = None) -> Array:
    dtype = dtype or x.dtype
    out = x.astype(dtype)
    out *= lax.rsqrt(jnp.mean(out**2, axis=-1, keepdims=True) + eps)
    return out.astype(x.dtype)


def get_embedding(w: Array, x: Array) -> Array:
    out_spec = jax.typeof(x).sharding
    w_dtype = w.dtype
    if w_dtype in [jnp.bfloat16, jnp.float16]:
        w = w.astype(jnp.float32)
    out = w.at[x].get(out_sharding=out_spec)
    return out.astype(w_dtype)


sdpa = jax.nn.dot_product_attention
