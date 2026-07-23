from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from math import sqrt

import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PRNGKeyArray

from ops import apply_rope, get_embedding, rmsnorm, sdpa
from util import RNG, DType, saved_property


@dataclass(frozen=True)
class LR:
    pre: float  # convert to unit std
    post: float  # convert to target std

    @property
    def total(self):
        return self.pre * self.post


@dataclass(frozen=True)
class BlockScales:
    embd: float = 1 / 64
    blocks: float = 1.0
    head: float = 1.0


@dataclass(frozen=True)
class Transformer:
    D: int = 1024
    L: int = 12
    M: int = 4096
    H: int = 16
    K: int = 64
    V: int = 50257
    scan_unroll: int | bool = True
    dtype: DType = "bfloat16"
    norm_dtype: DType | None = "float32"
    norm_eps: float = 1e-6
    rope_dtype: DType | None = "float32"
    rope_freq: int = 10_000
    flash_attention: bool = True
    grad_checkpoint: bool = False
    scales: BlockScales = BlockScales()

    def init(self, key: PRNGKeyArray):
        D, L, M, H, K, V = (getattr(self, k) for k in "DLMHKV")
        rng = RNG(key=key)

        def param(shape, *, std=1.0, init=rng.normal):
            return std * init(shape, dtype=self.dtype)

        params = dict(
            embd=param((V, D), std=1.0),
            blocks=dict(
                attn=dict(
                    q=param((L, D, H, K), std=1 / sqrt(D)),
                    k=param((L, D, H, K), std=1 / sqrt(D)),
                    v=param((L, D, H, K), std=1 / sqrt(D)),
                    head=param((L, H, K, D), std=sqrt(D) / (H * K)),
                ),
                mlp=dict(
                    up=param((L, D, M), std=1 / sqrt(D)),
                    head=param((L, M, D), std=sqrt(D) / M),
                ),
            ),
            head=param((D, V), init=jnp.zeros),
        )
        return params

    @saved_property
    def lrs(self):
        D, L, M, H, K = (getattr(self, k) for k in "DLMHK")
        lrs = dict(
            embd=LR(D, 1),
            blocks=dict(
                attn=dict(
                    q=LR(L * H * K, 1 / D),
                    k=LR(L * H * K, 1 / D),
                    v=LR(L * H * K, 1 / D),
                    head=LR(L * D, 1 / (K * H)),
                ),
                mlp=dict(
                    up=LR(L * M, 1 / D),
                    head=LR(L * D, 1 / M),
                ),
            ),
            head=LR(1, 1 / D),
        )
        scales = jax.tree.broadcast(asdict(self.scales), lrs)
        return jax.tree.map(lambda lr, s: LR(lr.pre, lr.post * s), lrs, scales)

    def __call__(self, p: dict, x: Array) -> Array:
        rope = partial(apply_rope, freq=self.rope_freq, dtype=self.rope_dtype)
        norm = partial(rmsnorm, eps=self.norm_eps, dtype=self.norm_dtype)
        attn_impl = "cudnn" if self.flash_attention else "xla"

        def mlp(h: Array, p: dict[str, Array]) -> Array:
            h = jnp.einsum("...D,DM->...M", h, p["up"])
            h = jax.nn.gelu(h)
            return jnp.einsum("...M,MD->...D", h, p["head"])

        def attn(h: Array, p: dict[str, Array]) -> Array:
            q, k, v = (jnp.einsum("...D,DHK->...HK", h, p[s]) for s in "qkv")
            q, k = rope(norm(q)), rope(norm(k))
            a = sdpa(q, k, v, is_causal=True, implementation=attn_impl)
            return jnp.einsum("...HK,HKD->...D", a, p["head"])

        def block(h: Array, p: dict):
            h += attn(norm(h), p["attn"]) / self.L
            h += mlp(norm(h), p["mlp"]) / self.L
            return h, None

        if self.grad_checkpoint:
            block = jax.checkpoint(block)

        h = get_embedding(p["embd"], x)
        h = lax.scan(block, h, p["blocks"], unroll=self.scan_unroll)[0]
        return jnp.einsum("...TD,DV->...TV", norm(h), p["head"])

    @saved_property
    def n_params(self) -> int:
        p = jax.eval_shape(self.init, jr.key(0))
        return sum(x.size for x in jax.tree_util.tree_leaves(p))

    @saved_property
    def embd_params(self) -> int:
        return 2 * self.D * self.V

    def flops_per_seq(self, seq_len: int) -> int:
        return 6 * self.n_params * seq_len + 12 * self.L * self.D * seq_len**2
