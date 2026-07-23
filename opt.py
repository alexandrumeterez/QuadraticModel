from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import jax
from jax import numpy as jnp
from jaxtyping import PyTree

from util import DType, astype, tree_to_dict


def lerp(a, b, s):
    return jax.tree.map(lambda ai, bi: ai + (bi - ai) * s, a, b)


class EMAState(NamedTuple):
    ema: dict[float, PyTree]
    count: int


@dataclass(frozen=True)
class EMA:
    pct: Sequence[float]
    dtype: DType = "float32"

    def init(self, params):
        params_f32 = astype(params, self.dtype)
        return EMAState(ema={p: params_f32 for p in self.pct}, count=0)

    def update(self, params, state):
        params = astype(params, self.dtype)

        def ema_update(ema, p):
            h = jnp.maximum(1.0, state.count * p / (1 - p))
            return lerp(ema, params, 1 / h)

        ema = {p: ema_update(state.ema[p], p) for p in self.pct}
        return EMAState(ema=ema, count=state.count + 1)


@dataclass(frozen=True)
class CosineSchedule:
    warmup_pct: float = 0.2
    init_value: float = 0.1
    peak_value: float = 1.0
    end_value: float = 0.0
    steps: int | float | None = None

    def __call__(self, t: int | float):
        assert self.steps is not None, "CosineSchedule requires steps to be set"
        t /= self.steps
        warmup = lerp(self.init_value, self.peak_value, t / self.warmup_pct)
        t_cos = jnp.minimum((t - self.warmup_pct) / (1 - self.warmup_pct), 1.0)
        cos = 0.5 * (1 + jnp.cos(jnp.pi * t_cos))
        cosine = lerp(self.end_value, self.peak_value, cos)
        return jnp.where(t < self.warmup_pct, warmup, cosine)


class AdamState(NamedTuple):
    mu: PyTree
    nu: PyTree
    count: int
    log_prod_b2: Any


@dataclass
class Adam:
    lr: float
    b1: float = 0.9
    b2_pct: float = 0.01
    eps: float = 1e-8
    dtype: DType | None = "float32"
    schedule: CosineSchedule = CosineSchedule()

    def init(self, params, param_lrs):
        self.param_lrs = param_lrs
        if self.dtype is not None:
            params = astype(params, self.dtype)
        mu = jax.tree.map(jnp.zeros_like, params)
        nu = jax.tree.map(jnp.zeros_like, params)
        log_prod_b2 = jnp.array(0.0)
        return AdamState(mu=mu, nu=nu, count=0, log_prod_b2=log_prod_b2)

    @staticmethod
    def bias_correct(v, log_prod):
        denom = -jnp.expm1(log_prod)
        denom = jnp.maximum(denom, jnp.array(1e-16, dtype=denom.dtype))
        return jax.tree.map(lambda x: x / denom.astype(x.dtype), v)

    def update(self, grads, state):
        mu, nu, count, log_prod_b2 = state
        count += 1

        base_lr = self.schedule(count) * self.lr
        grads = astype(grads, mu)
        updates = jax.tree.map(lambda g, lr: g * lr.pre, grads, self.param_lrs)

        mu = lerp(mu, updates, 1 - self.b1)

        m = jnp.maximum(20.0, self.b2_pct * count)
        beta2_t = 1.0 - 1.0 / m
        log_prod_b2 = log_prod_b2 + jnp.log(beta2_t).astype(log_prod_b2.dtype)
        nu = lerp(nu, jax.tree.map(jnp.square, updates), 1 - beta2_t)

        log_prod_b1 = count * jnp.log(self.b1)
        mu_hat = self.bias_correct(mu, log_prod_b1)
        nu_hat = self.bias_correct(nu, log_prod_b2)
        updates = jax.tree.map(
            lambda m, n: base_lr * m / (jnp.sqrt(n) + self.eps), mu_hat, nu_hat
        )
        updates = jax.tree.map(lambda g, lr: -g * lr.post, updates, self.param_lrs)
        next_state = AdamState(mu=mu, nu=nu, count=count, log_prod_b2=log_prod_b2)

        eff_lrs = jax.tree.map(lambda lr: lr.pre * lr.post, self.param_lrs)
        eff_lrs = jax.tree.map(
            lambda lr, n: lr / (jnp.sqrt(n) + self.eps), eff_lrs, nu_hat
        )
        dL = jax.tree.map(lambda g, dp: -jnp.vdot(g, dp), grads, updates)
        log = dict(
            base_lr=base_lr,
            beta2=beta2_t,
            mu=tree_to_dict(jax.tree.map(jnp.mean, mu_hat)),
            nu=tree_to_dict(jax.tree.map(jnp.mean, nu_hat)),
            eff_lr=tree_to_dict(jax.tree.map(jnp.mean, eff_lrs)),
            dL=tree_to_dict(dL),
        )
        return updates, next_state, log

    def get_nu_hat(self, state):
        return self.bias_correct(state.nu, state.log_prod_b2)
