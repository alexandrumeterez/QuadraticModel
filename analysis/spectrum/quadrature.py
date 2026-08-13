from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh_tridiagonal


jax.config.update("jax_enable_x64", True)


@jax.jit
def radau_batch_jax(a, b, lams, left_locked, right_locked, scale):
    off = b[:-1]
    beta = b[-1]
    extended_off = jnp.r_[off, beta]

    def inverse_step(d, values):
        alpha, beta_i = values
        return alpha - lams - beta_i * beta_i / d, None

    d, _ = jax.lax.scan(inverse_step, a[0] - lams, (a[1:], off))
    last_alpha = lams + beta * beta / d
    diagonals = jnp.concatenate(
        [jnp.repeat(a[None, :], lams.shape[0], axis=0), last_alpha[:, None]],
        axis=1,
    )
    eigenvalues = jax.vmap(
        lambda diagonal: jax.scipy.linalg.eigh_tridiagonal(
            diagonal, extended_off, eigvals_only=True
        )
    )(diagonals)

    def weights(diagonal, theta):
        x_prev = jnp.ones_like(theta)
        x = (theta - diagonal[0]) / extended_off[0]
        total = x_prev * x_prev + x * x
        normalizer = jnp.maximum(1.0, jnp.maximum(jnp.abs(x_prev), jnp.abs(x)))
        x_prev /= normalizer
        x /= normalizer
        total /= normalizer * normalizer
        log_scale = jnp.log(normalizer)

        def recurrence(carry, values):
            x_prev, x, total, log_scale = carry
            alpha, beta_prev, beta_next = values
            x_next = ((theta - alpha) * x - beta_prev * x_prev) / beta_next
            normalizer = jnp.maximum(1.0, jnp.maximum(jnp.abs(x), jnp.abs(x_next)))
            return (
                x / normalizer,
                x_next / normalizer,
                (total + x_next * x_next) / (normalizer * normalizer),
                log_scale + jnp.log(normalizer),
            ), None

        (_, _, total, log_scale), _ = jax.lax.scan(
            recurrence,
            (x_prev, x, total, log_scale),
            (diagonal[1:-1], extended_off[:-1], extended_off[1:]),
        )
        return jnp.exp(-jnp.log(total) - 2 * log_scale)

    weights_by_probe = scale * jax.vmap(weights)(diagonals, eigenvalues)
    index = jnp.arange(eigenvalues.shape[1])
    locked = (index < right_locked) | (
        index >= eigenvalues.shape[1] - left_locked
    )
    free = ~locked
    weights_by_probe = jnp.where(locked[None, :], 1.0, weights_by_probe)
    weights_by_probe = jnp.where(
        free[None, :],
        weights_by_probe
        * (scale - left_locked - right_locked)
        / jnp.sum(
            jnp.where(free[None, :], weights_by_probe, 0.0),
            axis=1,
            keepdims=True,
        ),
        weights_by_probe,
    )
    forced_index = jnp.argmin(jnp.abs(eigenvalues - lams[:, None]), axis=1)
    not_forced = index[None, :] != forced_index[:, None]
    above = jnp.sum(
        jnp.where(
            not_forced & (eigenvalues > lams[:, None]), weights_by_probe, 0.0
        ),
        axis=1,
    )
    forced = weights_by_probe[jnp.arange(lams.shape[0]), forced_index]
    return jnp.stack([above, above + forced], axis=1)


def weights_from_eigenvalues(theta, diagonal, off_diagonal):
    x_prev = np.ones_like(theta)
    x = (theta - diagonal[0]) / off_diagonal[0]
    total = x_prev * x_prev + x * x
    normalizer = np.maximum(1.0, np.maximum(np.abs(x_prev), np.abs(x)))
    x_prev /= normalizer
    x /= normalizer
    total /= normalizer * normalizer
    log_scale = np.log(normalizer)
    for i in range(1, len(diagonal) - 1):
        x_next = (
            (theta - diagonal[i]) * x - off_diagonal[i - 1] * x_prev
        ) / off_diagonal[i]
        normalizer = np.maximum(1.0, np.maximum(np.abs(x), np.abs(x_next)))
        total = (total + x_next * x_next) / (normalizer * normalizer)
        x_prev = x / normalizer
        x = x_next / normalizer
        log_scale += np.log(normalizer)
    return np.exp(-np.log(total) - 2 * log_scale)


def radau_batch_scipy(a, b, lams, left_locked, right_locked, scale):
    off = b[: len(a) - 1]
    beta = b[len(a) - 1]
    d = a[0] - lams
    for i in range(1, len(a)):
        d = a[i] - lams - off[i - 1] ** 2 / d
    extended_off = np.r_[off, beta]
    result = []
    for lam, last_alpha in zip(lams, lams + beta**2 / d):
        diagonal = np.r_[a, last_alpha]
        theta = eigh_tridiagonal(
            diagonal,
            extended_off,
            eigvals_only=True,
            check_finite=False,
            lapack_driver="sterf",
        )
        weights = scale * weights_from_eigenvalues(theta, diagonal, extended_off)
        if left_locked or right_locked:
            locked = np.zeros_like(theta, dtype=bool)
            if right_locked:
                locked[:right_locked] = True
            if left_locked:
                locked[-left_locked:] = True
            weights[locked] = 1.0
            weights[~locked] *= (
                scale - left_locked - right_locked
            ) / weights[~locked].sum()
        forced_index = np.argmin(np.abs(theta - lam))
        not_forced = np.ones_like(theta, dtype=bool)
        not_forced[forced_index] = False
        above = weights[not_forced & (theta > lam)].sum()
        result.append((above, above + weights[forced_index]))
    return np.asarray(result)
