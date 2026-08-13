from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

from quadrature import radau_batch_jax, radau_batch_scipy


def signed_log(x, threshold):
    return np.sign(x) * np.log1p(np.abs(x) / threshold)


def signed_exp(x, threshold):
    return np.sign(x) * threshold * np.expm1(np.abs(x))


def evaluation_grid(low, high, size, threshold):
    epsilon = 3e-8
    if low > 0 and high > 0:
        return np.geomspace(low * (1 + epsilon), high * (1 - epsilon), size)
    if low < 0 and high < 0:
        return -np.geomspace(
            abs(low) * (1 - epsilon), abs(high) * (1 + epsilon), size
        )
    transformed_low, transformed_high = signed_log(
        np.asarray([low, high]), threshold
    )
    margin = epsilon * (transformed_high - transformed_low)
    return signed_exp(
        np.linspace(transformed_low + margin, transformed_high - margin, size),
        threshold,
    )


def nudge_from_ritz(value, ritz_values, relative_distance=3e-8):
    index = np.argmin(np.abs(ritz_values - value))
    epsilon = relative_distance * max(
        abs(ritz_values[index]), abs(value), np.finfo(float).tiny
    )
    if abs(value - ritz_values[index]) < epsilon:
        side = np.sign(value - ritz_values[index]) or -1.0
        value = ritz_values[index] + side * epsilon
    return value


def distance_to_polyline_squared(x, y, polyline_x, polyline_y):
    x0, y0 = polyline_x[:-1][None], polyline_y[:-1][None]
    dx = (polyline_x[1:] - polyline_x[:-1])[None]
    dy = (polyline_y[1:] - polyline_y[:-1])[None]
    x, y = x[:, None], y[:, None]
    weight = np.clip(
        ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0, 1
    )
    return np.min(
        (x - x0 - weight * dx) ** 2 + (y - y0 - weight * dy) ** 2,
        axis=1,
    )


def band_midpoint(values, lower, upper, threshold):
    lower_log = np.log(lower)
    upper_log = np.log(upper)
    values_log = signed_log(values, threshold)
    left, right = lower_log.copy(), upper_log.copy()
    for _ in range(48):
        center = 0.5 * (left + right)
        closer_to_lower = distance_to_polyline_squared(
            center, values_log, lower_log, values_log
        ) < distance_to_polyline_squared(center, values_log, upper_log, values_log)
        left = np.where(closer_to_lower, center, left)
        right = np.where(closer_to_lower, right, center)
    return np.exp(0.5 * (left + right))


def load_lanczos(path):
    with np.load(path) as data:
        alphas = np.asarray(data["alphas"], dtype=float)
        betas = np.asarray(data["betas"], dtype=float)
    if alphas.ndim != 1 or betas.ndim != 1 or len(alphas) != len(betas):
        raise ValueError(f"Expected equally sized one-dimensional alphas/betas in {path}")
    if len(alphas) < 2 or not np.isfinite(alphas).all() or not np.isfinite(betas).all():
        raise ValueError(f"Invalid Lanczos coefficients in {path}")
    return alphas, betas


def spectrum_curve(
    alphas,
    betas,
    curvature,
    num_params,
    rel_tol=1e-6,
    linthresh=1e-8,
    grid_size=400,
    radau_chunk=32,
    backend="scipy",
):
    eigenvalues, eigenvectors = eigh_tridiagonal(alphas, betas[:-1])
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    weights = eigenvectors[0] ** 2
    residuals = betas[-1] * np.abs(eigenvectors[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        locked = residuals / np.abs(eigenvalues) < rel_tol

    left_locked = 0
    while left_locked < len(eigenvalues) and locked[left_locked]:
        left_locked += 1
    right_locked = 0
    while (
        right_locked < len(eigenvalues) - left_locked
        and locked[len(eigenvalues) - 1 - right_locked]
    ):
        right_locked += 1

    positive_semidefinite = curvature == "gn"
    if positive_semidefinite:
        right_locked = 0
        cut = int((eigenvalues > 0).sum())
    else:
        cut = len(eigenvalues)

    rank_weights = num_params * weights / weights.sum()
    if left_locked:
        rank_weights[:left_locked] = 1
    if right_locked:
        rank_weights[-right_locked:] = 1
    if left_locked + right_locked < len(rank_weights):
        stop = len(rank_weights) - right_locked
        rank_weights[left_locked:stop] *= (
            num_params - left_locked - right_locked
        ) / rank_weights[left_locked:stop].sum()

    ranks = np.empty_like(eigenvalues)
    if left_locked:
        ranks[:left_locked] = np.arange(1, left_locked + 1)
    if right_locked:
        ranks[-right_locked:] = num_params - right_locked + np.arange(
            1, right_locked + 1
        )
    if left_locked + right_locked < len(ranks):
        stop = len(ranks) - right_locked
        edges = np.r_[0, np.cumsum(rank_weights[left_locked:stop])]
        lower_rank = left_locked + 0.5 + edges[:-1]
        upper_rank = left_locked + 0.5 + edges[1:]
        ranks[left_locked:stop] = np.sqrt(
            np.maximum(lower_rank, 1e-300) * np.maximum(upper_rank, 1e-300)
        )

    result = {
        "x": ranks,
        "y": eigenvalues,
        "L": np.asarray(left_locked),
        "R": np.asarray(right_locked),
        "cut": np.asarray(cut),
    }
    if left_locked < cut:
        low = 0 if positive_semidefinite else eigenvalues[cut - 1]
        high = eigenvalues[left_locked - 1] if left_locked else eigenvalues[0]
        if high > low:
            grid = evaluation_grid(low, high, grid_size, linthresh)
            grid = np.asarray(
                [nudge_from_ritz(value, eigenvalues) for value in grid]
            )
            bounds = []
            for start in range(0, len(grid), radau_chunk):
                values = grid[start : start + radau_chunk]
                if backend == "jax":
                    bounds.append(
                        np.asarray(
                            radau_batch_jax(
                                alphas,
                                betas,
                                values,
                                np.asarray(left_locked),
                                np.asarray(right_locked),
                                np.asarray(num_params),
                            )
                        )
                    )
                else:
                    bounds.append(
                        radau_batch_scipy(
                            alphas,
                            betas,
                            values,
                            left_locked,
                            right_locked,
                            num_params,
                        )
                    )
            bounds = np.concatenate(bounds)
            lower = np.maximum.accumulate(bounds[:, 0][::-1])[::-1]
            upper = np.minimum.accumulate(bounds[:, 1])
            lower = np.maximum(lower, 1e-12)
            upper = np.maximum(upper, lower)
            midpoint = band_midpoint(grid, lower, upper, linthresh)
            dot_ranks = ranks.copy()
            unlocked = slice(left_locked, cut)
            dot_ranks[unlocked] = np.exp(
                np.interp(
                    signed_log(eigenvalues[unlocked], linthresh),
                    signed_log(grid, linthresh),
                    np.log(midpoint),
                )
            )
            result.update(
                g=grid,
                lo=lower,
                hi=upper,
                mid=midpoint,
                dot_x=dot_ranks,
            )
    result.setdefault("g", np.asarray([]))
    result.setdefault("lo", np.asarray([]))
    result.setdefault("hi", np.asarray([]))
    result.setdefault("mid", np.asarray([]))
    result.setdefault("dot_x", ranks)
    return result


def read_manifest(path):
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"key", "path", "curvature"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain {', '.join(sorted(required))}")
        for row in reader:
            lanczos_path = Path(row["path"]).expanduser()
            if not lanczos_path.is_absolute():
                lanczos_path = path.parent / lanczos_path
            rows.append((row["key"].strip(), lanczos_path, row["curvature"].strip()))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Lanczos coefficients into Gauss--Radau spectrum curves."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--curvature", choices=("gn", "hessian"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-params", type=int, required=True)
    parser.add_argument("--n-tokens", type=int, default=10_000_000)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--linthresh", type=float, default=1e-8)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--radau-chunk", type=int, default=32)
    parser.add_argument("--backend", choices=("scipy", "jax"), default="scipy")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.input and (not args.key or not args.curvature):
        parser.error("--input requires --key and --curvature")
    if args.manifest and (args.key or args.curvature):
        parser.error("--key and --curvature are only valid with --input")
    if (
        args.num_params <= 0
        or args.n_tokens <= 0
        or args.rel_tol <= 0
        or args.linthresh <= 0
        or args.grid_size <= 1
        or args.radau_chunk <= 0
    ):
        parser.error("Numeric arguments must be positive and --grid-size must exceed one")
    return args


def main():
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")
    specs = (
        [(args.key, args.input, args.curvature)]
        if args.input
        else read_manifest(args.manifest)
    )
    if any(not key for key, _, _ in specs):
        raise ValueError("Spectrum keys cannot be empty")
    if len({key for key, _, _ in specs}) != len(specs):
        raise ValueError("Spectrum keys must be unique")

    output = {
        "n_params": np.asarray(args.num_params),
        "n_tokens": np.asarray(args.n_tokens),
        "grid_n": np.asarray(args.grid_size),
        "rel_tol": np.asarray(args.rel_tol),
        "linthresh": np.asarray(args.linthresh),
    }
    lanczos_steps = set()
    for key, path, curvature in specs:
        if curvature not in {"gn", "hessian"}:
            raise ValueError(f"Invalid curvature {curvature!r} for {key}")
        alphas, betas = load_lanczos(path)
        lanczos_steps.add(len(alphas))
        print(f"Processing {key}: {path}", flush=True)
        curve = spectrum_curve(
            alphas,
            betas,
            curvature,
            args.num_params,
            rel_tol=args.rel_tol,
            linthresh=args.linthresh,
            grid_size=args.grid_size,
            radau_chunk=args.radau_chunk,
            backend=args.backend,
        )
        output.update({f"{key}_{name}": value for name, value in curve.items()})
    if len(lanczos_steps) != 1:
        raise ValueError(f"All runs must have the same Lanczos depth: {lanczos_steps}")
    output["m"] = np.asarray(lanczos_steps.pop())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **output)
    os.replace(temporary, args.output)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
