import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
import yaml


META_COLUMNS = [
    "sample_index",
    "checkpoint_offset_sample_index",
    "checkpoint_step",
    "ema",
    "preconditioned",
    "loss",
    "logit_entropy_mean",
    "max_prob_mean",
    "target_prob_mean",
    "token_count",
]

EMA_LABEL = "ema0p04"
EMA_SUFFIX = "_ema_0p04"
SPLIT_DIR = "hessian_frob2_ema0p04_splits"


def _closest_checkpoint_step(checkpoint_steps: str, final_step: int, target_pct: float) -> int:
    steps = [int(s) for s in str(checkpoint_steps).split()]
    target = target_pct * final_step
    return min(steps, key=lambda step: abs(step - target))


def _param_count(checkpoint_root: Path) -> int:
    with (checkpoint_root / "config.yaml").open("r") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["model"]["n_params"])


def _probe_path(checkpoint_dir: Path, probe_idx: int) -> Path:
    return checkpoint_dir / f"hessian_frob2{EMA_SUFFIX}_probe{probe_idx}.csv"


def _load_probe_stack(
    checkpoint_dir: Path,
    num_probes: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    first_path = _probe_path(checkpoint_dir, 0)
    if not first_path.exists():
        raise FileNotFoundError(first_path)

    meta = pd.read_csv(first_path, usecols=META_COLUMNS)
    sample_index = meta["sample_index"].to_numpy()
    values = []

    for probe_idx in range(num_probes):
        path = _probe_path(checkpoint_dir, probe_idx)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, usecols=["sample_index", "h_frob2_est"])
        current_index = df["sample_index"].to_numpy()
        if len(current_index) != len(sample_index) or not np.array_equal(
            current_index, sample_index
        ):
            raise ValueError(f"sample_index mismatch in {path}")
        values.append(df["h_frob2_est"].to_numpy(dtype=np.float64))

    return meta, np.stack(values, axis=1)


def _make_average_df(
    meta: pd.DataFrame,
    probe_values: np.ndarray,
    param_count: int,
) -> pd.DataFrame:
    finite = np.isfinite(probe_values)
    finite_count = finite.sum(axis=1)
    safe = np.where(finite, probe_values, 0.0)

    mean = np.full(probe_values.shape[0], np.nan, dtype=np.float64)
    has_probe = finite_count > 0
    mean[has_probe] = safe[has_probe].sum(axis=1) / finite_count[has_probe]

    std = np.full_like(mean, np.nan)
    has_std = finite_count > 1
    centered = np.where(finite, probe_values - mean[:, None], 0.0)
    std[has_std] = np.sqrt(
        (centered[has_std] ** 2).sum(axis=1) / (finite_count[has_std] - 1)
    )
    stderr = std / np.sqrt(finite_count)

    h_frob_norm = np.sqrt(np.maximum(mean, 0.0))
    h_frob_rms_entry = h_frob_norm / param_count

    out = meta.copy()
    out.insert(5, "num_probes", finite_count.astype(np.int16))
    out.insert(6, "h_frob2_mean", mean)
    out.insert(7, "h_frob2_std", std)
    out.insert(8, "h_frob2_stderr", stderr)
    out.insert(9, "h_frob_norm", h_frob_norm)
    out.insert(10, "h_frob_rms_entry", h_frob_rms_entry)
    return out


def _stats(values: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "variance": None,
            "threshold_q99": threshold,
        }
    qs = {
        f"q{int(q * 100):02d}": float(np.quantile(finite, q))
        for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    }
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "variance": float(finite.var()),
        "threshold_q99": float(threshold),
        **qs,
    }


def _write_outputs(
    avg_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    checkpoint_step: int,
    batch_size: int,
    quantile: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    avg_path = output_dir / "scores.csv"
    top_path = output_dir / "outliers.csv"
    rest_path = output_dir / "filtered_data.csv"
    stats_path = output_dir / "stats.json"

    split_values = avg_df["h_frob2_mean"].to_numpy()
    finite_split_values = split_values[np.isfinite(split_values)]
    if finite_split_values.size == 0:
        raise ValueError(f"No finite h_frob2_mean values in {output_dir}")
    threshold = float(np.quantile(finite_split_values, quantile))
    top_mask = avg_df["h_frob2_mean"] >= threshold
    top_df = avg_df.loc[top_mask]
    rest_df = avg_df.loc[~top_mask]

    avg_df.to_csv(avg_path, index=False)
    top_df.to_csv(top_path, index=False)
    rest_df.to_csv(rest_path, index=False)

    stats = {
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "base_batch_size": batch_size,
        "ema": EMA_LABEL,
        "num_rows": int(len(avg_df)),
        "num_top": int(len(top_df)),
        "num_rest": int(len(rest_df)),
        "split_rule": "top if h_frob2_mean >= q99 threshold; rest otherwise",
        "split_metric": "h_frob2_mean",
        "split_quantile": quantile,
        "threshold_h_frob2_mean": threshold,
        "threshold_h_frob_norm": float(np.sqrt(max(threshold, 0.0))),
        "threshold_h_frob_rms_entry": float(
            avg_df.loc[top_mask, "h_frob_rms_entry"].min()
        )
        if len(top_df)
        else None,
        "all_h_frob2_mean": _stats(avg_df["h_frob2_mean"].to_numpy(), threshold),
        "rest_h_frob2_mean": _stats(rest_df["h_frob2_mean"].to_numpy(), threshold),
        "top_h_frob2_mean": _stats(top_df["h_frob2_mean"].to_numpy(), threshold),
        "output_files": {
            "averaged": str(avg_path),
            "top": str(top_path),
            "rest": str(rest_path),
        },
    }
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    return avg_path, top_path, rest_path, stats_path


def main(
    analysis_csv: Path = Path("analysis/analysis_runs.csv"),
    sweep: str = "cosine",
    target_pct: float = 0.8,
    num_probes: int = 10,
    quantile: float = 0.99,
    all_checkpoints: bool = False,
):
    rows = []
    with analysis_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["sweep"] == sweep:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows found for sweep={sweep!r} in {analysis_csv}")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if num_probes <= 0:
        raise ValueError("num_probes must be positive")
    made = []
    for row in rows:
        run_id = row["wandb_run_id"]
        checkpoint_root = Path(row["checkpoint_dir"])
        batch_size = int(row["batch_size"])
        final_step = int(row["final_step"])
        param_count = _param_count(checkpoint_root)
        if all_checkpoints:
            checkpoint_steps = [int(s) for s in str(row["checkpoint_steps"]).split()]
        else:
            checkpoint_steps = [
                _closest_checkpoint_step(row["checkpoint_steps"], final_step, target_pct)
            ]

        for checkpoint_step in checkpoint_steps:
            checkpoint_dir = checkpoint_root / str(checkpoint_step)
            print(f"Processing {run_id} step={checkpoint_step} B={batch_size} {EMA_LABEL}")
            first_probe_path = _probe_path(checkpoint_dir, 0)
            if first_probe_path.exists() and first_probe_path.stat().st_size == 0:
                print(f"Skipping empty probe file: {first_probe_path}")
                continue
            meta, probe_values = _load_probe_stack(checkpoint_dir, num_probes)
            avg_df = _make_average_df(meta, probe_values, param_count)
            output_dir = checkpoint_dir / SPLIT_DIR
            paths = _write_outputs(
                avg_df,
                output_dir,
                run_id,
                checkpoint_step,
                batch_size,
                quantile,
            )
            for path in paths:
                print(path)
            made.extend(paths)

    print(f"Made {len(made)} output files")


if __name__ == "__main__":
    tyro.cli(main)
