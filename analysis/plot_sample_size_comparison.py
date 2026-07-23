"""Plot the 10M-versus-100M token Lanczos sample-size comparison."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadratic-model-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N_PARAMS = 167_772_160
CACHE_SLOTS = {
    (1, 10_000_000): 0,
    (64, 10_000_000): 4,
    (1024, 10_000_000): 8,
    (1, 100_000_000): 0,
    (64, 100_000_000): 1,
    (1024, 100_000_000): 2,
}


def load(cache_dir: Path, batch: int, n_tokens: int):
    slot = CACHE_SLOTS[(batch, n_tokens)]
    matches = sorted(cache_dir.glob(f"gn_adam_ntok{n_tokens}_r0_*_{slot}_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one cache for B={batch}, n_tokens={n_tokens}; found {matches}"
        )
    with np.load(matches[0]) as cache:
        result = {key: cache[key] for key in cache.files}
    for key in ("L", "R", "cut", "pos"):
        result[key] = int(result[key])
    return result


def draw(ax, spectrum, color: str, label: str, floor: float, legend: bool):
    line = dict(
        color=color,
        linewidth=1.75,
        alpha=0.92,
        dash_capstyle="round",
        dash_joinstyle="round",
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    if spectrum["L"]:
        q = (
            np.isfinite(spectrum["x"][: spectrum["L"]])
            & np.isfinite(spectrum["y"][: spectrum["L"]])
            & (spectrum["y"][: spectrum["L"]] >= floor)
        )
        ax.plot(
            spectrum["x"][: spectrum["L"]][q],
            spectrum["y"][: spectrum["L"]][q],
            **line,
        )
    q = (
        np.isfinite(spectrum["grid"])
        & np.isfinite(spectrum["lo"])
        & np.isfinite(spectrum["hi"])
        & np.isfinite(spectrum["mid"])
        & (spectrum["grid"] >= spectrum["mid_min_y"])
        & (spectrum["grid"] >= floor)
    )
    ax.fill_betweenx(
        spectrum["grid"][q],
        spectrum["lo"][q],
        spectrum["hi"][q],
        color=color,
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(
        spectrum["mid"][q],
        spectrum["grid"][q],
        label=label if legend else None,
        **line,
    )


def plot(cache_dir: Path, output_dir: Path) -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.75), sharex=True, sharey=False)
    draws = (
        (10_000_000, "10M tok", "#0072B2"),
        (100_000_000, "100M tok", "#D62728"),
    )
    x_padding = 0.025 * np.log10(N_PARAMS)
    for j, (ax, batch) in enumerate(zip(axes, (1, 64, 1024))):
        spectra = {label: load(cache_dir, batch, n_tokens) for n_tokens, label, _ in draws}
        positive = []
        for spectrum in spectra.values():
            y = spectrum["y"][: spectrum["cut"]]
            positive.append(y[np.isfinite(y) & (y > 0)])
        y = np.concatenate(positive)
        low, high = np.log10(y.min()), np.log10(y.max())
        y_padding = 0.045 * (high - low)
        floor = 10.0 ** (low - y_padding)
        for _, label, color in draws:
            draw(ax, spectra[label], color, label, floor, legend=j == 0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10.0 ** -x_padding, 10.0 ** (np.log10(N_PARAMS) + x_padding))
        ax.set_ylim(floor, 10.0 ** (high + y_padding))
        ax.set_title(f"B={batch}", fontsize=11, fontweight="semibold", pad=7)
        ax.grid(True, which="both", color="#d4d4d4", alpha=0.5, linewidth=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=9,
        handlelength=2.4,
    )
    fig.supxlabel("estimated eigenvalue index", fontsize=10.5, y=0.02)
    fig.supylabel("eigenvalue", fontsize=10.0, x=0.034)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.23, top=0.8, wspace=0.22)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "gn_adam_10m_100m_sample_size_comparison.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())
