"""Plot spectrum summaries from processed Gauss--Radau arrays."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadratic-model-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np


N_PARAMS = 167_772_160
VOCAB_SIZE = 8_192
BATCHES = (1, 64, 1024)
COLORS = {1: "#0072B2", 64: "#D55E00", 1024: "#009E73"}


def paper_style() -> None:
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
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


class SpectrumCache:
    def __init__(self, path: Path):
        self.path = path
        self.z = np.load(path)

    def curve(self, batch: int, pct: int, curvature: str, preconditioner: str):
        p = f"B{batch}_P{pct}_{curvature}_{preconditioner}"
        out = {
            name: self.z[f"{p}_{name}"]
            for name in ("x", "y", "dot_x", "g", "lo", "hi", "mid")
        }
        out.update(
            {
                name: int(self.z[f"{p}_{name}"])
                for name in ("L", "R", "cut")
            }
        )
        return out


def positive_scale(curve, anchor: float = 1e6) -> float:
    g = curve["g"]
    x = curve["mid"]
    q = np.isfinite(g) & np.isfinite(x) & (g > 0) & (x > 0)
    order = np.argsort(x[q])
    return float(
        np.exp(
            np.interp(
                np.log(anchor),
                np.log(x[q][order]),
                np.log(g[q][order]),
            )
        )
    )


def draw_positive(
    ax,
    curve,
    *,
    color: str,
    linestyle: str = "-",
    scale: float = 1.0,
    band_alpha: float = 0.10,
):
    line = dict(
        color=color,
        linestyle=linestyle,
        linewidth=1.35,
        solid_capstyle="round",
        dash_capstyle="round",
    )
    locked = curve["L"]
    if locked:
        x = curve["x"][:locked]
        y = curve["y"][:locked] / scale
        q = np.isfinite(x) & np.isfinite(y) & (y > 0)
        ax.plot(x[q], y[q], **line)
    g = curve["g"] / scale
    mid = curve["mid"]
    lo = curve["lo"]
    hi = curve["hi"]
    q = np.isfinite(g) & np.isfinite(mid) & np.isfinite(lo) & np.isfinite(hi) & (g > 0)
    ax.fill_betweenx(g[q], lo[q], hi[q], color=color, alpha=band_alpha, linewidth=0)
    ax.plot(mid[q], g[q], **line)


def combined_positive(curve):
    parts_x = []
    parts_y = []
    if curve["L"]:
        parts_x.append(curve["x"][: curve["L"]])
        parts_y.append(curve["y"][: curve["L"]])
    q = (
        np.isfinite(curve["g"])
        & np.isfinite(curve["mid"])
        & (curve["g"] > 0)
        & (curve["mid"] > 0)
    )
    parts_x.append(curve["mid"][q])
    parts_y.append(curve["g"][q])
    x = np.concatenate(parts_x)
    y = np.concatenate(parts_y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique = np.r_[True, np.diff(x) > 0]
    return x[unique], y[unique]


def finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


def plot_evolution(cache: SpectrumCache, schedule: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.92, 5.16), sharex=True, sharey=True)
    for ax, (preconditioner, title) in zip(
        axes,
        (("adam", "Preconditioned Gauss-Newton"), ("sgd", "Raw Gauss-Newton")),
    ):
        for batch in BATCHES:
            for pct, linestyle in ((50, "-."), (100, "-")):
                draw_positive(
                    ax,
                    cache.curve(batch, pct, "gn", preconditioner),
                    color=COLORS[batch],
                    linestyle=linestyle,
                    band_alpha=0.055,
                )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, N_PARAMS)
        ax.set_ylim(1e-8, 2e1)
        ax.yaxis.set_major_locator(FixedLocator([1e1, 1e0, 1e-2, 1e-4, 1e-6, 1e-8]))
        ax.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
        ax.grid(True, which="both", color="#d4d4d4", alpha=0.55, linewidth=0.45)
    axes[0].legend(
        [Line2D([0], [0], color=COLORS[b], linewidth=1.5) for b in BATCHES],
        [rf"$B={b}$" for b in BATCHES],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    axes[1].legend(
        [
            Line2D([0], [0], color="black", linestyle="-.", linewidth=1.3),
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.3),
        ],
        [r"$T=50\%$", r"$T=100\%$"],
        frameon=False,
        loc="lower left",
        fontsize=8,
    )
    fig.supxlabel("estimated eigenvalue index", fontsize=10, y=0.025)
    fig.supylabel("eigenvalue", fontsize=10, x=0.03)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.12, top=0.96, hspace=0.10)
    finish(fig, output / f"evolution_{schedule}.pdf")


def plot_negative(cache: SpectrumCache, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.84, 2.78))
    for batch in BATCHES:
        for preconditioner, linestyle in (("adam", "-"), ("sgd", "--")):
            c = cache.curve(batch, 100, "hessian", preconditioner)
            line = dict(color=COLORS[batch], linestyle=linestyle, linewidth=1.35)
            if c["R"]:
                x = N_PARAMS + 1 - c["x"][-c["R"] :][::-1]
                y = -c["y"][-c["R"] :][::-1]
                q = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                ax.plot(x[q], y[q], **line)
            q = (
                np.isfinite(c["g"])
                & np.isfinite(c["mid"])
                & np.isfinite(c["lo"])
                & np.isfinite(c["hi"])
                & (c["g"] < 0)
            )
            y = -c["g"][q]
            x = N_PARAMS + 1 - c["mid"][q]
            lo = N_PARAMS + 1 - c["hi"][q]
            hi = N_PARAMS + 1 - c["lo"][q]
            ax.fill_betweenx(y, lo, hi, color=COLORS[batch], alpha=0.08, linewidth=0)
            order = np.argsort(x)
            ax.plot(x[order], y[order], **line)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, N_PARAMS)
    ax.set_ylim(1e-8, 2e2)
    ax.yaxis.set_major_locator(FixedLocator([1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8]))
    ax.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
    ax.set_title(r"$T=100\%$", fontsize=10)
    ax.set_xlabel(r"estimated eigenvalue index of $-H$", fontsize=9.5)
    ax.set_ylabel(r"eigenvalue of $-H$", fontsize=9.5)
    first = ax.legend(
        [Line2D([0], [0], color=COLORS[b], linewidth=1.5) for b in BATCHES],
        [rf"$B={b}$" for b in BATCHES],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    ax.add_artist(first)
    ax.legend(
        [
            Line2D([0], [0], color="#0072B2", linestyle="-", linewidth=1.3),
            Line2D([0], [0], color="#0072B2", linestyle="--", linewidth=1.3),
        ],
        ["Preconditioned", "Raw"],
        frameon=False,
        loc="lower left",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.22, top=0.88)
    finish(fig, output / "negative_evals_cosine.pdf")


def plot_universality(cache: SpectrumCache, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.92, 5.625), sharex=True, sharey=True)
    for ax, (curvature, title, exponent) in zip(
        axes,
        (("gn", "Gauss-Newton", 0.96), ("hessian", "Hessian", 0.65)),
    ):
        for batch in BATCHES:
            for preconditioner, linestyle in (("adam", "-"), ("sgd", "--")):
                c = cache.curve(batch, 100, curvature, preconditioner)
                scale = positive_scale(c)
                draw_positive(
                    ax,
                    c,
                    color=COLORS[batch],
                    linestyle=linestyle,
                    scale=scale,
                    band_alpha=0.055,
                )
        xx = np.geomspace(1e1, 1e8, 200)
        yy = (xx / 1e6) ** (-exponent)
        ax.plot(xx, yy, color="black", linestyle="--", linewidth=1.1, alpha=0.85)
        ax.text(2.3e5, 3.2 if curvature == "gn" else 2.2, rf"$\lambda_i\propto i^{{-{exponent:.2f}}}$", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, N_PARAMS)
        ax.set_ylim(1e-4, 1e6)
        ax.yaxis.set_major_locator(FixedLocator([1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4]))
        ax.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
    axes[0].legend(
        [Line2D([0], [0], color=COLORS[b], linewidth=1.5) for b in BATCHES],
        [rf"$B={b}$" for b in BATCHES],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    axes[1].legend(
        [
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.3),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.3),
        ],
        ["Preconditioned", "Raw"],
        frameon=False,
        loc="lower left",
        fontsize=8,
    )
    fig.supxlabel("estimated eigenvalue index", fontsize=10, y=0.025)
    fig.supylabel("eigenvalue ratio", fontsize=10, x=0.03)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.11, top=0.97, hspace=0.10)
    finish(fig, output / "universality_cosine.pdf")


def plot_token_probabilities(cache: SpectrumCache, token_probs: Path, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.92, 2.84))
    specs = [
        ("gn", 50, r"GN, $T=50\%$", "#0072B2"),
        ("gn", 100, r"GN, $T=100\%$", "#009E73"),
        ("hessian", 50, r"Hessian, $T=50\%$", "#D55E00"),
        ("hessian", 100, r"Hessian, $T=100\%$", "#CC79A7"),
    ]
    for curvature, pct, label, color in specs:
        x, y = combined_positive(cache.curve(1, pct, curvature, "adam"))
        q = (x <= VOCAB_SIZE) & (y > 0)
        x, y = x[q], y[q]
        y = y / float(np.exp(np.interp(0.0, np.log(x), np.log(y))))
        ax.plot(x, y, color=color, linewidth=1.35, label=label)
    probs = np.sort(np.load(token_probs).astype(float))[::-1]
    x = np.arange(1, len(probs) + 1)
    y = np.sqrt(probs / probs[0])
    ax.plot(x, y, color="black", linestyle="--", linewidth=1.25, label=r"$\sqrt{\mathrm{tok\ probs}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, VOCAB_SIZE)
    ax.set_ylim(1e-2, 1.2)
    ax.yaxis.set_major_locator(FixedLocator([1e0, 1e-1, 1e-2]))
    ax.xaxis.set_major_locator(FixedLocator([1e0, 1e1, 1e2, 1e3]))
    ax.set_xlabel("estimated eigenvalue index", fontsize=9.5)
    ax.set_ylabel("eigenvalue ratio", fontsize=9.5)
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.22, top=0.96)
    finish(fig, output / "tok_probs.pdf")


def plot_all(cache_dir: Path, token_probs: Path, output_dir: Path) -> None:
    paper_style()
    cosine = SpectrumCache(cache_dir / "spectrum_3x3.npz")
    constant = SpectrumCache(cache_dir / "spectrum_3x3_constant.npz")
    plot_evolution(cosine, "cosine", output_dir)
    plot_evolution(constant, "constant", output_dir)
    plot_negative(cosine, output_dir)
    plot_universality(cosine, output_dir)
    plot_token_probabilities(cosine, token_probs, output_dir)
