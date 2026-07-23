"""Source-condition panels from plot-ready cached curves."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plot_spectrum_summary import finish, paper_style


BATCHES = (1, 64, 1024)
CHECKPOINTS = (10, 50, 100)
FIT_RANGE = (1e4, 1e6)
CURVES = {
    "gn": ("GN", "#1f5f8b", "#6da4d8"),
    "hessian": ("Hessian", "#a4550d", "#f4a261"),
}


def load_curves(path: Path, schedule: str):
    with np.load(path) as z:
        return {
            (batch, pct, kind): {
                field: z[f"{schedule}_B{batch}_P{pct}_{kind}_{field}"]
                for field in ("x", "lo", "hi", "mid")
            }
            for batch in BATCHES
            for pct in CHECKPOINTS
            for kind in CURVES
        }


def load_legacy(path: Path):
    with np.load(path) as z:
        return {
            (batch, 100, kind): {
                field: z[f"B{batch}_{kind}_{field}"]
                for field in ("x", "lo", "hi", "mid")
            }
            for batch in BATCHES
            for kind in CURVES
        }


def draw(ax, curve, kind: str, ylim, fit=False, linewidth=1.45):
    _, edge, fill = CURVES[kind]
    x, lo, hi, mid = (curve[k] for k in ("x", "lo", "hi", "mid"))
    ax.plot(x, mid, color=edge, linewidth=linewidth, zorder=5)
    ax.fill_between(
        x, np.maximum(lo, ylim[0]), np.minimum(hi, ylim[1]),
        color=fill, alpha=0.28, linewidth=0, zorder=1,
    )
    if fit:
        q = np.isfinite(mid) & (mid > 0) & (x >= FIT_RANGE[0]) & (x <= FIT_RANGE[1])
        slope, intercept = np.polyfit(np.log(x[q]), np.log(mid[q]), 1)
        exponent = -slope
        fit_x = np.geomspace(*FIT_RANGE, 100)
        fit_y = np.exp(intercept) * fit_x**slope
        ax.plot(fit_x, fit_y, "k--", linewidth=1, zorder=8)
        text_x = 2.4e5
        text_y = np.exp(intercept) * text_x**slope
        ax.text(text_x, text_y * (1.75 if kind == "gn" else 0.55), rf"$i^{{-{exponent:.2f}}}$", fontsize=8)


def legend(fit=False):
    handles = [
        (Patch(facecolor=fill, alpha=0.28, edgecolor="none"), Line2D([], [], color=edge, linewidth=1.7))
        for _, edge, fill in CURVES.values()
    ]
    labels = ["GN", r"Hessian ($\lambda>0$)"]
    if fit:
        handles.append(Line2D([], [], color="black", linestyle="--", linewidth=1))
        labels.append("power-law fit")
    return handles, labels


def one_by_three(curves, output: Path, name: str, fit=False, legacy=False):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.75), sharex=True, sharey=True)
    values = np.concatenate([curves[b, 100, k]["mid"] for b in BATCHES for k in CURVES])
    values = values[np.isfinite(values) & (values > 0)]
    ylim = (10 ** np.floor(np.log10(values.min())), 10 ** np.ceil(np.log10(values.max())))
    xlim = (
        min(curves[b, 100, k]["x"][0] for b in BATCHES for k in CURVES),
        max(curves[b, 100, k]["x"][-1] for b in BATCHES for k in CURVES),
    )
    for ax, batch in zip(axes, BATCHES):
        ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim)
        for kind in CURVES:
            draw(ax, curves[batch, 100, kind], kind, ylim, fit, 1.55 if legacy else 1.45)
        ax.set_title(f"B={batch}", fontsize=11, fontweight="semibold", pad=7)
        ax.grid(True, which="both", color="#d4d4d4", alpha=0.5, linewidth=0.45)
    handles, labels = legend(fit)
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc="upper center",
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=9, handlelength=2.4)
    fig.supxlabel("estimated eigenvalue index", fontsize=10.5, y=0.02)
    fig.supylabel(r"estimate of $g_i^2/(2\lambda_i)$", fontsize=10, x=0.034)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.23, top=0.81, wspace=0.08)
    finish(fig, output / f"{name}.pdf")


def three_by_three(curves, schedule: str, output: Path):
    ylim = (1e-13, 1e-3) if schedule == "cosine" else (1e-14, 1e-4)
    fig, axes = plt.subplots(3, 3, figsize=(7.8, 7), sharex=True, sharey=True)
    for i, batch in enumerate(BATCHES):
        for j, pct in enumerate(CHECKPOINTS):
            ax = axes[i, j]
            ax.set(xscale="log", yscale="log", xlim=(0.8, 1.25e8), ylim=ylim)
            ax.axvspan(*FIT_RANGE, color="0.5", alpha=0.1, linewidth=0)
            ax.axvline(8192, color="#56B4E9", alpha=0.28, linewidth=0.8)
            for kind in CURVES:
                draw(ax, curves[batch, pct, kind], kind, ylim, True)
            if i == 0:
                ax.set_title(rf"$T={pct}\%$", fontsize=10)
            if j == 0:
                ax.set_ylabel(rf"$B={batch}$" + "\n" + r"$g_i^2/(2\lambda_i)$", fontsize=9)
            ax.grid(True, which="both", color="#d4d4d4", alpha=0.5, linewidth=0.45)
    axes[0, 0].legend(
        [Line2D([], [], color=CURVES[k][1], linewidth=1.6) for k in CURVES],
        ["GN", "Hessian"], frameon=False, loc="lower left", fontsize=8,
    )
    fig.supxlabel("estimated eigenvalue index", fontsize=10, y=0.025)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.09, top=0.95, wspace=0.04, hspace=0.08)
    finish(fig, output / f"source_g2_over_lambda_3x3_powerlaw_{schedule}.pdf")


def plot_all(data_dir: Path, cache_dir: Path, output: Path):
    paper_style()
    cosine = load_curves(data_dir / "source_curves.npz", "cosine")
    constant = load_curves(data_dir / "source_curves.npz", "constant")
    one_by_three(load_legacy(cache_dir / "source_g2_over_lambda_1x3.npz"), output,
                 "source_g2_over_lambda_1x3", legacy=True)
    one_by_three(cosine, output, "source_cosine", fit=True)
    three_by_three(cosine, "cosine", output)
    three_by_three(constant, "constant", output)
