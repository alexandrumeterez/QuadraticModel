"""Plot training dynamics and heatmaps from cached tables."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadratic-model-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd


BATCHES = (1, 64, 1024)
LR_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
BSZ_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def classic_paper_style() -> None:
    plt.style.use("classic")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.linestyle": (0, (1, 4)),
            "grid.color": "0.55",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, path: Path, *, dpi=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(path.resolve())


def linearization_series(data, schedule: str, batch: int, transform: str):
    sub = data[(data["schedule"] == schedule) & (data["base_batch_size"] == batch)]
    specs = {
        "raw": ("tokens_raw", "L0_raw", "L_minus_L0"),
        "prox": ("tokens_prox", "L0_prox", "L_prox_minus_L0"),
        "quad": ("tokens_quad", "L0_quad", "L_quad_minus_L0"),
    }
    x_col, base_col, delta_col = specs[transform]
    result = []
    for _, block in sub.groupby(["checkpoint_run_id", "checkpoint_step"], sort=True):
        block = block.sort_values("eval_index")
        x = block[x_col].to_numpy(float)
        y = block[base_col].to_numpy(float) + block[delta_col].to_numpy(float)
        q = np.isfinite(x) & np.isfinite(y)
        result.append((x[q], y[q]))
    return result


def plot_linearization(data: pd.DataFrame, schedule: str, output_dir: Path) -> None:
    classic_paper_style()
    font_scale = 1.3
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.5), sharex=True)
    styles = {
        "raw": dict(color="black", linestyle="--", linewidth=3.0, alpha=0.9),
        "prox": dict(color="tab:blue", linestyle="-", linewidth=3.0, alpha=0.6),
        "quad": dict(color="tab:orange", linestyle="-", linewidth=3.0, alpha=0.6),
    }
    inset_styles = {key: {**value, "linewidth": 2.2} for key, value in styles.items()}
    constant_inset_ylims = {
        ("prox", 1): (2.59, 2.61),
        ("prox", 64): (2.59, 2.61),
        ("prox", 1024): (2.7, 2.73),
        ("quad", 1): (2.59, 2.61),
        ("quad", 64): (2.59, 2.61),
        ("quad", 1024): (2.7, 2.73),
    }
    for row, transform in enumerate(("prox", "quad")):
        for col, batch in enumerate(BATCHES):
            ax = axes[row, col]
            series = []
            for label in ("raw", transform):
                series.extend((x, y, label) for x, y in linearization_series(data, schedule, batch, label))
            starts = np.array(sorted({float(x[0]) for x, _, _ in series if len(x)}))
            inset_xlim = (starts[-2], starts[-1])
            inset = ax.inset_axes([0.42, 0.48, 0.48, 0.45])
            zoom_values = []
            for x, y, label in series:
                ax.plot(x, y, **styles[label])
                zoom = (x >= inset_xlim[0]) & (x <= inset_xlim[1])
                if zoom.any():
                    inset.plot(x[zoom], y[zoom], **inset_styles[label])
                    zoom_values.append(y[zoom])
                ax.scatter(
                    x[0],
                    y[0],
                    marker=".",
                    s=40 * font_scale,
                    linewidths=font_scale,
                    color=styles[label]["color"],
                    zorder=10,
                )
            inset.set_xlim(*inset_xlim)
            fixed_ylim = constant_inset_ylims.get((transform, batch)) if schedule == "constant" else None
            if fixed_ylim is not None:
                inset.set_ylim(*fixed_ylim)
            else:
                values = np.concatenate(zoom_values)
                low, high = np.nanmin(values), np.nanmax(values)
                pad = 0.08 * (high - low)
                inset.set_ylim(low - pad, high + pad)
            inset.yaxis.tick_right()
            inset.yaxis.set_label_position("right")
            y_formatter = ScalarFormatter(useOffset=False)
            y_formatter.set_scientific(False)
            inset.yaxis.set_major_formatter(y_formatter)
            inset.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e9:g}"))
            inset.xaxis.offsetText.set_visible(False)
            inset.tick_params(axis="both", labelsize=7.5 * font_scale)
            inset.grid(True, alpha=0.5)
            x0, x1 = inset.get_xlim()
            y0, y1 = inset.get_ylim()
            ax.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="gray", facecolor="none", linewidth=1.2, alpha=0.65, zorder=5)
            )
            for xa, xb in ((x0, 0.0), (x1, 1.0)):
                ax.add_artist(
                    ConnectionPatch(
                        xyA=(xa, y1),
                        coordsA=ax.transData,
                        xyB=(xb, 0.0),
                        coordsB=inset.transAxes,
                        axesA=ax,
                        axesB=inset,
                        color="gray",
                        linewidth=1.0,
                        alpha=0.65,
                        zorder=5,
                    )
                )
            if row == 0:
                ax.set_title(f"B={batch}", fontsize=12 * font_scale)
            if col == 0:
                ax.set_ylabel("Validation Loss", fontsize=12 * font_scale)
            if row == 1:
                ax.set_xlabel("Tokens", fontsize=12 * font_scale)
            ax.tick_params(axis="both", labelsize=10 * font_scale, top=False, right=False)
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
            ax.grid(True, alpha=0.85)
            ax.set_ylim((2.55, 3.2) if batch == 1 else (2.55, 3.25) if batch == 64 else (2.6, 5.3))
    handles = [
        Line2D([0], [0], label="LLM", **styles["raw"]),
        Line2D([0], [0], label="Linearized Model", **styles["prox"]),
        Line2D(
            [0],
            [0],
            label="Linearized Model + Quadratic Loss (Taylor Approximation)",
            **styles["quad"],
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
        fontsize=11 * font_scale,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, output_dir / f"{schedule}_linearization_loss_plots.pdf")


def bool_column(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def displayed_bsz_mults(base: int):
    return [
        value
        for value in BSZ_MULTS
        if base * value >= 1 and np.isclose(base * value, round(base * value), atol=1e-9)
    ]


def probability_matrix(block, bsz_mults):
    values = np.full((len(bsz_mults), len(LR_MULTS)), np.nan)
    abort = np.zeros_like(values, dtype=bool)
    for row in block.itertuples(index=False):
        bsz, lr = float(row.bsz_mult), float(row.lr_mult)
        if bsz not in bsz_mults or lr not in LR_MULTS:
            continue
        y, x = bsz_mults.index(bsz), LR_MULTS.index(lr)
        if row.status == "abort" or bool(row.invalid_batch_size):
            abort[y, x] = True
        elif row.status in {"ok", "spike"}:
            values[y, x] = float(row.unstable_fraction)
    return values, abort


def plot_eos_grid(
    csv_path: Path,
    schedule: str,
    output_dir: Path,
    *,
    threshold_label: str | None = None,
    output_name: str | None = None,
) -> None:
    classic_paper_style()
    data = pd.read_csv(csv_path)
    data["invalid_batch_size"] = bool_column(data["invalid_batch_size"])
    table = (
        data[["pretrain_run_id", "base_batch_size", "checkpoint_step"]]
        .drop_duplicates()
        .sort_values(["base_batch_size", "checkpoint_step"])
    )
    bases = sorted(table["base_batch_size"].unique())
    bsz_rows = [displayed_bsz_mults(int(base)) for base in bases]
    fig = plt.figure(figsize=(15, 11.0))
    grid = fig.add_gridspec(
        len(bases),
        3,
        height_ratios=[len(values) for values in bsz_rows],
        width_ratios=[len(LR_MULTS)] * 3,
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.90 if threshold_label is not None else 0.96,
        wspace=0.28,
        hspace=0.55,
    )
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#d0d0d0")
    norm = Normalize(0.0, 1.0, clip=True)
    for r, (base, bsz_mults) in enumerate(zip(bases, bsz_rows)):
        steps = sorted(table.loc[table["base_batch_size"] == base, "checkpoint_step"].unique())
        step_labels = {steps[0]: "10%", steps[1]: "50%", steps[2]: "100%"}
        for c, step in enumerate(steps):
            ax = fig.add_subplot(grid[r, c])
            block = data[(data["base_batch_size"] == base) & (data["checkpoint_step"] == step)]
            values, abort = probability_matrix(block, bsz_mults)
            ax.imshow(values, origin="lower", aspect="equal", interpolation="nearest", cmap=cmap, norm=norm)
            overlay = np.ma.masked_where(~abort, abort)
            ax.imshow(overlay, origin="lower", aspect="equal", interpolation="nearest", cmap=ListedColormap(["black"]), vmin=0, vmax=1)
            ax.set_xticks(range(len(LR_MULTS)), [str(x) for x in LR_MULTS], fontsize=13)
            ax.set_yticks(range(len(bsz_mults)), [str(x) for x in bsz_mults], fontsize=13)
            ax.set_xlabel(r"$\eta$ multiplier", fontsize=15.6)
            ax.set_ylabel(r"$B$ multiplier", fontsize=15.6)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.0)
            ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=13)
            ax.grid(False, which="major")
            ax.set_xticks(np.arange(-0.5, len(LR_MULTS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(bsz_mults), 1), minor=True)
            ax.grid(which="minor", color="0.55", linestyle=(0, (1, 4)), linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
            ax.set_box_aspect(len(bsz_mults) / len(LR_MULTS))
            ax.set_title(rf"$B={int(base)}$ ({step_labels[step]})", fontsize=15.6)
    colorbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax, label="unstable fraction")
    if threshold_label is not None:
        fig.text(
            0.48,
            0.98,
            rf"{schedule.capitalize()} EOS stability: $\max_t\,\mathcal{{L}}_{{\rm lin}} > {threshold_label}$",
            ha="center",
            va="top",
            fontsize=17,
        )
    save(
        fig,
        output_dir
        / (output_name or f"{schedule}_probability_10seed_eta_B_grid.pdf"),
    )


def plot_pretraining_heatmap(csv_path: Path, output_dir: Path) -> None:
    plt.style.use("classic")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
        }
    )
    data = pd.read_csv(csv_path)
    data["final_eval_loss"] = pd.to_numeric(data["final_eval_loss"], errors="coerce")
    finite = data["final_eval_loss"].replace([np.inf, -np.inf], np.nan).dropna()
    batch_sizes = sorted(data["batch_size"].dropna().astype(int).unique())
    etas = sorted(data["eta"].dropna().astype(float).unique())
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#d0d0d0")
    norm = Normalize(float(finite.min()), float(finite.max()), clip=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), constrained_layout=True)
    titles = {
        "constant": "constant: final eval loss, EMA 0.04",
        "cosine": "cosine: final eval loss, EMA 0.04",
    }
    for ax, sweep in zip(axes, ("constant", "cosine")):
        values = np.full((len(etas), len(batch_sizes)), np.nan)
        sub = data[data["sweep"] == sweep]
        for row in sub.itertuples(index=False):
            if np.isfinite(row.final_eval_loss):
                values[etas.index(float(row.eta)), batch_sizes.index(int(row.batch_size))] = row.final_eval_loss
        ax.pcolormesh(
            np.arange(len(batch_sizes) + 1) - 0.5,
            np.arange(len(etas) + 1) - 0.5,
            np.ma.masked_invalid(values),
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="white",
            linewidth=0.8,
            antialiased=False,
        )
        ax.set_title(titles[sweep])
        ax.set_xticks(range(len(batch_sizes)), [str(value) for value in batch_sizes], rotation=45)
        ax.set_yticks(range(len(etas)), [f"{value:g}" for value in etas])
        ax.set_xlabel(r"$B$")
        ax.set_ylabel(r"$\eta$")
        ax.grid(False)
        for y in range(len(etas)):
            for x in range(len(batch_sizes)):
                value = values[y, x]
                if not np.isfinite(value):
                    continue
                red, green, blue, _ = cmap(norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=9, color="black" if luminance > 0.55 else "white")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), shrink=0.9)
    colorbar.set_label("final eval/ema_0.04 loss")
    save(fig, output_dir / "final_eval_ema0p04_loss_by_eta_and_B_notebook.pdf", dpi=200)


def plot_all(data_dir: Path, output_dir: Path) -> None:
    linearization = pd.read_csv(data_dir / "linearization_eval.csv")
    for schedule in ("cosine", "constant"):
        plot_linearization(linearization, schedule, output_dir)
        plot_eos_grid(data_dir / f"eos_{schedule}.csv", schedule, output_dir)
    plot_pretraining_heatmap(data_dir / "pretraining_sweep.csv", output_dir)
