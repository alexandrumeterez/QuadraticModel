"""Large spectrum and eigenvector panels from final paper caches."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

from plot_spectrum_summary import BATCHES, N_PARAMS, VOCAB_SIZE, SpectrumCache, finish


CHECKPOINTS = (10, 50, 100)
SPECTRA = (
    ("gn", "adam", "GN Adam", "#0072B2", "-"),
    ("hessian", "adam", "H Adam", "#D62728", "-"),
    ("gn", "sgd", "GN raw", "#0072B2", "--"),
    ("hessian", "sgd", "H raw", "#D62728", "--"),
)
LAYERS = (
    ("embd", "embed", "#7E57C2"),
    ("blocks.attn.q", "attn q", "#0072B2"),
    ("blocks.attn.k", "attn k", "#56B4E9"),
    ("blocks.attn.v", "attn v", "#009E73"),
    ("blocks.attn.head", "attn out", "#CC79A7"),
    ("blocks.mlp.up", "mlp up", "#E69F00"),
    ("blocks.mlp.head", "mlp down", "#D55E00"),
    ("head", "unembed", "#5f5f5f"),
)
LAYER_COLORS = {key: color for key, _, color in LAYERS}


def panel_style():
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#fbfbfb",
        "axes.grid": True, "grid.alpha": 0.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def yfmt(value, _):
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    return rf"${sign}10^{{{int(np.round(np.log10(abs(value))))}}}$"


def draw_spectrum(ax, curve, color, label=None, linestyle="-", floor=None, alpha=0.12):
    line = dict(color=color, linestyle=linestyle, linewidth=1.75, alpha=0.92,
                dash_capstyle="round", dash_joinstyle="round",
                solid_capstyle="round", solid_joinstyle="round")
    for where in (slice(None, curve["L"]), slice(-curve["R"], None)):
        if (where.stop or where.start):
            x, y = curve["x"][where], curve["y"][where]
            q = np.isfinite(x) & np.isfinite(y)
            if floor is not None:
                q &= y >= floor
            ax.plot(x[q], y[q], **line)
    g = curve["g"]
    q = np.isfinite(g) & np.isfinite(curve["mid"])
    if floor is not None:
        q &= g >= floor
    ax.fill_betweenx(g[q], curve["lo"][q], curve["hi"][q], color=color, alpha=alpha, linewidth=0)
    ax.plot(curve["mid"][q], g[q], label=label, **line)


def curve(cache, batch, pct, curvature, preconditioner, midpoints=None):
    result = cache.curve(batch, pct, curvature, preconditioner)
    if midpoints is not None:
        result["mid"] = midpoints[f"B{batch}_P{pct}_{curvature}_{preconditioner}_mid"]
    return result


def spectrum_grid(cache: SpectrumCache, name: str, output: Path, midpoints=None):
    rows, cols = BATCHES, CHECKPOINTS
    bottom, top = {}, {}
    for batch in rows:
        candidates = []
        for pct in cols:
            for preconditioner in ("adam", "sgd"):
                record = curve(cache, batch, pct, "gn", preconditioner, midpoints)
                positive = record["y"][:record["cut"]]
                if len(positive):
                    candidates.append(max(positive[-1], 1e-7))
        bottom[batch] = min(candidates)
        high = 1e-7
        for pct in cols:
            for curvature, preconditioner, *_ in SPECTRA:
                record = curve(cache, batch, pct, curvature, preconditioner, midpoints)
                for values in (record["y"][:record["cut"]], record["g"]):
                    q = np.isfinite(values) & (values >= bottom[batch])
                    if q.any():
                        high = max(high, values[q].max())
        top[batch] = 10 ** np.ceil(np.log10(high))

    plt.rcParams["grid.alpha"] = 0.2
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.09, top=0.86, wspace=0.05, hspace=0.08)
    for i, batch in enumerate(rows):
        for j, pct in enumerate(cols):
            ax = axes[i, j]
            for curvature, preconditioner, label, color, linestyle in SPECTRA:
                draw_spectrum(ax, curve(cache, batch, pct, curvature, preconditioner, midpoints), color,
                              label if i == j == 0 else None, linestyle, bottom[batch])
            ax.axvline(VOCAB_SIZE, color="0.35", linestyle=":", linewidth=0.85)
            ticks = list(range(int(np.ceil(np.log10(bottom[batch]))), int(np.ceil(np.log10(top[batch]))) + 1, 2))
            if ticks[-1] != int(np.ceil(np.log10(top[batch]))):
                ticks.append(int(np.ceil(np.log10(top[batch]))))
            ax.set(xscale="log", yscale="log", xlim=(0.8, N_PARAMS), ylim=(bottom[batch], top[batch]))
            ax.set_yticks(10 ** np.array(ticks, dtype=float))
            ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
            if i == 0:
                ax.set_title(f"{pct}% through training", fontsize=12, pad=8, fontweight="semibold")
            if j == 0:
                ax.annotate(f"B={batch}", (-0.3, 0.5), xycoords="axes fraction", ha="center",
                            va="center", rotation=90, fontsize=12, fontweight="semibold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.97), fontsize=11, handlelength=2.8)
    fig.supxlabel("estimated eigenvalue index", fontsize=12, y=0.03)
    fig.supylabel("eigenvalue", fontsize=12, x=0.06)
    finish(fig, output / f"{name}.pdf")


def densities(path: Path):
    with np.load(path) as z:
        return {
            (batch, kind): (
                z[f"B{batch}_{kind}_keys"].astype(str).tolist(),
                z[f"B{batch}_{kind}_x"], z[f"B{batch}_{kind}_share"],
            )
            for batch in BATCHES for kind in ("gn", "hessian")
        }


def stack(ax, record, line_width=0.85):
    keys, x, shares = record
    ax.stackplot(x, shares, colors=[LAYER_COLORS[k] for k in keys], linewidth=0, alpha=0.96)
    ax.axvline(VOCAB_SIZE, color="0.35", linestyle=":", linewidth=line_width)
    ax.set(xscale="log", xlim=(0.8, N_PARAMS), ylim=(0, 1), yticks=[0, 0.5, 1])


def evec_panel(schedule: str, cache_dir: Path, output: Path, midpoints=None):
    data = densities(cache_dir / f"evec_2x3{'_constant' if schedule == 'constant' else ''}.npz")
    cosine = schedule == "cosine"
    nrows = 3 if cosine else 2
    fig = plt.figure(figsize=(10.8, 6.4 if cosine else 4.25))
    grid = fig.add_gridspec(
        nrows, 3, height_ratios=[1.35, 0.62, 0.62] if cosine else None,
        left=0.11, right=0.995, bottom=0.14 if cosine else 0.22,
        top=0.84 if cosine else 0.86, wspace=0.06, hspace=0.1,
    )
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(3)] for i in range(nrows)])
    spectra = SpectrumCache(cache_dir / "spectrum_3x3.npz") if cosine else None
    for j, batch in enumerate(BATCHES):
        if cosine:
            ax = axes[0, j]
            for curvature, label, color in (("gn", "GN Adam", "#0072B2"), ("hessian", "H Adam", "#D62728")):
                draw_spectrum(ax, curve(spectra, batch, 100, curvature, "adam", midpoints), color, label)
            ax.axhline(0, color="0.55", linewidth=0.7)
            ax.axvline(VOCAB_SIZE, color="0.35", linestyle=":", linewidth=0.85)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set(xlim=(0.8, N_PARAMS), ylim=(-1e3, 1e3))
            ax.set_yticks([-1e3, -1, -1e-3, -1e-6, 0, 1e-6, 1e-3, 1, 1e3])
            ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
            ax.tick_params(labelbottom=False)
        axes[0, j].set_title(f"B={batch}", fontsize=12, pad=8, fontweight="semibold")
        for row, kind in enumerate(("gn", "hessian"), start=1 if cosine else 0):
            stack(axes[row, j], data[batch, kind])
            if row < nrows - 1:
                axes[row, j].tick_params(labelbottom=False)
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    if cosine:
        axes[0, 0].set_ylabel("eigenvalue", fontsize=12)
        axes[1, 0].set_ylabel("GN evec\nshare", fontsize=10)
        axes[2, 0].set_ylabel("H evec\nshare", fontsize=10)
        lines = [
            Line2D([], [], color=color, linewidth=2, label=label)
            for _, label, color in (
                ("gn", "GN Adam", "#0072B2"),
                ("hessian", "H Adam", "#D62728"),
            )
        ]
        fig.legend(handles=lines, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.965), fontsize=11, handlelength=2.8)
    else:
        axes[0, 0].set_ylabel("GN Ritz-vector\nmass share", fontsize=9.5)
        axes[1, 0].set_ylabel("H Ritz-vector\nmass share", fontsize=9.5)
    patches = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in LAYERS]
    fig.legend(handles=patches, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=8.2, handlelength=1.4, columnspacing=0.9)
    fig.supxlabel("estimated eigenvalue index", fontsize=12, y=0.075 if cosine else 0.105)
    finish(fig, output / f"evec_2x3{'_constant' if not cosine else ''}.pdf")


def alignment_panels(path: Path, output: Path):
    plt.rcParams.update({"grid.alpha": 0.16, "font.size": 8})
    patches = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in LAYERS]
    with np.load(path) as z:
        for schedule in ("cosine", "constant"):
            fig = plt.figure(figsize=(11.2, 11.6))
            grid = fig.add_gridspec(3, 3, left=0.095, right=0.995, bottom=0.115,
                                    top=0.9, wspace=0.08, hspace=0.16)
            for i, batch in enumerate(BATCHES):
                for j, pct in enumerate(CHECKPOINTS):
                    sub = grid[i, j].subgridspec(3, 1, hspace=0.06)
                    for k, (kind, label) in enumerate((("gn", "GN"), ("hessian", "H"), ("neg_hessian", "-H"))):
                        ax = fig.add_subplot(sub[k])
                        prefix = f"{schedule}_B{batch}_P{pct}_{kind}"
                        keys = z[f"{prefix}_keys"].astype(str).tolist()
                        stack(ax, (keys, z[f"{prefix}_x"], z[f"{prefix}_share"]), 0.75)
                        if j:
                            ax.tick_params(labelleft=False)
                        if i < 2 or k < 2:
                            ax.tick_params(labelbottom=False)
                        if j == 0:
                            ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=15, fontsize=8.5)
                        if i == 0 and k == 0:
                            ax.set_title(f"{pct}%", fontsize=12, pad=7, fontweight="semibold")
                        if j == 0 and k == 1:
                            ax.annotate(f"B={batch}", (-0.38, 0.5), xycoords="axes fraction", rotation=90,
                                        ha="center", va="center", fontsize=12, fontweight="semibold")
            fig.legend(handles=patches, loc="lower center", ncol=8, frameon=False,
                       bbox_to_anchor=(0.5, 0.02), fontsize=8.2, handlelength=1.4, columnspacing=0.9)
            fig.supxlabel("estimated eigenvalue index", fontsize=12, y=0.065)
            fig.supylabel("Ritz-vector mass share", fontsize=12, x=0.012)
            fig.suptitle(f"{schedule.title()} eigenvector alignment", fontsize=15, y=0.972)
            finish(fig, output / f"evec_alignment_mega_{schedule}.pdf")


def plot_all(data_dir: Path, cache_dir: Path, output: Path):
    panel_style()
    with np.load(data_dir / "paper_spectrum_midpoints.npz") as midpoints:
        spectrum_grid(SpectrumCache(cache_dir / "spectrum_3x3.npz"), "spectrum_3x3_log", output, midpoints)
        spectrum_grid(SpectrumCache(cache_dir / "spectrum_3x3_constant.npz"), "spectrum_3x3_constant_log", output)
        evec_panel("cosine", cache_dir, output, midpoints)
        evec_panel("constant", cache_dir, output)
        alignment_panels(cache_dir / "evec_alignment_mega.npz", output)
