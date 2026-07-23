"""Reproduce every paper figure from the committed plotting data."""

import argparse
from pathlib import Path

from plot_paper_panels import plot_all as plot_paper_panels
from plot_sample_size_comparison import plot as plot_sample_size
from plot_source_conditions import plot_all as plot_sources
from plot_spectrum_summary import plot_all as plot_spectrum_summaries
from plot_training_figures import plot_all as plot_training


EXPECTED = {
    "constant_linearization_loss_plots.pdf",
    "constant_probability_10seed_eta_B_grid.pdf",
    "cosine_linearization_loss_plots.pdf",
    "cosine_probability_10seed_eta_B_grid.pdf",
    "evec_2x3.pdf",
    "evec_2x3_constant.pdf",
    "evec_alignment_mega_constant.pdf",
    "evec_alignment_mega_cosine.pdf",
    "evolution_constant.pdf",
    "evolution_cosine.pdf",
    "final_eval_ema0p04_loss_by_eta_and_B_notebook.pdf",
    "gn_adam_10m_100m_sample_size_comparison.pdf",
    "negative_evals_cosine.pdf",
    "source_cosine.pdf",
    "source_g2_over_lambda_1x3.pdf",
    "source_g2_over_lambda_3x3_powerlaw_constant.pdf",
    "source_g2_over_lambda_3x3_powerlaw_cosine.pdf",
    "spectrum_3x3_constant_log.pdf",
    "spectrum_3x3_log.pdf",
    "tok_probs.pdf",
    "universality_cosine.pdf",
}


def main(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "analysis/data"
    cache = data / "cache"
    output.mkdir(parents=True, exist_ok=True)
    plot_training(data, output)
    plot_spectrum_summaries(cache, data / "token_probs.npy", output)
    plot_sources(data, cache, output)
    plot_sample_size(data / "sample_size", output)
    plot_paper_panels(data, cache, output)
    missing = EXPECTED - {path.name for path in output.glob("*.pdf")}
    if missing:
        raise RuntimeError("Missing figures: " + ", ".join(sorted(missing)))
    print(f"Reproduced all {len(EXPECTED)} figures in {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    main(parser.parse_args().output_dir)
