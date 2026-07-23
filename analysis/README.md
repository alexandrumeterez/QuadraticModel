# Reproducing the figures

From the repository root, run:

```bash
python analysis/reproduce_figures.py --output-dir reproduced_figures
```

This creates and checks all 21 retained PDF figures.

Everything required is in this directory:

- `reproduce_figures.py` is the only entry point.
- `plot_*.py` contain the five small, figure-specific plotting groups.
- `data/` contains the processed CSV/NPY/NPZ inputs.
- `requirements.txt` records the tested plotting environment.

The reproduction path does not use W&B, checkpoints, training code, or a
private scratch filesystem.
