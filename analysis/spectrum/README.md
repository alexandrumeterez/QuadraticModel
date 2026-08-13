# Lanczos spectrum estimation

`run_basis.py` is the full-reorthogonalization implementation used for the
paper spectra. It supports the Gauss--Newton matrix and full Hessian with two
geometries:

- `adam`: `lr.total / (sqrt(nu_hat) + eps)`
- `sgd`: `lr.total` only (the CompleteP geometry, not the identity)

Both operators are symmetrized as `sqrt(P) @ curvature @ sqrt(P)`. Runs use
EMA 0.04 by default, restore the checkpointed training stream, and select rows
from `hessian_frob2_ema0p04_splits/filtered_data.csv`.

Submit all four curvature/geometry combinations and six starting vectors with:

```bash
export DATA_DIR=/path/to/data
export PYTHON_BIN=.venv/bin/python
sbatch analysis/spectrum/run_basis.sh /path/to/checkpoint
```

`run_three_term.py` is the lower-memory three-term recurrence used for
diagnostics. `run_dist.py` distributes the full basis across multiple Slurm
nodes and accepts `--output-root` for results.
