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

## Gauss--Radau post-processing

`postprocess.py` converts the `alphas` and `betas` saved by Lanczos into the
rank curves and error bands consumed by the figure code. For one spectrum:

```bash
python analysis/spectrum/postprocess.py \
  --input /path/to/r0_result.npz \
  --key B64_P100_gn_sgd \
  --curvature gn \
  --num-params 167772160 \
  --output spectrum_b64_p100.npz
```

To rebuild a complete figure cache, create a CSV with one row per curve:

```csv
key,path,curvature
B1_P10_gn_adam,/path/to/result.npz,gn
B1_P10_hessian_adam,/path/to/result.npz,hessian
```

Then run:

```bash
python analysis/spectrum/postprocess.py \
  --manifest spectra.csv \
  --num-params 167772160 \
  --output analysis/data/cache/spectrum_3x3.npz \
  --force
```

The default SciPy backend runs on CPU. Pass `--backend jax` to use the JAX
implementation on an accelerator. `--num-params` is `model.n_params` from the
checkpoint's `config.yaml`.
