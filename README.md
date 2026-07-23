# Quadratic models of LLM pretraining

This repository contains the 150M-parameter FineWeb training code, the
per-sequence curvature preprocessing used by the experiments, and the
processed inputs needed to reproduce every retained figure.

## Environment

Use Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and pretraining

`DATA_DIR` is the root containing `fineweb_edu_10B_arrayrecord/`.

```bash
export DATA_DIR=/path/to/data
bash fineweb/download_fineweb.sh
python fineweb/parquet_to_ar.py
```

The Slurm sweep reproduces the paper's constant and cosine 3B-token
pretraining grid. Override `DATA_DIR`, `LOG_DIR`, or `PYTHON_BIN` as needed:

```bash
sbatch sweep.sh
```

The retained preprocessing workflow estimates per-sequence preconditioned
Gauss--Newton Frobenius norms and removes the top one percent. Create
`analysis/analysis_runs.csv` with the selected run metadata, then run:

```bash
sbatch --array=0-N preprocessing/run_sample_hessian_frob2.sh
python preprocessing/split_hessian_frob_q99.py
```

Keep the `ds/process_*-of-*.json` item in each selected checkpoint. It stores
the Grain iterator position needed to reconstruct the exact post-checkpoint
training stream.

## Figures

The figures require only Matplotlib, NumPy, and Pandas:

```bash
python analysis/reproduce_figures.py --output-dir reproduced_figures
```

The command checks all 21 retained PDFs. See
[`analysis/README.md`](analysis/README.md) for details.
