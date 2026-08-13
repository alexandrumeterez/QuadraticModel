#!/bin/bash
#SBATCH --job-name=spectrum_three_term
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_grads
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=1450G
#SBATCH --array=0-23

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${DATA_DIR:?set DATA_DIR to the dataset root}"

ckpt="$1"
cfg="${2:-$(dirname "$ckpt")/config.yaml}"
m="${M:-300}"
batch_size="${BASIS_BATCH_SIZE:-8}"
n_tokens="${N_TOKENS:-10000000}"
ema="${EMA:-0.04}"
dtype="${DTYPE:-float32}"
V0S="${V0S:-r0 r1 r2 r3 r4 g}"
CURVATURES="${CURVATURES:-gn hessian gn hessian}"
PRECONDITIONERS="${PRECONDITIONERS:-adam adam sgd sgd}"

TASK_ID="$SLURM_ARRAY_TASK_ID"
JOB_ID="$SLURM_ARRAY_JOB_ID"
LOG_SUBDIR="logs/$(date +%Y_%m_%d)_${JOB_ID}_three_term"
mkdir -p "$LOG_SUBDIR"
cp -n "${BASH_SOURCE[0]}" "$LOG_SUBDIR/run_three_term.sh"
exec >"$LOG_SUBDIR/$TASK_ID.log" 2>&1

read -r -a v0s <<< "$V0S"
read -r -a curvatures <<< "$CURVATURES"
read -r -a preconditioners <<< "$PRECONDITIONERS"
n_variants="${#curvatures[@]}"
v0="${v0s[$(( TASK_ID / n_variants ))]}"
variant_id="$(( TASK_ID % n_variants ))"
curvature="${curvatures[$variant_id]}"
preconditioner="${preconditioners[$variant_id]}"

export PYTHONPATH=.
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-cuda_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TQDM_MININTERVAL=30

PY_ARGS=(
  --ckpt "$ckpt"
  --cfg "$cfg"
  --m "$m"
  --batch-size "$batch_size"
  --n-tokens "$n_tokens"
  --ema "$ema"
  --dtype "$dtype"
  --curvature "$curvature"
  --preconditioner "$preconditioner"
  --v0 "$v0"
)

echo "TASK_ID=$TASK_ID ckpt=$ckpt cfg=$cfg curvature=$curvature preconditioner=$preconditioner n_tokens=$n_tokens m=$m batch_size=$batch_size ema=$ema dtype=$dtype v0=$v0"
printf "PY_ARGS:"
printf " %q" "${PY_ARGS[@]}"
printf "\n"

"$PYTHON_BIN" -u analysis/spectrum/run_three_term.py "${PY_ARGS[@]}"
