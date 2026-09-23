#!/bin/bash
#SBATCH --job-name=eos
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_grads
#SBATCH --time=24:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24

set -euo pipefail

REST_SAMPLE_CSV_REL="${REST_SAMPLE_CSV_REL:-${REST_G2_CSV_REL:-sample_g2_over_nu_splits/sample_g2_over_nu_rest_below_top1pct_any_weight.csv}}"
LINEARIZATION_EMA="${LINEARIZATION_EMA:-}"
LINEARIZATION_TRANSFORM="${LINEARIZATION_TRANSFORM:-quad}"

usage() {
  echo "Usage: run_eos.sh <multiplier_pair_csv> <checkpoint_path1> [checkpoint_path2 ...]"
}

append_csv_values() {
  local csv="$1"
  local -n out="$2"
  local parts=()
  IFS=',' read -r -a parts <<< "$csv"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] || continue
    out+=("$part")
  done
}

MULTIPLIER_PAIR_CSV="${1:-}"

if [[ -z "$MULTIPLIER_PAIR_CSV" ]]; then
  usage >&2
  exit 1
fi
shift

(( $# > 0 )) || {
  usage >&2
  exit 1
}

CHECKPOINT_PATHS=("$@")
MULTIPLIER_PAIRS=()
append_csv_values "$MULTIPLIER_PAIR_CSV" MULTIPLIER_PAIRS

(( ${#MULTIPLIER_PAIRS[@]} > 0 )) || {
  echo "No multiplier pairs provided" >&2
  exit 1
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STEP_COUNT="${#CHECKPOINT_PATHS[@]}"
PAIR_COUNT="${#MULTIPLIER_PAIRS[@]}"
TOTAL_JOBS=$(( STEP_COUNT * PAIR_COUNT ))

if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "TASK_ID $TASK_ID out of range [0, $((TOTAL_JOBS - 1))]" >&2
  exit 1
fi

JOB_GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
LOG_SUBDIR="$(date +%Y_%m_%d)_${JOB_GROUP_ID}"
mkdir -p "logs/$LOG_SUBDIR"
cp -n "${BASH_SOURCE[0]}" "logs/$LOG_SUBDIR/linearization_full_run_eos.sh" 2>/dev/null || true
exec >"logs/$LOG_SUBDIR/$TASK_ID.log" 2>&1

HAD_NOUNSET=0
case $- in
  *u*) HAD_NOUNSET=1 ;;
esac
set +u
source ~/.bashrc
if (( HAD_NOUNSET )); then
  set -u
fi
if [[ -n "${VENVS:-}" && -f "$VENVS/jax/bin/activate" ]]; then
  source "$VENVS/jax/bin/activate"
else
  source ~/pax/bin/activate
fi

export DATA_DIR="${LINEARIZATION_DATA_DIR:-/n/holylabs/LABS/kempner_adamian_lab/Everyone}"
export TQDM_MININTERVAL=30
export PYTHONPATH=.
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$JOB_GROUP_ID}"

PAIR_INDEX=$(( TASK_ID % PAIR_COUNT ))
STEP_INDEX=$(( TASK_ID / PAIR_COUNT ))

CHECKPOINT_PATH="${CHECKPOINT_PATHS[$STEP_INDEX]}"
CHECKPOINT_ROOT="$(dirname "$CHECKPOINT_PATH")"
SELECTED_STEP="$(basename "$CHECKPOINT_PATH")"
PAIR_SPEC="${MULTIPLIER_PAIRS[$PAIR_INDEX]}"
if [[ "$PAIR_SPEC" != *:* ]]; then
  echo "Invalid multiplier pair: $PAIR_SPEC" >&2
  exit 1
fi
LR_MULT="${PAIR_SPEC%%:*}"
BSZ_MULT="${PAIR_SPEC#*:}"
TRANSFORM="$LINEARIZATION_TRANSFORM"
TRAIN="full"
OPTIMIZER="sgdm"

[[ -d "$CHECKPOINT_PATH" ]] || {
  echo "Checkpoint step dir not found: $CHECKPOINT_PATH" >&2
  exit 1
}
REST_SAMPLE_CSV="$CHECKPOINT_PATH/$REST_SAMPLE_CSV_REL"
[[ -s "$REST_SAMPLE_CSV" ]] || {
  echo "Required rest-sample CSV not found or empty: $REST_SAMPLE_CSV" >&2
  exit 1
}

read -r BASE_BATCH_SIZE TARGET_BATCH_SIZE VALID_BATCH_SIZE < <(
  python - "$CHECKPOINT_ROOT/config.yaml" "$BSZ_MULT" <<'PY'
import math
import sys

import yaml

config_path = sys.argv[1]
bsz_mult = float(sys.argv[2])
with open(config_path) as f:
    cfg = yaml.safe_load(f)
base_batch_size = int(cfg["batch_size"])
target = base_batch_size * bsz_mult
valid = target >= 1 and math.isclose(target, round(target), rel_tol=0.0, abs_tol=1e-9)
print(base_batch_size, target, int(valid))
PY
)

if [[ "$VALID_BATCH_SIZE" != "1" ]]; then
  echo "Skipping non-integer/invalid batch size: base_batch_size=$BASE_BATCH_SIZE bsz_mult=$BSZ_MULT target_batch_size=$TARGET_BATCH_SIZE"
  exit 0
fi

CMD=(
  python -u analysis/linearization_full/run.py
  --checkpoint-dir "$CHECKPOINT_PATH" \
  --transform "$TRANSFORM" \
  --train "$TRAIN" \
  --optimizer "$OPTIMIZER" \
  --lr-mult "$LR_MULT" \
  --bsz-mult "$BSZ_MULT" \
  --rest-csv-rel "$REST_SAMPLE_CSV_REL"
)
if [[ -n "$LINEARIZATION_EMA" ]]; then
  CMD+=(--ema "$LINEARIZATION_EMA")
fi

echo "TASK_ID=$TASK_ID checkpoint_root=$CHECKPOINT_ROOT selected_step=$SELECTED_STEP checkpoint=$CHECKPOINT_PATH rest_sample_csv=$REST_SAMPLE_CSV transform=$TRANSFORM train=$TRAIN optimizer=$OPTIMIZER lr_mult=$LR_MULT bsz_mult=$BSZ_MULT ema=${LINEARIZATION_EMA:-last} wandb_group=$WANDB_RUN_GROUP"
"${CMD[@]}"
