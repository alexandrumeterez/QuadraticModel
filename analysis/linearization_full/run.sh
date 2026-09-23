#!/bin/bash
#SBATCH --job-name=lin_full
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_adamian_lab
#SBATCH --time=24:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24

CHECKPOINT_PATH="${1:?Usage: run.sh <checkpoint_path> [clip_threshold] [lr_mult] [bsz_mult] [ema]}"
CLIP_THRESHOLD="${2:-inf}"
LR_MULT="${3:-1.0}"
BSZ_MULT="${4:-1.0}"
EMA="${5:-0}"

JOB_GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOG_SUBDIR="$(date +%Y_%m_%d)_${JOB_GROUP_ID}"
mkdir -p "logs/$LOG_SUBDIR"
cp -n "${BASH_SOURCE[0]}" "logs/$LOG_SUBDIR/linearization_full_run.sh"
exec >"logs/$LOG_SUBDIR/$TASK_ID.log" 2>&1

source ~/.bashrc
source "$VENVS/jax/bin/activate"

export DATA_DIR="/n/holylabs/LABS/kempner_adamian_lab/Everyone"
export TQDM_MININTERVAL=30
export PYTHONPATH=.
export WANDB_RUN_GROUP="$JOB_GROUP_ID"

[[ -d "$CHECKPOINT_PATH" ]] || {
  echo "Checkpoint step dir not found: $CHECKPOINT_PATH"
  exit 1
}

TRANSFORMS=(
  none none none none
  prox prox prox prox
  quad quad quad quad
  none none none none
  quad quad quad quad
)
TRAINS=(
  full full full full
  full full full full
  full full full full
  head head head head
  head head head head
)
OPTIMIZERS=(
  adam stale_adam frozen_adam sgdm
  adam stale_adam frozen_adam sgdm
  adam stale_adam frozen_adam sgdm
  adam stale_adam frozen_adam sgdm
  adam stale_adam frozen_adam sgdm
)

TOTAL_JOBS="${#TRANSFORMS[@]}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "TASK_ID $TASK_ID out of range [0, $((TOTAL_JOBS - 1))]"
  exit 1
fi

TRANSFORM="${TRANSFORMS[$TASK_ID]}"
TRAIN="${TRAINS[$TASK_ID]}"
OPTIMIZER="${OPTIMIZERS[$TASK_ID]}"

echo "TASK_ID=$TASK_ID checkpoint=$CHECKPOINT_PATH transform=$TRANSFORM train=$TRAIN optimizer=$OPTIMIZER clip_threshold=$CLIP_THRESHOLD lr_mult=$LR_MULT bsz_mult=$BSZ_MULT ema=$EMA"
python -u analysis/linearization_full/run.py \
  --checkpoint-dir "$CHECKPOINT_PATH" \
  --ema "$EMA" \
  --transform "$TRANSFORM" \
  --train "$TRAIN" \
  --optimizer "$OPTIMIZER" \
  --clip-threshold "$CLIP_THRESHOLD" \
  --lr-mult "$LR_MULT" \
  --bsz-mult "$BSZ_MULT"
