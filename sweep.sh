#!/bin/bash
#SBATCH --job-name=lr_bs_sweep
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_grads
#SBATCH --time=24:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-69%16

JOB_GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOG_SUBDIR="$(date +%Y_%m_%d)_${JOB_GROUP_ID}"
mkdir -p "logs/$LOG_SUBDIR"
cp -n "${BASH_SOURCE[0]}" "logs/$LOG_SUBDIR/sweep.sh"
exec >"logs/$LOG_SUBDIR/$TASK_ID.log" 2>&1

export DATA_DIR="${DATA_DIR:-/n/netscratch/sham_lab/Everyone/ameterez}"
LOG_DIR="${LOG_DIR:-/n/netscratch/sham_lab/Everyone/ameterez/tf_linearization}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$LOG_DIR"

SWEEPS=(constant cosine)
END_VALUES=(1.0 0.1)
BATCH_SIZES=(1 4 16 64 256 1024 4096)
LEARNING_RATES=(4 2 1 0.5 0.25)

N_BS=${#BATCH_SIZES[@]}
N_LR=${#LEARNING_RATES[@]}
TOTAL_JOBS=$((${#SWEEPS[@]} * N_BS * N_LR))

if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "TASK_ID $TASK_ID out of range [0, $((TOTAL_JOBS - 1))]"
  exit 1
fi

SWEEP_IDX=$((TASK_ID / (N_BS * N_LR)))
BS_IDX=$((TASK_ID % (N_BS * N_LR) / N_LR))
LR_IDX=$((TASK_ID % N_LR))
SWEEP="${SWEEPS[$SWEEP_IDX]}"
END_VALUE="${END_VALUES[$SWEEP_IDX]}"
BATCH_SIZE="${BATCH_SIZES[$BS_IDX]}"
BASE_LR="${LEARNING_RATES[$LR_IDX]}"
LR=$(awk "BEGIN {b=$BATCH_SIZE; printf \"%.6g\", sqrt(b < 64 ? b : 64) * $BASE_LR}")

export TQDM_MININTERVAL=30
export WANDB_RUN_GROUP="$JOB_GROUP_ID"
echo "TASK_ID=$TASK_ID sweep=$SWEEP batch_size=$BATCH_SIZE lr=$LR"

"$PYTHON_BIN" -u pretrain.py \
  --save-checkpoint \
  --n-eval 30 \
  --batch-size "$BATCH_SIZE" \
  --tokens 3000000000 \
  --eval-batch-size 128 \
  --max-eval-tokens 50000000 \
  --ghost-batch-size 16 \
  --ema.pct 0.04 0.08 \
  --log-dir "$LOG_DIR" \
  --dataset.seq-len 1024 \
  --dataset.vocab-size 8192 \
  --opt.lr "$LR" \
  --opt.b1 0.9 \
  --opt.b2-pct 0.01 \
  --opt.eps 1e-8 \
  --opt.dtype float32 \
  --opt.schedule.init-value 0.1 \
  --opt.schedule.peak-value 1.0 \
  --opt.schedule.warmup-pct 0.1 \
  --opt.schedule.end-value "$END_VALUE" \
  model:olmo150m
