#!/bin/bash
#SBATCH --job-name=hess_frob2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_grads
#SBATCH --time=24:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24

set -euo pipefail

usage() {
  echo "Usage:"
  echo "  sbatch --array=0-\$((N_TASKS - 1))%64 preprocessing/run_sample_hessian_frob2.sh [analysis_runs.csv] [num_samples] [num_probe_jobs] [target_pct|all] [probe_index_offset]"
  echo
  echo "Defaults:"
  echo "  analysis_runs.csv: analysis/analysis_runs.csv"
  echo "  num_samples: -1, meaning use NUM_TOKENS from the post-checkpoint training stream"
  echo "  num_probe_jobs: 10 means ten independent one-probe jobs per checkpoint"
  echo "  target_pct: all, meaning run every checkpoint step listed in the CSV"
  echo "  probe_index_offset: 0; set to 5 to write probe5.. after an existing probe0..4 run"
  echo
  echo "Environment:"
  echo "  SWEEP_FILTER is empty by default, meaning include every sweep in the CSV"
  echo "  EMA=0.04 starts from the EMA params and matching EMA preconditioner"
  echo "  PREPROCESSING_DATA_DIR=/n/netscratch/sham_lab/Everyone/ameterez"
  echo "  NUM_TOKENS=400000000"
  echo "  MAX_TASKS is controlled by the sbatch array % limit"
  echo "  INCLUDE_ADAM_PRECOND=1 uses the EMA Adam preconditioner when EMA=0.04"
  echo "  SKIP_EXISTING=1 to exit successfully when the output CSV already exists"
  echo "  ALLOW_PAST_TRAIN_END=1 to score the requested horizon past cfg.steps"
  echo "  REPEAT_DATASET=0 to avoid cycling the dataset when scoring past cfg.steps"
}

ANALYSIS_CSV="${1:-analysis/analysis_runs.csv}"
NUM_SAMPLES="${2:--1}"
NUM_PROBE_JOBS="${3:-10}"
TARGET_PCT="${4:-all}"
PROBE_INDEX_OFFSET="${5:-0}"
SWEEP_FILTER="${SWEEP_FILTER:-}"
EMA="${EMA:-0.04}"
NUM_TOKENS="${NUM_TOKENS:-400000000}"
PROBE_SEED_BASE="${PROBE_SEED_BASE:-0}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"
INCLUDE_ADAM_PRECOND="${INCLUDE_ADAM_PRECOND:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
ALLOW_PAST_TRAIN_END="${ALLOW_PAST_TRAIN_END:-1}"
REPEAT_DATASET="${REPEAT_DATASET:-0}"

export DATA_DIR="${PREPROCESSING_DATA_DIR:-/n/netscratch/sham_lab/Everyone/ameterez}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-30}"
export PYTHONPATH="${PYTHONPATH:-.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

mapfile -t TASK_INFO < <("$PYTHON_BIN" - "$ANALYSIS_CSV" "$TASK_ID" "$SWEEP_FILTER" "$EMA" "$TARGET_PCT" "$NUM_PROBE_JOBS" "$PROBE_INDEX_OFFSET" <<'PY'
import csv
import sys
from pathlib import Path

analysis_csv = Path(sys.argv[1])
task_id = int(sys.argv[2])
sweep_filter = sys.argv[3]
ema = sys.argv[4]
target_pct_arg = sys.argv[5].strip().lower()
num_probe_jobs = int(sys.argv[6])
probe_index_offset = int(sys.argv[7])

if num_probe_jobs <= 0:
    raise SystemExit(f"num_probe_jobs must be positive, got {num_probe_jobs}")
if probe_index_offset < 0:
    raise SystemExit(f"probe_index_offset must be nonnegative, got {probe_index_offset}")

if not analysis_csv.exists():
    raise SystemExit(f"analysis CSV not found: {analysis_csv}")
if target_pct_arg not in {"all", "*", "each", "every"}:
    try:
        target_pct = float(target_pct_arg)
    except ValueError as exc:
        raise SystemExit(
            "target_pct must be a float or one of all/*/each/every, "
            f"got {target_pct_arg!r}"
        ) from exc
else:
    target_pct = None

items = []
seen = set()
with analysis_csv.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if sweep_filter and row.get("sweep") != sweep_filter:
            continue
        run_id = row["wandb_run_id"]
        checkpoint_dir = Path(row["checkpoint_dir"])
        batch_size = int(row["batch_size"])
        learning_rate = float(row["learning_rate"])
        final_step = int(row["final_step"])
        steps = [int(x) for x in row["checkpoint_steps"].split()]
        if not steps:
            continue
        if target_pct is None:
            selected_steps = steps
        else:
            target_step = target_pct * final_step
            selected_steps = [min(steps, key=lambda s: (abs(s - target_step), s))]

        for selected_step in selected_steps:
            checkpoint_path = checkpoint_dir / str(selected_step)
            if not checkpoint_path.is_dir():
                raise SystemExit(f"selected checkpoint dir missing: {checkpoint_path}")
            row_key = (str(checkpoint_path), batch_size)
            if row_key in seen:
                continue
            seen.add(row_key)
            actual_pct = selected_step / final_step if final_step else float("nan")
            for probe_idx in range(num_probe_jobs):
                global_probe_idx = probe_index_offset + probe_idx
                items.append(
                    (
                        run_id,
                        str(checkpoint_dir),
                        str(checkpoint_path),
                        str(selected_step),
                        str(batch_size),
                        f"{learning_rate:g}",
                        ema,
                        f"{actual_pct:.8f}",
                        str(global_probe_idx),
                        str(len(items)),
                    )
                )

if not items:
    raise SystemExit(
        f"No tasks built from {analysis_csv} with sweep_filter={sweep_filter!r}"
    )
if not 0 <= task_id < len(items):
    raise SystemExit(f"Task id {task_id} out of range [0, {len(items) - 1}]")

for value in items[task_id]:
    print(value)
print(len(items))
PY
)

RUN_ID="${TASK_INFO[0]}"
CHECKPOINT_ROOT="${TASK_INFO[1]}"
CHECKPOINT_PATH="${TASK_INFO[2]}"
CHECKPOINT_STEP="${TASK_INFO[3]}"
BASE_BATCH_SIZE="${TASK_INFO[4]}"
BASE_LR="${TASK_INFO[5]}"
EMA="${TASK_INFO[6]}"
ACTUAL_PCT="${TASK_INFO[7]}"
PROBE_INDEX="${TASK_INFO[8]}"
TASK_INDEX="${TASK_INFO[9]}"
TOTAL_TASKS="${TASK_INFO[10]}"
PROBE_SEED="$((PROBE_SEED_BASE + PROBE_INDEX))"

if [[ "$INCLUDE_ADAM_PRECOND" == "1" ]]; then
  METRIC_PREFIX="hessian_frob2"
else
  METRIC_PREFIX="hessian_frob2_raw"
fi
EMA_SUFFIX=""
if [[ "$EMA" != "0" && "$EMA" != "0.0" ]]; then
  EMA_TAG="${EMA//./p}"
  EMA_SUFFIX="_ema_${EMA_TAG}"
fi
OUTPUT_FILE="$CHECKPOINT_PATH/${METRIC_PREFIX}${EMA_SUFFIX}_probe${PROBE_INDEX}.csv"

JOB_GROUP_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
LOG_SUBDIR="$(date +%Y_%m_%d)_${JOB_GROUP_ID}"
mkdir -p "logs/$LOG_SUBDIR"
cp -n "${BASH_SOURCE[0]}" "logs/$LOG_SUBDIR/preprocessing_run_sample_hessian_frob2.sh" || true
exec >"logs/$LOG_SUBDIR/$TASK_ID.log" 2>&1

CMD=(
  "$PYTHON_BIN" -u preprocessing/sample_hessian_frob2.py
  --checkpoint-dir "$CHECKPOINT_PATH"
  --num-samples "$NUM_SAMPLES"
  --num-tokens "$NUM_TOKENS"
  --num-probes 1
  --probe-seed "$PROBE_SEED"
  --flush-every "$FLUSH_EVERY"
  --output-file "$OUTPUT_FILE"
)

if [[ "$EMA" != "0" && "$EMA" != "0.0" ]]; then
  CMD+=(--ema "$EMA")
fi
if [[ "$INCLUDE_ADAM_PRECOND" == "1" ]]; then
  CMD+=(--include-adam-precond)
else
  CMD+=(--no-include-adam-precond)
fi
if [[ "$ALLOW_PAST_TRAIN_END" == "1" ]]; then
  CMD+=(--allow-past-training-end)
else
  CMD+=(--no-allow-past-training-end)
fi
if [[ "$REPEAT_DATASET" == "1" ]]; then
  CMD+=(--repeat-dataset)
else
  CMD+=(--no-repeat-dataset)
fi

echo "analysis_csv=$ANALYSIS_CSV sweep_filter=$SWEEP_FILTER task_id=$TASK_ID task_index=$TASK_INDEX total_tasks=$TOTAL_TASKS run_id=$RUN_ID checkpoint_root=$CHECKPOINT_ROOT checkpoint_step=$CHECKPOINT_STEP checkpoint=$CHECKPOINT_PATH target_pct=$TARGET_PCT actual_pct=$ACTUAL_PCT base_batch_size=$BASE_BATCH_SIZE base_lr=$BASE_LR ema=$EMA num_samples=$NUM_SAMPLES num_tokens=$NUM_TOKENS num_probe_jobs=$NUM_PROBE_JOBS probe_index_offset=$PROBE_INDEX_OFFSET probe_index=$PROBE_INDEX probe_seed=$PROBE_SEED output_file=$OUTPUT_FILE include_adam_precond=$INCLUDE_ADAM_PRECOND skip_existing=$SKIP_EXISTING allow_past_train_end=$ALLOW_PAST_TRAIN_END repeat_dataset=$REPEAT_DATASET python=$PYTHON_BIN"
if [[ "$SKIP_EXISTING" == "1" && -s "$OUTPUT_FILE" ]]; then
  echo "Skipping existing nonempty output: $OUTPUT_FILE"
  exit 0
fi
"${CMD[@]}"
