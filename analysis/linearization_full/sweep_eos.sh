#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
ANALYSIS_CSV="$REPO_ROOT/analysis/analysis_runs.csv"
PYTHON_BIN="${PYTHON_BIN:-$HOME/pax/bin/python}"
REST_SAMPLE_CSV_REL="sample_g2_over_nu_splits/sample_g2_over_nu_rest_below_top1pct_any_weight.csv"
CUSTOM_REST_SAMPLE_CSV_REL=0

usage() {
  echo "Usage: sweep_eos.sh [--analysis-csv PATH] [--sweep NAME] [--ema VALUE] [--transform NAME] [--rest-csv-rel PATH] [--lr-mult VALUE[,VALUE...]]... [--bsz-mult VALUE[,VALUE...]]... [--max-parallel N] [--wandb-group NAME]"
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

append_unique_pair() {
  local pair="$1"
  local -n out="$2"
  local -n seen="$3"
  if [[ -n "${seen[$pair]:-}" ]]; then
    return
  fi
  out+=("$pair")
  seen["$pair"]=1
}

collect_checkpoint_paths() {
  "$PYTHON_BIN" - "$ANALYSIS_CSV" "$REST_SAMPLE_CSV_REL" "$SWEEP_FILTER" <<'PY'
import csv
import sys
from pathlib import Path

analysis_csv = Path(sys.argv[1]).resolve()
rest_rel = Path(sys.argv[2])
sweep_filter = sys.argv[3]

if not analysis_csv.exists():
    raise SystemExit(f"analysis CSV not found: {analysis_csv}")

paths = []
seen = set()
missing = []
with analysis_csv.open(newline="") as f:
    reader = csv.DictReader(f)
    required = {"checkpoint_dir", "checkpoint_steps"}
    missing_columns = sorted(required - set(reader.fieldnames or []))
    if missing_columns:
        raise SystemExit(
            f"{analysis_csv} is missing required columns: {missing_columns}"
        )
    for row_idx, row in enumerate(reader):
        if sweep_filter and row.get("sweep") != sweep_filter:
            continue
        checkpoint_dir = Path(row["checkpoint_dir"])
        for step in row["checkpoint_steps"].split():
            checkpoint_path = checkpoint_dir / step
            key = str(checkpoint_path)
            if key in seen:
                continue
            seen.add(key)
            rest_csv = checkpoint_path / rest_rel
            if not checkpoint_path.is_dir():
                missing.append(f"checkpoint dir: {checkpoint_path}")
                continue
            if not rest_csv.is_file() or rest_csv.stat().st_size == 0:
                missing.append(f"rest CSV: {rest_csv}")
                continue
            paths.append(str(checkpoint_path.resolve()))

if missing:
    print(
        f"Skipping {len(missing)} checkpoint entries from {analysis_csv}"
        " because required files are missing.",
        file=sys.stderr,
    )
    for item in missing[:80]:
        print(f"  {item}", file=sys.stderr)
    if len(missing) > 80:
        print(f"  ... and {len(missing) - 80} more", file=sys.stderr)

if not paths:
    raise SystemExit(
        f"No checkpoint paths with generated {rest_rel} found in {analysis_csv}"
    )

for path in paths:
    print(path)
PY
}

MAX_PARALLEL=64
WANDB_GROUP=cosine_online_2_ckpts
SWEEP_FILTER=""
LINEARIZATION_EMA=""
LINEARIZATION_TRANSFORM="${LINEARIZATION_TRANSFORM:-quad}"
DEFAULT_LR_MULTS=(0.250000 0.500000 1.000000 2.000000 4.000000)
DEFAULT_BSZ_MULTS=(0.25 0.5 1.0 2.0 4.0)
LR_MULTS=()
BSZ_MULTS=()

while (( $# > 0 )); do
  case "$1" in
    --analysis-csv)
      ANALYSIS_CSV="$(realpath "${2:?Missing value for --analysis-csv}")"
      shift 2
      ;;
    --sweep)
      SWEEP_FILTER="${2:?Missing value for --sweep}"
      shift 2
      ;;
    --ema)
      LINEARIZATION_EMA="${2:?Missing value for --ema}"
      shift 2
      ;;
    --transform)
      LINEARIZATION_TRANSFORM="${2:?Missing value for --transform}"
      shift 2
      ;;
    --rest-csv-rel)
      REST_SAMPLE_CSV_REL="${2:?Missing value for --rest-csv-rel}"
      CUSTOM_REST_SAMPLE_CSV_REL=1
      shift 2
      ;;
    --lr-mult)
      append_csv_values "${2:?Missing value for --lr-mult}" LR_MULTS
      shift 2
      ;;
    --bsz-mult)
      append_csv_values "${2:?Missing value for --bsz-mult}" BSZ_MULTS
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="${2:?Missing value for --max-parallel}"
      shift 2
      ;;
    --wandb-group)
      WANDB_GROUP="${2:?Missing value for --wandb-group}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unexpected positional argument: $1" >&2
      echo "Checkpoint paths are read from --analysis-csv, defaulting to $ANALYSIS_CSV" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if (( ${#LR_MULTS[@]} == 0 )); then
  LR_MULTS=("${DEFAULT_LR_MULTS[@]}")
fi
if (( ${#BSZ_MULTS[@]} == 0 )); then
  BSZ_MULTS=("${DEFAULT_BSZ_MULTS[@]}")
fi

if [[ "$CUSTOM_REST_SAMPLE_CSV_REL" == "0" && -n "$LINEARIZATION_EMA" && "$LINEARIZATION_EMA" != "0" ]]; then
  EMA_LABEL="${LINEARIZATION_EMA//./p}"
  REST_SAMPLE_CSV_REL="sample_g2_over_nu_ema_${EMA_LABEL}_splits/sample_g2_over_nu_rest_below_top1pct_any_weight.csv"
fi

mapfile -t CHECKPOINT_PATHS < <(collect_checkpoint_paths)
STEP_COUNT="${#CHECKPOINT_PATHS[@]}"

MULTIPLIER_PAIRS=()
declare -A SEEN_PAIRS=()
for lr_mult in "${LR_MULTS[@]}"; do
  for bsz_mult in "${BSZ_MULTS[@]}"; do
    append_unique_pair "${lr_mult}:${bsz_mult}" MULTIPLIER_PAIRS SEEN_PAIRS
  done
done

PAIR_COUNT="${#MULTIPLIER_PAIRS[@]}"
TOTAL_JOBS=$(( STEP_COUNT * PAIR_COUNT ))

(( TOTAL_JOBS > 0 )) || {
  echo "No jobs to submit" >&2
  exit 1
}

MULTIPLIER_PAIR_CSV="$(IFS=,; echo "${MULTIPLIER_PAIRS[*]}")"

echo "analysis_csv=$ANALYSIS_CSV"
if [[ -n "$SWEEP_FILTER" ]]; then
  echo "sweep_filter=$SWEEP_FILTER"
fi
echo "checkpoint_steps=$STEP_COUNT"
echo "required_rest_csv=$REST_SAMPLE_CSV_REL"
echo "lr_mults=${LR_MULTS[*]}"
echo "bsz_mults=${BSZ_MULTS[*]}"
echo "multiplier_pairs=${MULTIPLIER_PAIRS[*]}"
if [[ -n "$WANDB_GROUP" ]]; then
  echo "wandb_group=$WANDB_GROUP"
fi
if [[ -n "$LINEARIZATION_EMA" ]]; then
  echo "ema=$LINEARIZATION_EMA"
fi
echo "Submitting $TOTAL_JOBS jobs with transform=$LINEARIZATION_TRANSFORM train=full optimizer=sgdm"

cd "$REPO_ROOT"
if [[ -n "$WANDB_GROUP" ]]; then
  WANDB_RUN_GROUP="$WANDB_GROUP" REST_SAMPLE_CSV_REL="$REST_SAMPLE_CSV_REL" LINEARIZATION_EMA="$LINEARIZATION_EMA" LINEARIZATION_TRANSFORM="$LINEARIZATION_TRANSFORM" sbatch --array="0-$((TOTAL_JOBS - 1))%${MAX_PARALLEL}" \
    "$SCRIPT_DIR/run_eos.sh" \
    "$MULTIPLIER_PAIR_CSV" \
    "${CHECKPOINT_PATHS[@]}"
else
  REST_SAMPLE_CSV_REL="$REST_SAMPLE_CSV_REL" LINEARIZATION_EMA="$LINEARIZATION_EMA" LINEARIZATION_TRANSFORM="$LINEARIZATION_TRANSFORM" sbatch --array="0-$((TOTAL_JOBS - 1))%${MAX_PARALLEL}" \
    "$SCRIPT_DIR/run_eos.sh" \
    "$MULTIPLIER_PAIR_CSV" \
    "${CHECKPOINT_PATHS[@]}"
fi
