#!/bin/bash
CHECKPOINT_ROOT="${1:?Usage: sweep.sh <checkpoint_root> [step] [clip_threshold] [lr_mult] [bsz_mult] [ema]}"
STEP="${2:-2525593}"
CLIP_THRESHOLD="${3:-inf}"
LR_MULT="${4:-1.0}"
BSZ_MULT="${5:-1.0}"
EMA="${6:-0}"
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
CHECKPOINT_PATH="$CHECKPOINT_ROOT/$STEP"

[[ -d "$CHECKPOINT_PATH" ]] || {
  echo "Checkpoint step dir not found: $CHECKPOINT_PATH"
  exit 1
}

TOTAL_JOBS=20
echo "Submitting $TOTAL_JOBS jobs for checkpoint $CHECKPOINT_PATH clip_threshold=$CLIP_THRESHOLD lr_mult=$LR_MULT bsz_mult=$BSZ_MULT ema=$EMA"

sbatch --array="0-$((TOTAL_JOBS - 1))%16" \
  "$SCRIPT_DIR/run.sh" "$CHECKPOINT_PATH" "$CLIP_THRESHOLD" "$LR_MULT" "$BSZ_MULT" "$EMA"
