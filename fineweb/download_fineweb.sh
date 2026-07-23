#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the dataset root}"
mkdir -p "$DATA_DIR"
HF_HOME="$(mktemp -d "$DATA_DIR/.hf_home.XXXXXX")"
export HF_HOME
trap 'rm -rf "$HF_HOME"' EXIT
mkdir -p "$HF_HOME" "$DATA_DIR/fineweb_edu_10B_parquet"

uvx hf download \
  HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --local-dir "$DATA_DIR/fineweb_edu_10B_parquet" \
  --include "sample/10BT/*.parquet"
