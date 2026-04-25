#!/usr/bin/env bash
set -euo pipefail

BATCH_NAME="${1:-e1_repair_remaining_20260426}"
WORKERS="${2:-9}"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n sim_base python check/launch_e1_benchmark.py run \
  --tool gplearn \
  --slice-csv "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_gplearn.csv" \
  --params-json "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params/gplearn.json" \
  --output-root "$REMOTE_ROOT/experiments/${BATCH_NAME}/iaaccn22/gplearn" \
  --seed 1314 \
  --workers "$WORKERS"
