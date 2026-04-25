#!/usr/bin/env bash
set -euo pipefail

BATCH_NAME="${1:-e1_repair_remaining_20260426}"
WORKERS="${2:-13}"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n sim_tpsr python check/launch_e1_benchmark.py run \
  --tool tpsr \
  --slice-csv "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_tpsr.csv" \
  --params-json "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params/tpsr.json" \
  --output-root "$REMOTE_ROOT/experiments/${BATCH_NAME}/iaaccn25/tpsr" \
  --seed 1314 \
  --workers "$WORKERS"
