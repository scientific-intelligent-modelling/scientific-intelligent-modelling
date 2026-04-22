#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <BATCH_NAME> <WORKERS>" >&2
  exit 2
fi

BATCH_NAME="$1"
WORKERS="$2"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n sim_base python check/launch_e1_benchmark.py run \
  --tool pyoperon \
  --slice-csv "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/slices/W2/pyoperon/iaaccn25.csv" \
  --params-json "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params/pyoperon.json" \
  --output-root "$REMOTE_ROOT/experiments/${BATCH_NAME}/iaaccn25" \
  --seed 1314 \
  --workers "$WORKERS"
