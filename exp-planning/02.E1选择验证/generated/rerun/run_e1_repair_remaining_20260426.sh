#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/family/workplace/scientific-intelligent-modelling"
REMOTE_ROOT="/home/zhangziwen/projects/scientific-intelligent-modelling"
BATCH_NAME="${BATCH_NAME:-e1_repair_remaining_20260426}"

start_job() {
  local host="$1"
  local session="$2"
  local job_rel="$3"
  local slice_rel="$4"
  local workers="$5"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p \"$REMOTE_ROOT/check\" \"$REMOTE_ROOT/$(dirname "$job_rel")\" \"$REMOTE_ROOT/$(dirname "$slice_rel")\" \"$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params\""
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/check/launch_e1_benchmark.py" "$host:$REMOTE_ROOT/check/launch_e1_benchmark.py"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$slice_rel" "$host:$REMOTE_ROOT/$slice_rel"
  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$job_rel" "$host:$REMOTE_ROOT/$job_rel"
  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "chmod +x \"$REMOTE_ROOT/$job_rel\" && tmux kill-session -t \"$session\" >/dev/null 2>&1 || true; tmux new-session -d -s \"$session\" /bin/bash \"$REMOTE_ROOT/$job_rel\" \"$BATCH_NAME\" \"$workers\""
  echo "STARTED ${host} ${session}"
}

start_job "iaaccn28" "e1_repair_dso" "exp-planning/02.E1选择验证/generated/rerun/run_e1_repair_remaining_20260426_dso_iaaccn28.sh" "exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_dso.csv" "15"
start_job "iaaccn22" "e1_repair_gplearn" "exp-planning/02.E1选择验证/generated/rerun/run_e1_repair_remaining_20260426_gplearn_iaaccn22.sh" "exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_gplearn.csv" "9"
start_job "iaaccn26" "e1_repair_llmsr" "exp-planning/02.E1选择验证/generated/rerun/run_e1_repair_remaining_20260426_llmsr_iaaccn26.sh" "exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_llmsr.csv" "9"
start_job "iaaccn23" "e1_repair_pyoperon" "exp-planning/02.E1选择验证/generated/rerun/run_e1_repair_remaining_20260426_pyoperon_iaaccn23.sh" "exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_pyoperon.csv" "14"
start_job "iaaccn24" "e1_repair_pysr" "exp-planning/02.E1选择验证/generated/rerun/run_e1_repair_remaining_20260426_pysr_iaaccn24.sh" "exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_pysr.csv" "8"
start_job "iaaccn25" "e1_repair_tpsr" "exp-planning/02.E1选择验证/generated/rerun/run_e1_repair_remaining_20260426_tpsr_iaaccn25.sh" "exp-planning/02.E1选择验证/generated/rerun/e1_repair_remaining_20260426_tpsr.csv" "13"

echo "REPAIR_RERUN_DISPATCHED ${BATCH_NAME}"
