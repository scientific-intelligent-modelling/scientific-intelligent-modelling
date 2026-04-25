#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REPO_ROOT / "exp-planning" / "02.E1选择验证"
DIGEST_NONVALID = E1_ROOT / "e1_result_digest_20260424-041046" / "e1_nonvalid_cases.csv"
CANDIDATE200_CSV = E1_ROOT / "generated" / "candidate200_unified.csv"
OUTPUT_DIR = E1_ROOT / "generated" / "rerun"
REMOTE_ROOT = "/home/zhangziwen/projects/scientific-intelligent-modelling"
SEED = 1314
BATCH_NAME = "e1_repair_remaining_20260426"


TOOL_RUN_CONFIG = {
    "gplearn": {"host": "iaaccn22", "conda_env": "sim_base", "workers": 9},
    "llmsr": {"host": "iaaccn26", "conda_env": "sim_llm", "workers": 9},
    "pyoperon": {"host": "iaaccn23", "conda_env": "sim_base", "workers": 14},
    "pysr": {"host": "iaaccn24", "conda_env": "sim_base", "workers": 8},
    "tpsr": {"host": "iaaccn25", "conda_env": "sim_tpsr", "workers": 13},
    "dso": {"host": "iaaccn28", "conda_env": "sim_dso", "workers": 15},
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _remote_job_script(*, tool: str, conda_env: str, slice_rel: str, workers: int, host: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

BATCH_NAME="${{1:-{BATCH_NAME}}}"
WORKERS="${{2:-{workers}}}"
REMOTE_ROOT="{REMOTE_ROOT}"

cd "$REMOTE_ROOT"
export PYTHONPATH=.

conda run -n {conda_env} python check/launch_e1_benchmark.py run \\
  --tool {tool} \\
  --slice-csv "$REMOTE_ROOT/{slice_rel}" \\
  --params-json "$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params/{tool}.json" \\
  --output-root "$REMOTE_ROOT/experiments/${{BATCH_NAME}}/{host}/{tool}" \\
  --seed {SEED} \\
  --workers "$WORKERS"
"""


def _local_launcher(jobs: list[dict[str, str]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'REPO_ROOT="{REPO_ROOT}"',
        f'REMOTE_ROOT="{REMOTE_ROOT}"',
        f'BATCH_NAME="${{BATCH_NAME:-{BATCH_NAME}}}"',
        "",
        "start_job() {",
        '  local host="$1"',
        '  local session="$2"',
        '  local job_rel="$3"',
        '  local slice_rel="$4"',
        '  local workers="$5"',
        '  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p \\"$REMOTE_ROOT/check\\" \\"$REMOTE_ROOT/$(dirname "$job_rel")\\" \\"$REMOTE_ROOT/$(dirname "$slice_rel")\\" \\"$REMOTE_ROOT/exp-planning/02.E1选择验证/generated/params\\""',
        '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/check/launch_e1_benchmark.py" "$host:$REMOTE_ROOT/check/launch_e1_benchmark.py"',
        '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$slice_rel" "$host:$REMOTE_ROOT/$slice_rel"',
        '  timeout 40 scp -o BatchMode=yes -o ConnectTimeout=10 "$REPO_ROOT/$job_rel" "$host:$REMOTE_ROOT/$job_rel"',
        '  timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "chmod +x \\"$REMOTE_ROOT/$job_rel\\" && tmux kill-session -t \\"$session\\" >/dev/null 2>&1 || true; tmux new-session -d -s \\"$session\\" /bin/bash \\"$REMOTE_ROOT/$job_rel\\" \\"$BATCH_NAME\\" \\"$workers\\""',
        '  echo "STARTED ${host} ${session}"',
        "}",
        "",
    ]
    for job in jobs:
        lines.append(
            f'start_job "{job["host"]}" "{job["session"]}" "{job["job_rel"]}" "{job["slice_rel"]}" "{job["workers"]}"'
        )
    lines.append("")
    lines.append('echo "REPAIR_RERUN_DISPATCHED ${BATCH_NAME}"')
    return "\n".join(lines) + "\n"


def main() -> None:
    candidates = {int(row["global_index"]): row for row in _load_csv(CANDIDATE200_CSV)}
    nonvalid = _load_csv(DIGEST_NONVALID)
    rows_by_tool: dict[str, list[dict[str, str]]] = {tool: [] for tool in TOOL_RUN_CONFIG}
    all_rows: list[dict[str, str]] = []
    for row in nonvalid:
        tool = row["method"]
        if tool not in TOOL_RUN_CONFIG:
            continue
        global_index = int(row["global_index"])
        candidate = dict(candidates[global_index])
        out = {
            "tool": tool,
            "rerun_reason": "e1_repair_remaining_20260426",
            "previous_result_relpath": row.get("result_relpath", ""),
            "previous_timeout_type": row.get("timeout_type", ""),
            "previous_identity_status": row.get("dataset_identity_status", ""),
            "previous_error": row.get("canonical_artifact_error") or row.get("error") or "",
            **candidate,
        }
        rows_by_tool[tool].append(out)
        all_rows.append(out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"{BATCH_NAME}.csv"
    _write_csv(all_path, all_rows)

    jobs: list[dict[str, str]] = []
    for tool, rows in sorted(rows_by_tool.items()):
        if not rows:
            continue
        cfg = TOOL_RUN_CONFIG[tool]
        slice_path = OUTPUT_DIR / f"{BATCH_NAME}_{tool}.csv"
        _write_csv(slice_path, rows)
        job_path = OUTPUT_DIR / f"run_{BATCH_NAME}_{tool}_{cfg['host']}.sh"
        job_path.write_text(
            _remote_job_script(
                tool=tool,
                conda_env=str(cfg["conda_env"]),
                slice_rel=_rel(slice_path),
                workers=int(cfg["workers"]),
                host=str(cfg["host"]),
            ),
            encoding="utf-8",
        )
        job_path.chmod(0o755)
        jobs.append(
            {
                "tool": tool,
                "host": str(cfg["host"]),
                "workers": str(cfg["workers"]),
                "tasks": str(len(rows)),
                "slice_rel": _rel(slice_path),
                "job_rel": _rel(job_path),
                "session": f"e1_repair_{tool}",
            }
        )

    manifest_path = OUTPUT_DIR / f"{BATCH_NAME}_manifest.csv"
    _write_csv(manifest_path, jobs)
    launch_path = OUTPUT_DIR / f"run_{BATCH_NAME}.sh"
    launch_path.write_text(_local_launcher(jobs), encoding="utf-8")
    launch_path.chmod(0o755)
    print(json.dumps({"all": str(all_path), "manifest": str(manifest_path), "launch": str(launch_path), "tasks": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
