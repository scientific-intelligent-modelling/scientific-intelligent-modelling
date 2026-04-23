#!/usr/bin/env python3
"""按 wave 顺序分发 E1，并对失败任务做有限重试。"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REMOTE_ROOT = Path("/home/zhangziwen/projects/scientific-intelligent-modelling")
SEED = 1314
DONE_STATUSES = {"ok", "timed_out"}
HOST_ADDR = {
    "iaaccn22": None,
    "iaaccn23": "10.10.100.23",
    "iaaccn24": "10.10.100.24",
    "iaaccn25": "10.10.100.25",
    "iaaccn26": "10.10.100.26",
    "iaaccn27": "10.10.100.27",
    "iaaccn28": "10.10.100.28",
    "iaaccn29": "10.10.100.29",
}
WAVE_TIMEOUT_HOURS = {
    "W1": 4.0,
    "W2": 5.0,
    "W3": 4.0,
    "W4": 5.0,
    "W5": 5.0,
}


def _run(host: str, command: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    addr = HOST_ADDR[host]
    if addr is None:
        cmd = ["bash", "-lc", command]
    else:
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            addr,
            command,
        ]
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _task_key(row: dict[str, str]) -> str:
    return f"{row['dataset_dir']}|seed={SEED}"


def _load_expected_keys(slice_rel: str) -> set[str]:
    path = REMOTE_ROOT / slice_rel
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {_task_key(row) for row in csv.DictReader(f)}


def _read_status_lines(host: str, batch_name: str) -> list[str]:
    status_path = REMOTE_ROOT / "experiments" / batch_name / host / "__launcher__" / "task_status.jsonl"
    cmd = f"test -f {shlex.quote(str(status_path))} && cat {shlex.quote(str(status_path))} || true"
    result = _run(host, cmd, timeout=60)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _summarize_job(job: dict[str, str], batch_name: str) -> dict[str, Any]:
    expected = _load_expected_keys(job["slice_rel"])
    latest: dict[str, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for line in _read_status_lines(job["host"], batch_name):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_key = item.get("task_key")
        if isinstance(task_key, str):
            latest[task_key] = item
    for task_key in expected:
        status = str(latest.get(task_key, {}).get("status") or "missing")
        status_counts[status] = status_counts.get(status, 0) + 1
    done = sum(count for status, count in status_counts.items() if status in DONE_STATUSES)
    failed = len(expected) - done
    return {
        "wave": job["wave"],
        "tool": job["tool"],
        "host": job["host"],
        "expected": len(expected),
        "done": done,
        "failed": failed,
        "status_counts": status_counts,
    }


def _tmux_running(job: dict[str, str]) -> bool:
    result = _run(job["host"], f"tmux has-session -t {shlex.quote(job['session'])}", timeout=20)
    return result.returncode == 0


def _start_job(job: dict[str, str], batch_name: str, *, retry: bool) -> None:
    remote_job = REMOTE_ROOT / job["remote_job_rel"]
    args = [
        "/bin/bash",
        str(remote_job),
        batch_name,
        str(job["workers"]),
    ]
    if retry:
        args.append("retry")
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    session = shlex.quote(job["session"])
    cmd = (
        f"cd {shlex.quote(str(REMOTE_ROOT))} && "
        f"chmod +x {shlex.quote(str(remote_job))} && "
        f"(tmux kill-session -t {session} >/dev/null 2>&1 || true) && "
        f"tmux new-session -d -s {session} {quoted_args}"
    )
    result = _run(job["host"], cmd, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"启动失败 {job['host']} {job['session']}: {result.stderr.strip()}")


def _kill_job(job: dict[str, str], batch_name: str) -> None:
    session = shlex.quote(job["session"])
    batch_pattern = shlex.quote(batch_name)
    cmd = (
        f"(tmux kill-session -t {session} >/dev/null 2>&1 || true); "
        f"(pkill -f {batch_pattern} >/dev/null 2>&1 || true)"
    )
    _run(job["host"], cmd, timeout=60)


def _wait_for_wave(jobs: list[dict[str, str]], *, wave: str, batch_name: str, poll_seconds: int) -> bool:
    deadline = time.time() + WAVE_TIMEOUT_HOURS[wave] * 3600
    while time.time() < deadline:
        running = [job for job in jobs if _tmux_running(job)]
        summaries = [_summarize_job(job, batch_name) for job in jobs]
        done = sum(item["done"] for item in summaries)
        expected = sum(item["expected"] for item in summaries)
        print(
            json.dumps(
                {
                    "event": "poll",
                    "wave": wave,
                    "batch_name": batch_name,
                    "running": [job["host"] for job in running],
                    "done": done,
                    "expected": expected,
                    "time": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not running:
            return True
        time.sleep(poll_seconds)
    for job in jobs:
        _kill_job(job, batch_name)
    return False


def _wave_failed(jobs: list[dict[str, str]], batch_name: str) -> tuple[int, list[dict[str, Any]]]:
    summaries = [_summarize_job(job, batch_name) for job in jobs]
    failed = sum(item["failed"] for item in summaries)
    return failed, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="顺序运行 E1 W1~W5 并补跑失败任务")
    parser.add_argument("--manifest", default=str(REMOTE_ROOT / "exp-planning/02.E1选择验证/generated/wave_manifest.csv"))
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--waves", nargs="+", default=["W1", "W2", "W3", "W4", "W5"])
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retry-limit", type=int, default=2)
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    wave_jobs = {wave: [row for row in manifest if row["wave"] == wave] for wave in args.waves}
    all_summaries: dict[str, list[dict[str, Any]]] = {}

    for wave in args.waves:
        jobs = wave_jobs[wave]
        batch_name = f"e1_candidate200_seed{SEED}_v2_{wave.lower()}_{args.stamp}"
        print(json.dumps({"event": "wave_start", "wave": wave, "batch_name": batch_name}, ensure_ascii=False), flush=True)

        for attempt in range(args.retry_limit + 1):
            retry = attempt > 0
            print(json.dumps({"event": "attempt_start", "wave": wave, "attempt": attempt, "retry": retry}, ensure_ascii=False), flush=True)
            for job in jobs:
                _start_job(job, batch_name, retry=retry)
                print(json.dumps({"event": "job_started", "wave": wave, "attempt": attempt, "host": job["host"], "tool": job["tool"]}, ensure_ascii=False), flush=True)
            completed = _wait_for_wave(jobs, wave=wave, batch_name=batch_name, poll_seconds=args.poll_seconds)
            failed, summaries = _wave_failed(jobs, batch_name)
            all_summaries[wave] = summaries
            print(json.dumps({"event": "attempt_done", "wave": wave, "attempt": attempt, "completed": completed, "failed": failed, "summaries": summaries}, ensure_ascii=False), flush=True)
            if completed and failed == 0:
                break
            if attempt == args.retry_limit:
                raise SystemExit(f"{wave} 仍有失败任务: {failed}")

    summary_path = REMOTE_ROOT / "experiments" / f"e1_candidate200_seed{SEED}_v2_{args.stamp}_orchestrator_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "all_done", "summary_path": str(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
