#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REMOTE_HELPER = r"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric(metric: dict[str, object] | None, key: str) -> object | None:
    if not isinstance(metric, dict):
        return None
    return metric.get(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    base = batch_dir / args.method
    if not base.exists():
        return

    status_files = sorted(base.glob(f"seed*/{args.host_label}/__launcher__/task_status.jsonl"))
    for status_file in status_files:
        parts = status_file.parts
        try:
            seed_name = parts[-4]
            seed = int(seed_name.removeprefix("seed"))
        except Exception:
            seed = None

        for line in status_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            result_path = Path(record.get("experiment_dir", "")) / "result.json"
            if not result_path.exists():
                result_path = Path(record["result_path"])
            result: dict[str, object] = {}
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    result = {}

            valid = result.get("valid")
            id_test = result.get("id_test")
            ood_test = result.get("ood_test")
            canonical_artifact = result.get("canonical_artifact")
            row = {
                "host_label": args.host_label,
                "method": args.method,
                "seed": seed,
                "task_key": record.get("task_key"),
                "dataset_name": record.get("dataset_name"),
                "dataset_dir": record.get("dataset_dir"),
                "global_index": record.get("global_index"),
                "task_status": record.get("status"),
                "task_error": record.get("error"),
                "task_seconds": record.get("seconds"),
                "finished_at": record.get("finished_at"),
                "log_path": record.get("log_path"),
                "experiment_dir": record.get("experiment_dir"),
                "result_path": str(result_path),
                "result_status": result.get("status"),
                "result_error": result.get("error"),
                "equation": result.get("equation"),
                "canonical_artifact_present": canonical_artifact is not None,
                "recovered_after_timeout": result.get("recovered_after_timeout"),
                "valid_r2": _metric(valid, "r2"),
                "valid_rmse": _metric(valid, "rmse"),
                "valid_nmse": _metric(valid, "nmse"),
                "id_r2": _metric(id_test, "r2"),
                "id_rmse": _metric(id_test, "rmse"),
                "id_nmse": _metric(id_test, "nmse"),
                "ood_r2": _metric(ood_test, "r2"),
                "ood_rmse": _metric(ood_test, "rmse"),
                "ood_nmse": _metric(ood_test, "nmse"),
            }
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
"""


HOST_SPECS = [
    {"host": "iaaccn22", "method": "pysr", "host_label": "iaaccn22"},
    {"host": "iaaccn23", "method": "pysr", "host_label": "iaaccn23"},
    {"host": "iaaccn24", "method": "pysr", "host_label": "iaaccn24"},
    {"host": "iaaccn25", "method": "pysr", "host_label": "iaaccn25", "via": "iaaccn22", "via_host": "10.10.100.25"},
    {"host": "iaaccn26", "method": "llmsr", "host_label": "iaaccn26"},
    {"host": "iaaccn27", "method": "llmsr", "host_label": "iaaccn27"},
    {"host": "iaaccn28", "method": "llmsr", "host_label": "iaaccn28"},
    {"host": "iaaccn29", "method": "llmsr", "host_label": "iaaccn29"},
]


def run_cmd(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def run_argv(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True)


def _ssh_prefix(host: str, timeout_sec: int = 40) -> str:
    return f"timeout {timeout_sec} ssh -o BatchMode=yes -o ConnectTimeout=8 {shlex.quote(host)}"


def _scp(local_path: Path, host: str, remote_path: str) -> None:
    cmd = (
        f"scp -o BatchMode=yes -o ConnectTimeout=8 "
        f"{shlex.quote(str(local_path))} {shlex.quote(host)}:{shlex.quote(remote_path)}"
    )
    completed = run_cmd(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"scp 到 {host} 失败: {completed.stderr.strip()}")


def _remote_file_exists(host: str, remote_path: str, timeout_sec: int = 20) -> bool:
    completed = run_cmd(
        f"timeout {timeout_sec} ssh -o BatchMode=yes -o ConnectTimeout=8 {shlex.quote(host)} "
        f"{shlex.quote(f'test -f {remote_path} && echo OK')}"
    )
    return completed.returncode == 0 and completed.stdout.strip() == "OK"


def _deploy_helper(local_helper: Path, spec: dict[str, str]) -> None:
    if "via" not in spec:
        if _remote_file_exists(spec["host"], "/tmp/aggregate_formal200_remote.py"):
            return
        try:
            _scp(local_helper, spec["host"], "/tmp/aggregate_formal200_remote.py")
        except RuntimeError:
            if not _remote_file_exists(spec["host"], "/tmp/aggregate_formal200_remote.py"):
                raise
        return
    if _remote_file_exists(spec["via"], "/tmp/aggregate_formal200_remote.py"):
        via_has_helper = True
    else:
        via_has_helper = False
    try:
        if not via_has_helper:
            _scp(local_helper, spec["via"], "/tmp/aggregate_formal200_remote.py")
    except RuntimeError:
        if not _remote_file_exists(spec["via"], "/tmp/aggregate_formal200_remote.py"):
            raise
    via_cmd = (
        f"{_ssh_prefix(spec['via'], 50)} "
        f"\"scp -o BatchMode=yes -o ConnectTimeout=8 /tmp/aggregate_formal200_remote.py "
        f"{shlex.quote(spec['via_host'])}:/tmp/aggregate_formal200_remote.py || "
        f"ssh -o BatchMode=yes -o ConnectTimeout=8 {shlex.quote(spec['via_host'])} "
        f"'test -f /tmp/aggregate_formal200_remote.py && echo OK'\""
    )
    completed = run_cmd(via_cmd)
    if completed.returncode != 0 and "OK" not in completed.stdout:
        raise RuntimeError(f"经 {spec['via']} 分发到 {spec['host']} 失败: {completed.stderr.strip()}")


def _fetch_rows(spec: dict[str, str], batch_dir: str) -> list[dict[str, Any]]:
    remote_cmd = (
        f"python /tmp/aggregate_formal200_remote.py "
        f"--batch-dir {shlex.quote(batch_dir)} "
        f"--method {shlex.quote(spec['method'])} "
        f"--host-label {shlex.quote(spec['host_label'])}"
    )
    if "via" not in spec:
        completed = run_cmd(f"{_ssh_prefix(spec['host'], 50)} {shlex.quote(remote_cmd)}")
    else:
        completed = run_argv(
            [
                "timeout",
                "45",
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                spec["via"],
                f'timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=8 {spec["via_host"]} "{remote_cmd}"',
            ]
        )
    if completed.returncode != 0:
        raise RuntimeError(f"抓取 {spec['host']} 结果失败: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_candidate_meta(candidate_json: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(candidate_json.read_text(encoding="utf-8"))
    merged: dict[str, dict[str, Any]] = {}
    for pool_name in ("pool_A", "pool_B", "pool_C"):
        for item in payload.get(pool_name, []):
            merged[item["dataset_dir"]] = {
                "candidate_pool": pool_name.removeprefix("pool_"),
                "family": item.get("family"),
                "subgroup": item.get("subgroup"),
                "basename": item.get("basename"),
                "selection_mode": item.get("selection_mode"),
                "candidate_advantage_side": item.get("advantage_side"),
                "candidate_gap_score": item.get("overall_gap_score"),
                "candidate_signed_advantage": item.get("signed_advantage"),
            }
    return merged


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, Any]], output_md: Path, batch_dir: str) -> None:
    method_status = defaultdict(Counter)
    seed_status = defaultdict(Counter)
    family_method = defaultdict(Counter)
    metric_coverage = defaultdict(Counter)
    for row in rows:
        method_status[row["method"]][row["result_status"] or row["task_status"] or "unknown"] += 1
        seed_status[str(row["seed"])][row["method"]] += 1
        family_method[row.get("family") or "unknown"][row["method"]] += 1
        if row.get("id_r2") is not None and row.get("ood_r2") is not None:
            metric_coverage[row["method"]]["full_id_ood"] += 1
        elif row.get("equation"):
            metric_coverage[row["method"]]["equation_only_or_partial"] += 1
        else:
            metric_coverage[row["method"]]["no_equation"] += 1

    lines = [
        "# 正式实验结果汇总",
        "",
        f"- 批次目录：`{batch_dir}`",
        f"- 任务总数：`{len(rows)}`",
        "",
        "## 方法 × 状态",
        "",
    ]
    for method, counter in sorted(method_status.items()):
        lines.append(f"### `{method}`")
        for status, count in sorted(counter.items()):
            lines.append(f"- `{status}`: `{count}`")
        lines.append("")
    lines.append("## seed × 方法任务数")
    lines.append("")
    for seed, counter in sorted(seed_status.items()):
        lines.append(f"### `seed={seed}`")
        for method, count in sorted(counter.items()):
            lines.append(f"- `{method}`: `{count}`")
        lines.append("")
    lines.append("## 指标覆盖")
    lines.append("")
    for method, counter in sorted(metric_coverage.items()):
        lines.append(f"### `{method}`")
        for key, count in sorted(counter.items()):
            lines.append(f"- `{key}`: `{count}`")
        lines.append("")
    lines.append("## family × 方法任务数")
    lines.append("")
    for family, counter in sorted(family_method.items()):
        counts = ", ".join(f"{m}={c}" for m, c in sorted(counter.items()))
        lines.append(f"- `{family}`: {counts}")
    output_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合 formal200 正式实验任务级结果")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--candidate-json", default="/tmp/candidate_seeds_200_v3.json")
    parser.add_argument("--output-csv", default="/tmp/formal200_v1_results_table.csv")
    parser.add_argument("--output-md", default="/tmp/formal200_v1_results_summary.md")
    args = parser.parse_args()

    candidate_meta = _load_candidate_meta(Path(args.candidate_json))
    all_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="formal200_agg_") as td:
        helper = Path(td) / "aggregate_formal200_remote.py"
        helper.write_text(REMOTE_HELPER, encoding="utf-8")
        for spec in HOST_SPECS:
            _deploy_helper(helper, spec)
        for spec in HOST_SPECS:
            rows = _fetch_rows(spec, args.batch_dir)
            for row in rows:
                meta = candidate_meta.get(row["dataset_dir"], {})
                merged = {**row, **meta}
                all_rows.append(merged)

    all_rows.sort(key=lambda x: (x["method"], x["seed"], x["host_label"], x["global_index"]))
    _write_csv(all_rows, Path(args.output_csv))
    _write_summary(all_rows, Path(args.output_md), args.batch_dir)
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "output_csv": str(Path(args.output_csv)),
                "output_md": str(Path(args.output_md)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
