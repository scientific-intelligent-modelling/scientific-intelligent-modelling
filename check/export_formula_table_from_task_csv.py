#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


REMOTE_HELPER = r"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pick_formula(canonical: dict[str, object] | None, equation: object) -> str | None:
    if isinstance(canonical, dict):
        for key in [
            "normalized_expression",
            "instantiated_expression",
            "return_expression_source",
            "python_function_source",
            "raw_equation",
        ]:
            value = canonical.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(equation, str) and equation.strip():
        return equation.strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        result_path = Path(item["result_path"])
        result: dict[str, object] = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result = {}
        canonical = result.get("canonical_artifact")
        if not isinstance(canonical, dict):
            canonical = {}
        row = {
            "host_label": item.get("host_label"),
            "method": item.get("method"),
            "seed": item.get("seed"),
            "dataset_dir": item.get("dataset_dir"),
            "dataset_name": item.get("dataset_name"),
            "result_status": item.get("result_status"),
            "result_path": item.get("result_path"),
            "formatted_formula": _pick_formula(canonical, result.get("equation")),
            "raw_equation_kind": canonical.get("raw_equation_kind"),
            "normalization_mode": canonical.get("normalization_mode"),
            "artifact_valid": canonical.get("artifact_valid"),
            "ast_node_count": canonical.get("ast_node_count"),
            "tree_depth": canonical.get("tree_depth"),
            "operator_set": canonical.get("operator_set"),
            "canonical_validation_errors": canonical.get("validation_errors"),
        }
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
"""


HOST_MAP = {
    "iaaccn22": {"mode": "direct", "host": "iaaccn22"},
    "iaaccn23": {"mode": "via22", "via": "iaaccn22", "ip": "10.10.100.23"},
    "iaaccn24": {"mode": "via22", "via": "iaaccn22", "ip": "10.10.100.24"},
    "iaaccn25": {"mode": "via22", "via": "iaaccn22", "ip": "10.10.100.25"},
    "iaaccn26": {"mode": "via22", "via": "iaaccn22", "ip": "10.10.100.26"},
    "iaaccn27": {"mode": "via22", "via": "iaaccn22", "ip": "10.10.100.27"},
    "iaaccn28": {"mode": "via22", "via": "iaaccn22", "ip": "10.10.100.28"},
    "iaaccn29": {"mode": "via22", "via": "iaaccn22", "ip": "10.10.100.29"},
}


def run_cmd(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def run_argv(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True)


def _scp(local_path: Path, host: str, remote_path: str) -> None:
    cmd = (
        f"scp -o BatchMode=yes -o ConnectTimeout=8 "
        f"{shlex.quote(str(local_path))} {shlex.quote(host)}:{shlex.quote(remote_path)}"
    )
    completed = run_cmd(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"scp 到 {host} 失败: {completed.stderr.strip()}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_manifest_rows(task_csv: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with task_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["host_label"]].append(
                {
                    "host_label": row.get("host_label"),
                    "method": row.get("method"),
                    "seed": row.get("seed"),
                    "dataset_dir": row.get("dataset_dir"),
                    "dataset_name": row.get("dataset_name"),
                    "result_status": row.get("result_status"),
                    "result_path": row.get("result_path"),
                }
            )
    return grouped


def _deploy_and_run(host_label: str, manifest_local: Path, helper_local: Path) -> list[dict[str, Any]]:
    spec = HOST_MAP[host_label]
    remote_helper = "/tmp/export_formula_remote_helper.py"
    remote_manifest = f"/tmp/{manifest_local.name}"
    if spec["mode"] == "direct":
        _scp(helper_local, spec["host"], remote_helper)
        _scp(manifest_local, spec["host"], remote_manifest)
        cmd = (
            f"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=8 {shlex.quote(spec['host'])} "
            f"{shlex.quote(f'python {remote_helper} --manifest {remote_manifest}')}"
        )
        completed = run_cmd(cmd)
    else:
        _scp(helper_local, spec["via"], remote_helper)
        _scp(manifest_local, spec["via"], remote_manifest)
        relay = (
            f"scp -o BatchMode=yes -o ConnectTimeout=8 {remote_helper} {spec['ip']}:{remote_helper} >/dev/null && "
            f"scp -o BatchMode=yes -o ConnectTimeout=8 {remote_manifest} {spec['ip']}:{remote_manifest} >/dev/null && "
            f"timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=8 {spec['ip']} "
            f"\\\"python {remote_helper} --manifest {remote_manifest}\\\""
        )
        completed = run_argv(
            [
                "timeout",
                "80",
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                spec["via"],
                relay,
            ]
        )
    if completed.returncode != 0:
        raise RuntimeError(f"拉取 {host_label} 公式失败: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="从任务级结果表导出格式化公式表")
    parser.add_argument("--task-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    grouped = _group_manifest_rows(Path(args.task_csv))
    with tempfile.TemporaryDirectory(prefix="formula_export_") as td:
        td_path = Path(td)
        helper_local = td_path / "export_formula_remote_helper.py"
        helper_local.write_text(REMOTE_HELPER, encoding="utf-8")

        all_rows: list[dict[str, Any]] = []
        for host_label, rows in sorted(grouped.items()):
            manifest_local = td_path / f"{host_label}_manifest.jsonl"
            manifest_local.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            all_rows.extend(_deploy_and_run(host_label, manifest_local, helper_local))

    all_rows.sort(key=lambda x: (x["method"], str(x["seed"]), x["dataset_dir"]))
    _write_csv(Path(args.output_csv), all_rows)
    print(json.dumps({"rows": len(all_rows), "output_csv": str(Path(args.output_csv))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
