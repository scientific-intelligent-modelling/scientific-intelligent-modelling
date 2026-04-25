#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


DONE = {"ok", "timed_out"}


def slug(text: object, max_len: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "item"
    return value[:max_len]


def normalize_dataset_identity(path: Any) -> str:
    text = "" if path is None else str(path).strip().replace("\\", "/")
    marker = "sim-datasets-data/"
    if marker in text:
        return marker + text.split(marker, 1)[1].strip("/")
    return text.rstrip("/")


def task_key(row: dict[str, str], seed: int) -> str:
    return f"{row['dataset_dir']}|seed={seed}"


def load_latest(status_path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not status_path.exists():
        return latest
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        key = item.get("task_key")
        if isinstance(key, str):
            latest[key] = item
    return latest


def identity_error(row: dict[str, str], payload: dict[str, Any]) -> str:
    expected = normalize_dataset_identity(row.get("dataset_rel") or row.get("dataset_dir"))
    actual = normalize_dataset_identity(payload.get("dataset_dir"))
    if expected and actual == expected:
        return ""
    check = payload.get("dataset_identity_check") if isinstance(payload.get("dataset_identity_check"), dict) else {}
    if check.get("match") is True:
        return ""
    return f"dataset_identity_mismatch:expected={expected};actual={actual}"


def collect_host(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = out_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"all_results_{args.host}.jsonl"
    csv_path = out_dir / f"result_index_{args.host}.csv"

    batch_by_wave = {w: f"e1_candidate200_seed{args.seed}_v2_{w.lower()}_{args.stamp}" for w in ["W1", "W2", "W3", "W4", "W5"]}
    manifest_path = root / "exp-planning/02.E1选择验证/generated/wave_manifest.csv"
    manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))

    rows_out: list[dict[str, Any]] = []
    total = copied = missing = invalid = 0
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for job in manifest:
            if job["host"] != args.host:
                continue
            wave = job["wave"]
            tool = job["tool"]
            batch = batch_by_wave[wave]
            slice_rows = list(csv.DictReader((root / job["slice_rel"]).open(encoding="utf-8", newline="")))
            expected = {task_key(row, args.seed): row for row in slice_rows}
            status_path = root / "experiments" / batch / args.host / "__launcher__" / "task_status.jsonl"
            latest = load_latest(status_path)

            for key, row in expected.items():
                total += 1
                item = latest.get(key)
                status = str(item.get("status")) if item else "missing_status"
                result_path = item.get("result_path") if item else None
                global_index = str(row.get("global_index") or "x")
                dataset_name = row.get("dataset_name") or row.get("basename") or Path(row["dataset_dir"]).name
                rel_result = Path(wave) / tool / args.host / f"g{int(global_index):04d}_{slug(dataset_name)}" / "result.json"
                dst = results_root / rel_result
                error = ""
                payload = None

                if not item:
                    missing += 1
                    error = "missing_status"
                elif status not in DONE:
                    invalid += 1
                    error = f"unexpected_status:{status}"
                elif not isinstance(result_path, str) or not result_path:
                    missing += 1
                    error = "missing_result_path"
                elif not Path(result_path).is_file():
                    missing += 1
                    error = f"missing_result_file:{result_path}"
                else:
                    try:
                        payload = json.loads(Path(result_path).read_text(encoding="utf-8", errors="ignore"))
                    except Exception as exc:
                        invalid += 1
                        error = f"bad_result_json:{exc!r}"
                    else:
                        error = identity_error(row, payload)
                        if error:
                            invalid += 1
                        else:
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(result_path, dst)
                            copied += 1
                            record = {
                                "wave": wave,
                                "tool": tool,
                                "host": args.host,
                                "global_index": int(global_index),
                                "dataset_name": dataset_name,
                                "dataset_dir": row["dataset_dir"],
                                "task_key": key,
                                "status": status,
                                "seconds": item.get("seconds"),
                                "result_relpath": str(Path("results") / rel_result),
                                "source_result_path": result_path,
                                "rerun_reason": item.get("rerun_reason"),
                                "result": payload,
                            }
                            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                rows_out.append(
                    {
                        "wave": wave,
                        "tool": tool,
                        "host": args.host,
                        "global_index": global_index,
                        "dataset_name": dataset_name,
                        "dataset_dir": row["dataset_dir"],
                        "task_key": key,
                        "status": status,
                        "seconds": "" if not item else item.get("seconds"),
                        "result_relpath": "" if error else str(Path("results") / rel_result),
                        "source_result_path": "" if not result_path else result_path,
                        "rerun_reason": "" if not item else item.get("rerun_reason") or "",
                        "archive_error": error,
                    }
                )

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        fieldnames = [
            "wave",
            "tool",
            "host",
            "global_index",
            "dataset_name",
            "dataset_dir",
            "task_key",
            "status",
            "seconds",
            "result_relpath",
            "source_result_path",
            "rerun_reason",
            "archive_error",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "host": args.host,
        "total": total,
        "copied": copied,
        "missing": missing,
        "invalid": invalid,
        "out_dir": str(out_dir),
    }
    (out_dir / f"summary_{args.host}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if copied == total and missing == 0 and invalid == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="收集 E1 远端结果并校验数据集身份")
    parser.add_argument("host")
    parser.add_argument("out_dir")
    parser.add_argument("--root", default="/home/zhangziwen/projects/scientific-intelligent-modelling")
    parser.add_argument("--stamp", default="20260424-041046")
    parser.add_argument("--seed", type=int, default=1314)
    return parser


def main() -> int:
    return collect_host(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
