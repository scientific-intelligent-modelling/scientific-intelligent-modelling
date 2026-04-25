#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "e1_final_results_20260424-041046_clean"
DEFAULT_SLICE_CSV = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "generated" / "rerun" / "identity_drsr_20260426.csv"


def _normalize_dataset_identity(path: Any) -> str:
    text = "" if path is None else str(path).strip().replace("\\", "/")
    marker = "sim-datasets-data/"
    if marker in text:
        return marker + text.split(marker, 1)[1].strip("/")
    return text.rstrip("/")


def _load_slice(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["tool"], int(row["global_index"])): row for row in rows}


def _find_result(rerun_root: Path, tool: str, global_index: int, dataset_name: str) -> Path | None:
    label = f"g{global_index:04d}_{dataset_name}"
    candidates = [
        rerun_root / tool / label / "result.json",
        rerun_root / label / "result.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(
        path
        for path in rerun_root.glob(f"**/{label}/result.json")
        if tool in path.parts
    )
    return matches[0] if matches else None


def _identity_matches(payload: dict[str, Any], slice_row: dict[str, str]) -> bool:
    check = payload.get("dataset_identity_check") if isinstance(payload.get("dataset_identity_check"), dict) else {}
    if check.get("match") is True:
        return True
    expected = _normalize_dataset_identity(slice_row.get("dataset_rel") or slice_row.get("dataset_dir"))
    actual = _normalize_dataset_identity(payload.get("dataset_dir"))
    return bool(expected and actual == expected)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def ingest(args: argparse.Namespace) -> int:
    archive_root = Path(args.archive_root).resolve()
    rerun_root = Path(args.rerun_root).resolve()
    slice_rows = _load_slice(Path(args.slice_csv).resolve())
    all_results_path = archive_root / "all_results.jsonl"
    raw_rows = [json.loads(line) for line in all_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    report_rows: list[dict[str, Any]] = []
    replaced = 0
    for raw in raw_rows:
        key = (str(raw.get("tool")), int(raw.get("global_index")))
        if key not in slice_rows:
            continue
        slice_row = slice_rows[key]
        result_path = _find_result(
            rerun_root,
            key[0],
            key[1],
            slice_row.get("dataset_name") or raw.get("dataset_name") or "",
        )
        base = {
            "method": key[0],
            "global_index": key[1],
            "dataset_name": slice_row.get("dataset_name") or raw.get("dataset_name") or "",
            "result_relpath": raw.get("result_relpath") or "",
        }
        if result_path is None:
            report_rows.append({**base, "action": "missing_rerun_result", "reason": "result_not_found"})
            continue

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not _identity_matches(payload, slice_row):
            report_rows.append(
                {
                    **base,
                    "action": "rejected",
                    "reason": "dataset_identity_mismatch",
                    "rerun_result_path": str(result_path),
                    "rerun_dataset_dir": payload.get("dataset_dir"),
                }
            )
            continue

        archive_result_path = archive_root / str(raw.get("result_relpath") or "")
        archive_result_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, archive_result_path)
        raw["result"] = payload
        raw["status"] = payload.get("status") or raw.get("status")
        raw["seconds"] = payload.get("seconds")
        raw["source_result_path"] = str(result_path)
        raw["rerun_reason"] = args.rerun_reason
        raw["identity_rerun_ingested_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        replaced += 1
        report_rows.append(
            {
                **base,
                "action": "replaced",
                "reason": "ok",
                "rerun_result_path": str(result_path),
                "rerun_dataset_dir": payload.get("dataset_dir"),
                "status": payload.get("status"),
                "seconds": payload.get("seconds"),
            }
        )

    _write_jsonl(all_results_path, raw_rows)
    _write_report(archive_root / "summary" / "identity_rerun_ingest_20260426.json", report_rows)
    print(json.dumps({"checked": len(slice_rows), "replaced": replaced, "report_rows": len(report_rows)}, ensure_ascii=False))
    return 0 if replaced == len(slice_rows) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 E1 identity rerun 结果安全回填到 clean archive")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--slice-csv", default=str(DEFAULT_SLICE_CSV))
    parser.add_argument("--rerun-root", required=True)
    parser.add_argument("--rerun-reason", default="identity_collision_rerun_20260426")
    return parser


def main() -> int:
    return ingest(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
