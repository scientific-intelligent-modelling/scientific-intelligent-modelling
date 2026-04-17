#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
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


def _load_slice_dataset_dirs(slice_csv: Path) -> set[str]:
    with slice_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["dataset_dir"] for row in reader}


def _to_int_or_none(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_row(
    row: dict[str, Any],
    *,
    candidate_meta: dict[str, dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    dataset_dir = row.get("dataset_dir")
    normalized = {
        "host_label": row.get("host_label"),
        "method": row.get("method"),
        "seed": _to_int_or_none(row.get("seed")),
        "task_key": row.get("task_key"),
        "dataset_name": row.get("dataset_name"),
        "dataset_dir": dataset_dir,
        "global_index": _to_int_or_none(row.get("global_index")),
        "task_status": row.get("task_status"),
        "task_error": row.get("task_error"),
        "task_seconds": row.get("task_seconds"),
        "finished_at": row.get("finished_at"),
        "log_path": row.get("log_path"),
        "experiment_dir": row.get("experiment_dir"),
        "result_path": row.get("result_path"),
        "result_status": row.get("result_status"),
        "result_error": row.get("result_error"),
        "canonical_artifact_present": row.get("canonical_artifact_present"),
        "recovered_after_timeout": row.get("recovered_after_timeout"),
        "valid_r2": row.get("valid_r2"),
        "valid_rmse": row.get("valid_rmse"),
        "valid_nmse": row.get("valid_nmse"),
        "id_r2": row.get("id_r2"),
        "id_rmse": row.get("id_rmse"),
        "id_nmse": row.get("id_nmse"),
        "ood_r2": row.get("ood_r2"),
        "ood_rmse": row.get("ood_rmse"),
        "ood_nmse": row.get("ood_nmse"),
        "result_source": source,
        "unified_timeout_seconds": 3600,
    }
    normalized.update(candidate_meta.get(dataset_dir, {}))
    return normalized


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _build_dataset_method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset_dir"], row["method"])
        group = groups.setdefault(
            key,
            {
                "dataset_dir": row["dataset_dir"],
                "dataset_name": row.get("dataset_name"),
                "method": row["method"],
                "candidate_pool": row.get("candidate_pool"),
                "family": row.get("family"),
                "subgroup": row.get("subgroup"),
                "basename": row.get("basename"),
                "selection_mode": row.get("selection_mode"),
                "candidate_advantage_side": row.get("candidate_advantage_side"),
                "candidate_gap_score": row.get("candidate_gap_score"),
                "seed_count": 0,
                "status_counter": Counter(),
                "id_r2_vals": [],
                "id_nmse_vals": [],
                "ood_r2_vals": [],
                "ood_nmse_vals": [],
            },
        )
        group["seed_count"] += 1
        group["status_counter"][row.get("result_status") or row.get("task_status") or "unknown"] += 1
        for field, bucket in [
            ("id_r2", "id_r2_vals"),
            ("id_nmse", "id_nmse_vals"),
            ("ood_r2", "ood_r2_vals"),
            ("ood_nmse", "ood_nmse_vals"),
        ]:
            value = _metric_float(row.get(field))
            if value is not None:
                group[bucket].append(value)

    summary_rows: list[dict[str, Any]] = []
    for group in groups.values():
        row = {k: v for k, v in group.items() if not k.endswith("_vals") and k != "status_counter"}
        for status, count in sorted(group["status_counter"].items()):
            row[f"status_{status}"] = count
        for field, bucket in [
            ("id_r2", "id_r2_vals"),
            ("id_nmse", "id_nmse_vals"),
            ("ood_r2", "ood_r2_vals"),
            ("ood_nmse", "ood_nmse_vals"),
        ]:
            vals = group[bucket]
            row[f"{field}_mean"] = sum(vals) / len(vals) if vals else None
            row[f"{field}_count"] = len(vals)
        summary_rows.append(row)
    summary_rows.sort(key=lambda x: (x["method"], x["dataset_dir"]))
    return summary_rows


def _build_dataset_compare(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        rec = by_dataset.setdefault(row["dataset_dir"], {})
        for key in [
            "dataset_dir",
            "dataset_name",
            "candidate_pool",
            "family",
            "subgroup",
            "basename",
            "selection_mode",
            "candidate_advantage_side",
            "candidate_gap_score",
        ]:
            if row.get(key) not in (None, ""):
                rec[key] = row.get(key)
        method = row["method"]
        for key, value in row.items():
            if key in {
                "dataset_dir",
                "dataset_name",
                "candidate_pool",
                "family",
                "subgroup",
                "basename",
                "selection_mode",
                "candidate_advantage_side",
                "candidate_gap_score",
                "method",
            }:
                continue
            rec[f"{method}_{key}"] = value
    rows = list(by_dataset.values())
    rows.sort(key=lambda x: x["dataset_dir"])
    return rows


def _write_summary_md(path: Path, rows: list[dict[str, Any]], dataset_compare_rows: list[dict[str, Any]]) -> None:
    method_seed_status = defaultdict(Counter)
    method_cov = defaultdict(Counter)
    for row in rows:
        method_seed_status[(row["method"], row["seed"])][row.get("result_status") or row.get("task_status") or "unknown"] += 1
        if row.get("id_r2") not in (None, "", "None") and row.get("ood_r2") not in (None, "", "None"):
            method_cov[row["method"]]["full_id_ood"] += 1
        elif row.get("canonical_artifact_present"):
            method_cov[row["method"]]["equation_only_or_partial"] += 1
        else:
            method_cov[row["method"]]["no_equation"] += 1

    lines = [
        "# 三 seed 正式结果汇总（统一 1h 口径）",
        "",
        f"- 任务数：`{len(rows)}`",
        f"- 数据集对照行数：`{len(dataset_compare_rows)}`",
        "",
        "## 方法 × seed × 状态",
        "",
    ]
    for (method, seed), counter in sorted(method_seed_status.items()):
        lines.append(f"### `{method}` / `seed={seed}`")
        for status, count in sorted(counter.items()):
            lines.append(f"- `{status}`: `{count}`")
        lines.append("")
    lines.append("## 指标覆盖")
    lines.append("")
    for method, counter in sorted(method_cov.items()):
        lines.append(f"### `{method}`")
        for key, count in sorted(counter.items()):
            lines.append(f"- `{key}`: `{count}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 formal200 三个 seed 的正式结果（统一 1h 口径）")
    parser.add_argument("--two-seed-csv", required=True)
    parser.add_argument("--seed522-original-dir", required=True)
    parser.add_argument("--pysr-rerun-dir", required=True)
    parser.add_argument("--slice01-csv", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    two_seed_rows = _read_csv_rows(Path(args.two_seed_csv))
    seed522_rows = _read_jsonl_rows(Path(args.seed522_original_dir))
    rerun_rows = _read_jsonl_rows(Path(args.pysr_rerun_dir))
    slice01_dirs = _load_slice_dataset_dirs(Path(args.slice01_csv))
    candidate_meta = _load_candidate_meta(Path(args.candidate_json))

    final_rows: list[dict[str, Any]] = []

    # 520/521: use existing rows, but replace pysr slice_01 with rerun 1h rows.
    for row in two_seed_rows:
        seed = _to_int_or_none(row.get("seed"))
        method = row.get("method")
        dataset_dir = row.get("dataset_dir")
        if method == "pysr" and seed in {520, 521} and dataset_dir in slice01_dirs:
            continue
        final_rows.append(_normalize_row(row, candidate_meta=candidate_meta, source="formal520_521_original"))

    # 522 original: keep llmsr all rows and pysr rows outside slice_01.
    for row in seed522_rows:
        seed = _to_int_or_none(row.get("seed"))
        if seed != 522:
            continue
        method = row.get("method")
        dataset_dir = row.get("dataset_dir")
        if method == "pysr" and dataset_dir in slice01_dirs:
            continue
        final_rows.append(_normalize_row(row, candidate_meta=candidate_meta, source="formal522_original"))

    # Rerun rows: authoritative slice_01 1h pysr rows for 520/521/522.
    for row in rerun_rows:
        final_rows.append(_normalize_row(row, candidate_meta=candidate_meta, source="pysr_1h_rerun"))

    final_rows.sort(key=lambda x: (x["method"], x["seed"], x["dataset_dir"], x.get("host_label") or ""))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_csv = output_dir / "three_seed_formal_task_results.csv"
    summary_csv = output_dir / "three_seed_formal_dataset_method_summary.csv"
    compare_csv = output_dir / "three_seed_formal_dataset_compare.csv"
    summary_md = output_dir / "three_seed_formal_summary.md"

    _write_csv(task_csv, final_rows)
    dataset_method_summary = _build_dataset_method_summary(final_rows)
    _write_csv(summary_csv, dataset_method_summary)
    dataset_compare = _build_dataset_compare(dataset_method_summary)
    _write_csv(compare_csv, dataset_compare)
    _write_summary_md(summary_md, final_rows, dataset_compare)

    print(
        json.dumps(
            {
                "task_rows": len(final_rows),
                "dataset_compare_rows": len(dataset_compare),
                "task_csv": str(task_csv),
                "summary_csv": str(summary_csv),
                "compare_csv": str(compare_csv),
                "summary_md": str(summary_md),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
