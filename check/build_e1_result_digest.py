#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "e1_final_results_20260424-041046_clean"
CANDIDATE200_CSV = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "generated" / "candidate200_unified.csv"
OUTPUT_DIR = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "e1_result_digest_20260424-041046"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _scalar(v: Any) -> Any:
    if isinstance(v, (str, int, float)) or v is None:
        return v
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def _num(v: Any) -> float | None:
    if v in ("", None, "None"):
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nested(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _bool_text(v: bool) -> str:
    return "true" if v else "false"


def _candidate_index() -> dict[int, dict[str, str]]:
    rows = _load_csv(CANDIDATE200_CSV)
    return {int(row["global_index"]): row for row in rows}


def _timeout_type(
    task_status: str,
    result_status: str,
    valid_output: bool,
    has_expression: bool,
    artifact_valid: str,
    has_full_metrics: bool,
) -> str:
    if task_status != "timed_out" and result_status != "timed_out":
        return "not_timeout"
    if valid_output:
        return "budget_exhausted_with_output"
    if has_expression and artifact_valid == "true":
        return "partial_output" if not has_full_metrics else "invalid_output"
    if has_expression:
        return "unvalidated_expression"
    return "no_valid_output"


def _digest_row(raw: dict[str, Any], candidates: dict[int, dict[str, str]]) -> dict[str, Any]:
    result = raw.get("result") or {}
    global_index = int(raw["global_index"])
    candidate = candidates[global_index]
    artifact = result.get("canonical_artifact") if isinstance(result.get("canonical_artifact"), dict) else {}

    method = raw.get("tool") or result.get("tool") or ""
    task_status = raw.get("status") or ""
    result_status = result.get("status") or ""
    status = task_status or result_status
    seed = result.get("seed") or 1314
    expression = result.get("equation") or ""
    has_expression = bool(expression)

    valid_nmse = _num(_nested(result, "valid", "nmse"))
    id_nmse = _num(_nested(result, "id_test", "nmse"))
    ood_nmse = _num(_nested(result, "ood_test", "nmse"))
    id_r2 = _num(_nested(result, "id_test", "r2"))
    ood_r2 = _num(_nested(result, "ood_test", "r2"))
    has_full_metrics = id_nmse is not None and ood_nmse is not None

    artifact_valid_raw = artifact.get("artifact_valid")
    artifact_valid = "" if artifact_valid_raw is None else _bool_text(bool(artifact_valid_raw))
    artifact_ok = artifact_valid_raw is not False
    valid_output = bool(has_expression and artifact_ok and has_full_metrics)

    complexity = artifact.get("ast_node_count")
    tree_depth = artifact.get("tree_depth")
    variables = artifact.get("variables") or []
    operator_set = artifact.get("operator_set") or []

    return {
        "dataset_id": f"g{global_index:04d}",
        "global_index": global_index,
        "dataset_name": raw.get("dataset_name") or result.get("dataset") or candidate.get("dataset_name", ""),
        "dataset_dir": candidate.get("dataset_rel") or candidate.get("dataset_dir") or raw.get("dataset_dir", ""),
        "family": candidate.get("family", ""),
        "subgroup": candidate.get("subgroup", ""),
        "basename": candidate.get("basename", ""),
        "pool": candidate.get("pool", ""),
        "selection_mode": candidate.get("selection_mode", ""),
        "candidate_advantage_side": candidate.get("candidate_advantage_side", ""),
        "method": method,
        "seed": seed,
        "wave": raw.get("wave", ""),
        "host": raw.get("host", ""),
        "status": status,
        "task_status": task_status,
        "result_status": result_status,
        "status_mismatch": _bool_text(bool(task_status and result_status and task_status != result_status)),
        "valid_output": _bool_text(valid_output),
        "has_full_metrics": _bool_text(has_full_metrics),
        "has_expression": _bool_text(has_expression),
        "artifact_valid": artifact_valid,
        "timeout_type": _timeout_type(task_status, result_status, valid_output, has_expression, artifact_valid, has_full_metrics),
        "valid_nmse": valid_nmse,
        "id_nmse": id_nmse,
        "ood_nmse": ood_nmse,
        "id_r2": id_r2,
        "ood_r2": ood_r2,
        "runtime": _num(result.get("seconds")) or _num(raw.get("seconds")),
        "expression": expression,
        "expression_chars": len(expression),
        "complexity": complexity,
        "tree_depth": tree_depth,
        "variable_count": len(variables),
        "variables": ";".join(str(v) for v in variables),
        "operator_set": ";".join(str(v) for v in operator_set),
        "equation_count": result.get("equation_count"),
        "error": result.get("error") or raw.get("archive_error") or "",
        "canonical_artifact_error": result.get("canonical_artifact_error") or "",
        "result_relpath": raw.get("result_relpath", ""),
        "source_result_path": raw.get("source_result_path", ""),
        "rerun_reason": raw.get("rerun_reason") or "",
    }


def _counter_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    counters: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        counters[key]["total"] += 1
        counters[key][f"status_{row['status']}"] += 1
        counters[key][f"result_status_{row['result_status']}"] += 1
        counters[key][f"timeout_type_{row['timeout_type']}"] += 1
        if row["valid_output"] == "true":
            counters[key]["valid_output"] += 1
        if row["has_full_metrics"] == "true":
            counters[key]["has_full_metrics"] += 1
        if row["has_expression"] == "true":
            counters[key]["has_expression"] += 1
        if row["status_mismatch"] == "true":
            counters[key]["status_mismatch"] += 1

    out: list[dict[str, Any]] = []
    all_counter_keys = sorted({k for counter in counters.values() for k in counter})
    for key, counter in sorted(counters.items()):
        row = {field: value for field, value in zip(group_fields, key)}
        row.update({name: counter.get(name, 0) for name in all_counter_keys})
        row["valid_output_rate"] = counter.get("valid_output", 0) / counter["total"]
        out.append(row)
    return out


def _dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["global_index"])].append(row)

    out: list[dict[str, Any]] = []
    for _, group in sorted(groups.items()):
        first = group[0]
        status_counter = Counter(row["status"] for row in group)
        valid_methods = sorted(row["method"] for row in group if row["valid_output"] == "true")
        out.append(
            {
                "dataset_id": first["dataset_id"],
                "global_index": first["global_index"],
                "dataset_name": first["dataset_name"],
                "dataset_dir": first["dataset_dir"],
                "family": first["family"],
                "subgroup": first["subgroup"],
                "selection_mode": first["selection_mode"],
                "candidate_advantage_side": first["candidate_advantage_side"],
                "method_count": len(group),
                "valid_output_methods": len(valid_methods),
                "valid_output_method_names": ";".join(valid_methods),
                "status_ok": status_counter.get("ok", 0),
                "status_timed_out": status_counter.get("timed_out", 0),
                "status_error": status_counter.get("error", 0),
            }
        )
    return out


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    method_rows = _counter_rows(rows, ["method"])
    method_family_rows = _counter_rows(rows, ["method", "family"])
    dataset_rows = _dataset_rows(rows)
    status_mismatch_rows = [row for row in rows if row["status_mismatch"] == "true"]
    nonvalid_rows = [row for row in rows if row["valid_output"] != "true"]

    total = len(rows)
    valid = sum(1 for row in rows if row["valid_output"] == "true")
    full_metrics = sum(1 for row in rows if row["has_full_metrics"] == "true")
    status_counter = Counter(row["status"] for row in rows)
    result_status_counter = Counter(row["result_status"] for row in rows)
    timeout_counter = Counter(row["timeout_type"] for row in rows)
    status_mismatch = sum(1 for row in rows if row["status_mismatch"] == "true")

    lines = [
        "# E1 Result Digest",
        "",
        "## Scope",
        "",
        "- Input archive: `exp-planning/02.E1选择验证/e1_final_results_20260424-041046_clean/`",
        "- Unit: one row per `(dataset, method, seed)` run.",
        "- `valid_output=true` means the run has a final expression, no invalid canonical artifact flag, and finite `id_nmse` plus `ood_nmse`.",
        "- `timed_out` runs are not automatically discarded; `budget_exhausted_with_output` runs remain usable for rank-fidelity analysis.",
        "",
        "## Totals",
        "",
        f"- Total rows: `{total}`",
        f"- Valid output rows: `{valid}`",
        f"- Rows with finite ID/OOD NMSE: `{full_metrics}`",
        f"- Official launcher status: `{dict(sorted(status_counter.items()))}`",
        f"- Internal result.json status: `{dict(sorted(result_status_counter.items()))}`",
        f"- Status mismatch rows: `{status_mismatch}`",
        f"- Timeout type: `{dict(sorted(timeout_counter.items()))}`",
        "",
        "## Files",
        "",
        "- `e1_result_table.csv`: full 7 algorithm x 200 candidate table.",
        "- `e1_method_summary.csv`: method-level status and validity summary.",
        "- `e1_method_family_summary.csv`: method x family status and validity summary.",
        "- `e1_dataset_coverage.csv`: per-dataset valid method coverage.",
        "- `e1_status_mismatch.csv`: rows whose launcher status differs from internal `result.json` status.",
        "- `e1_nonvalid_cases.csv`: rows excluded by `valid_output=false`.",
        "",
        "## Method Summary",
        "",
        "| method | total | valid_output | valid_output_rate | status_ok | status_timed_out | status_mismatch | budget_exhausted_with_output | partial_output | no_valid_output |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in method_rows:
        lines.append(
            "| {method} | {total} | {valid_output} | {valid_output_rate:.3f} | {status_ok} | {status_timed_out} | {status_mismatch} | {budget} | {partial} | {none} |".format(
                method=row["method"],
                total=row.get("total", 0),
                valid_output=row.get("valid_output", 0),
                valid_output_rate=row.get("valid_output_rate", 0.0),
                status_ok=row.get("status_ok", 0),
                status_timed_out=row.get("status_timed_out", 0),
                status_mismatch=row.get("status_mismatch", 0),
                budget=row.get("timeout_type_budget_exhausted_with_output", 0),
                partial=row.get("timeout_type_partial_output", 0),
                none=row.get("timeout_type_no_valid_output", 0),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e1_result_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(output_dir / "e1_method_summary.csv", method_rows)
    _write_csv(output_dir / "e1_method_family_summary.csv", method_family_rows)
    _write_csv(output_dir / "e1_dataset_coverage.csv", dataset_rows)
    _write_csv(output_dir / "e1_status_mismatch.csv", status_mismatch_rows)
    _write_csv(output_dir / "e1_nonvalid_cases.csv", nonvalid_rows)


def build() -> None:
    candidates = _candidate_index()
    rows: list[dict[str, Any]] = []
    with (E1_ROOT / "all_results.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(_digest_row(json.loads(line), candidates))

    rows.sort(key=lambda row: (int(row["global_index"]), str(row["method"])))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "e1_result_table.csv", [{k: _scalar(v) for k, v in row.items()} for row in rows])
    _write_summary(rows, OUTPUT_DIR)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    build()
