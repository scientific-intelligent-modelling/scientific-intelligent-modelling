#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_build_canonical_artifact
from scientific_intelligent_modelling.benchmarks.runner import (
    _evaluate_prediction,
    _predict_from_canonical_artifact,
    load_canonical_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "e1_final_results_20260424-041046_clean"
DEFAULT_CANDIDATE_CSV = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "generated" / "candidate200_unified.csv"
DEFAULT_DIGEST_TABLE = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "e1_result_digest_20260424-041046" / "e1_result_table.csv"
DEFAULT_REPORT_JSON = DEFAULT_ARCHIVE_ROOT / "summary" / "exact_metric_backfill_20260426.json"
DEFAULT_TOOLS = ("pysr", "pyoperon", "dso", "tpsr")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "total_targeted": len(rows),
        "repaired": sum(1 for row in rows if row["action"] == "repaired"),
        "skipped_valid": sum(1 for row in rows if row["action"] == "skipped_valid"),
        "unrepaired": sum(1 for row in rows if row["action"] == "unrepaired"),
        "rows": rows,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _has_full_metrics(result: dict[str, Any]) -> bool:
    return _finite_number((result.get("id_test") or {}).get("nmse")) and _finite_number(
        (result.get("ood_test") or {}).get("nmse")
    )


def _is_valid_output(result: dict[str, Any]) -> bool:
    artifact = result.get("canonical_artifact") if isinstance(result.get("canonical_artifact"), dict) else {}
    return bool(result.get("equation")) and artifact.get("artifact_valid") is not False and _has_full_metrics(result)


def _candidate_paths(path: Path) -> dict[int, Path]:
    return {int(row["global_index"]): REPO_ROOT / row["dataset_rel"] for row in _load_csv(path)}


def _target_relpaths(path: Path, tools: set[str]) -> set[str]:
    targets: set[str] = set()
    for row in _load_csv(path):
        if row.get("method") not in tools:
            continue
        if row.get("valid_output") == "true":
            continue
        if row.get("dataset_identity_status") != "exact_match":
            continue
        relpath = str(row.get("result_relpath") or "").strip()
        if relpath:
            targets.add(relpath)
    return targets


def _repair_result(tool: str, result: dict[str, Any], dataset_dir: Path) -> tuple[dict[str, Any] | None, str]:
    equation = str(result.get("equation") or "").strip()
    if not equation:
        return None, "missing_equation"

    dataset = load_canonical_dataset(dataset_dir)
    artifact, artifact_error = safe_build_canonical_artifact(
        tool_name=tool,
        equation=equation,
        expected_n_features=len(dataset.feature_names),
    )
    if artifact is None:
        return None, f"artifact_error:{artifact_error}"
    if artifact.get("artifact_valid") is False:
        return None, "artifact_invalid:" + ";".join(str(x) for x in artifact.get("validation_errors") or [])

    try:
        valid_pred = _predict_from_canonical_artifact(artifact, dataset.valid.X) if dataset.valid else None
        id_pred = _predict_from_canonical_artifact(artifact, dataset.id_test.X) if dataset.id_test else None
        ood_pred = _predict_from_canonical_artifact(artifact, dataset.ood_test.X) if dataset.ood_test else None
        valid_metrics = _evaluate_prediction(dataset.valid, valid_pred)
        id_metrics = _evaluate_prediction(dataset.id_test, id_pred)
        ood_metrics = _evaluate_prediction(dataset.ood_test, ood_pred)
    except Exception as exc:
        return None, f"metric_error:{exc.__class__.__name__}:{exc}"

    if not (
        id_metrics
        and ood_metrics
        and _finite_number(id_metrics.get("nmse"))
        and _finite_number(ood_metrics.get("nmse"))
    ):
        return None, "metric_nonfinite"

    repaired = dict(result)
    repaired["canonical_artifact"] = artifact
    repaired["canonical_artifact_error"] = None
    repaired["valid"] = valid_metrics
    repaired["id_test"] = id_metrics
    repaired["ood_test"] = ood_metrics
    repaired["posthoc_repair"] = {
        "name": "exact_match_metric_backfill",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "Recomputed canonical artifact and valid/id/ood metrics for exact-match E1 archive rows.",
        "source_script": "check/backfill_e1_exact_result_metrics.py",
    }
    return repaired, "repaired"


def backfill(args: argparse.Namespace) -> int:
    archive_root = Path(args.archive_root).resolve()
    tools = {str(tool).strip() for tool in (args.tool or DEFAULT_TOOLS)}
    candidate_paths = _candidate_paths(Path(args.candidate_csv).resolve())
    target_relpaths = _target_relpaths(Path(args.digest_table).resolve(), tools)
    all_results_path = archive_root / "all_results.jsonl"
    raw_rows = [json.loads(line) for line in all_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    report_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        tool = str(raw.get("tool") or "")
        result_relpath = str(raw.get("result_relpath") or "").strip()
        if tool not in tools or result_relpath not in target_relpaths:
            continue

        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        global_index = int(raw["global_index"])
        base = {
            "method": tool,
            "dataset_id": f"g{global_index:04d}",
            "global_index": global_index,
            "dataset_name": raw.get("dataset_name") or result.get("dataset") or "",
            "host": raw.get("host") or "",
            "result_relpath": result_relpath,
        }

        if _is_valid_output(result):
            report_rows.append({**base, "action": "skipped_valid", "reason": "already_valid"})
            continue

        repaired, reason = _repair_result(tool, result, candidate_paths[global_index])
        if repaired is None:
            report_rows.append({**base, "action": "unrepaired", "reason": reason})
            continue

        raw["result"] = repaired
        result_path = archive_root / result_relpath
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_rows.append({**base, "action": "repaired", "reason": reason})

    _write_jsonl(all_results_path, raw_rows)
    _write_report(Path(args.report_json).resolve(), report_rows)
    result = {
        "target_rows": len(report_rows),
        "repaired": sum(1 for row in report_rows if row["action"] == "repaired"),
        "unrepaired": sum(1 for row in report_rows if row["action"] == "unrepaired"),
        "report_json": str(Path(args.report_json).resolve()),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回填 E1 exact-match 结果的 canonical artifact 与指标")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--digest-table", default=str(DEFAULT_DIGEST_TABLE))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--tool", action="append", choices=sorted(DEFAULT_TOOLS), help="只处理指定工具，可重复传参")
    return parser


def main() -> int:
    return backfill(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
