#!/usr/bin/env python3
from __future__ import annotations

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
E1_ROOT = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "e1_final_results_20260424-041046_clean"
CANDIDATE200_CSV = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "generated" / "candidate200_unified.csv"
REPORT_DIR = E1_ROOT / "summary"
REPORT_JSON = REPORT_DIR / "gplearn_backfill_20260425.json"
REPORT_CSV = REPORT_DIR / "gplearn_backfill_20260425.csv"


def _load_candidate_paths() -> dict[int, Path]:
    with CANDIDATE200_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return {int(row["global_index"]): REPO_ROOT / row["dataset_rel"] for row in rows}


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
    artifact_valid = artifact.get("artifact_valid")
    return bool(result.get("equation")) and artifact_valid is not False and _has_full_metrics(result)


def _repair_result(result: dict[str, Any], dataset_dir: Path) -> tuple[dict[str, Any] | None, str]:
    equation = str(result.get("equation") or "").strip()
    if not equation:
        return None, "missing_equation"

    dataset = load_canonical_dataset(dataset_dir)
    artifact, artifact_error = safe_build_canonical_artifact(
        tool_name="gplearn",
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
        "name": "gplearn_protected_semantics_backfill",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "Recomputed gplearn timeout outputs with protected div/log/sqrt semantics.",
        "source_script": "check/backfill_e1_gplearn_results.py",
    }
    return repaired, "repaired"


def _write_result_file(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_report(rows: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "total_checked": len(rows),
        "repaired": sum(1 for row in rows if row["action"] == "repaired"),
        "skipped_valid": sum(1 for row in rows if row["action"] == "skipped_valid"),
        "unrepaired": sum(1 for row in rows if row["action"] == "unrepaired"),
        "rows": rows,
    }
    REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    candidate_paths = _load_candidate_paths()
    all_results_path = E1_ROOT / "all_results.jsonl"
    raw_rows = [json.loads(line) for line in all_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report_rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        if raw.get("tool") != "gplearn":
            continue

        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        base_report = {
            "dataset_id": f"g{int(raw['global_index']):04d}",
            "global_index": int(raw["global_index"]),
            "dataset_name": raw.get("dataset_name") or result.get("dataset") or "",
            "host": raw.get("host") or "",
            "result_relpath": raw.get("result_relpath") or "",
        }

        if _is_valid_output(result):
            report_rows.append({**base_report, "action": "skipped_valid", "reason": "already_valid"})
            continue

        dataset_dir = candidate_paths[int(raw["global_index"])]
        repaired, reason = _repair_result(result, dataset_dir)
        if repaired is None:
            report_rows.append({**base_report, "action": "unrepaired", "reason": reason})
            continue

        raw["result"] = repaired
        result_path = E1_ROOT / str(raw.get("result_relpath") or "")
        _write_result_file(result_path, repaired)
        report_rows.append({**base_report, "action": "repaired", "reason": reason})

    _write_jsonl(all_results_path, raw_rows)
    _write_report(report_rows)
    print(
        json.dumps(
            {
                "gplearn_rows": len(report_rows),
                "repaired": sum(1 for row in report_rows if row["action"] == "repaired"),
                "unrepaired": sum(1 for row in report_rows if row["action"] == "unrepaired"),
                "report_json": str(REPORT_JSON),
                "report_csv": str(REPORT_CSV),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
