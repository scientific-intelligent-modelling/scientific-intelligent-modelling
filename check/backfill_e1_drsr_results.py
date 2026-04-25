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
DIGEST_TABLE = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "e1_result_digest_20260424-041046" / "e1_result_table.csv"
REPORT_DIR = E1_ROOT / "summary"
REPORT_JSON = REPORT_DIR / "drsr_backfill_20260426.json"
REPORT_CSV = REPORT_DIR / "drsr_backfill_20260426.csv"


# g0057 的原始 result.json 没把 DRSR best sample 的 params 带入 canonical artifact。
# 参数来自远端 iaaccn27 对应实验目录的 best_history/best_sample_0.json。
DRSR_PARAM_OVERRIDES: dict[int, dict[str, Any]] = {
    57: {
        "params": [
            6.6004375005997415,
            -0.3308097461621048,
            -1.363183971474182,
            -0.8465338458068004,
            0.008881302344160513,
            -0.2615890938584442,
            -0.44627006066666786,
            0.5454755536915996,
            -0.8715198373639206,
            -0.7293962736904025,
        ],
        "source": "iaaccn27:best_history/best_sample_0.json",
    }
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_candidate_paths() -> dict[int, Path]:
    return {int(row["global_index"]): REPO_ROOT / row["dataset_rel"] for row in _load_csv(CANDIDATE200_CSV)}


def _load_target_rows() -> set[str]:
    """只选择身份已确认的 DRSR 非有效结果，避免误修同名串档行。"""
    target_relpaths: set[str] = set()
    for row in _load_csv(DIGEST_TABLE):
        if row.get("method") != "drsr":
            continue
        if row.get("valid_output") == "true":
            continue
        if row.get("dataset_identity_status") != "exact_match":
            continue
        relpath = str(row.get("result_relpath") or "").strip()
        if relpath:
            target_relpaths.add(relpath)
    return target_relpaths


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


def _as_float_list(values: Any) -> list[float] | None:
    if not isinstance(values, list):
        return None
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            return None
    return out


def _parameter_values(result: dict[str, Any], global_index: int) -> tuple[list[float] | None, str]:
    artifact = result.get("canonical_artifact") if isinstance(result.get("canonical_artifact"), dict) else {}
    for key in ("parameter_values", "params", "fitted_params"):
        values = _as_float_list(artifact.get(key))
        if values is not None:
            return values, f"canonical_artifact.{key}"

    for key in ("parameter_values", "fitted_params"):
        values = _as_float_list(result.get(key))
        if values is not None:
            return values, key

    override = DRSR_PARAM_OVERRIDES.get(global_index)
    if override:
        values = _as_float_list(override.get("params"))
        if values is not None:
            return values, str(override.get("source") or "manual_override")

    return None, "missing"


def _repair_result(
    *,
    result: dict[str, Any],
    dataset_dir: Path,
    global_index: int,
) -> tuple[dict[str, Any] | None, str, str]:
    equation = str(result.get("equation") or "").strip()
    if not equation:
        return None, "missing_equation", "missing"

    params, param_source = _parameter_values(result, global_index)
    dataset = load_canonical_dataset(dataset_dir)
    artifact, artifact_error = safe_build_canonical_artifact(
        tool_name="drsr",
        equation=equation,
        expected_n_features=len(dataset.feature_names),
        parameter_values=params,
    )
    if artifact is None:
        return None, f"artifact_error:{artifact_error}", param_source
    if artifact.get("artifact_valid") is False:
        return None, "artifact_invalid:" + ";".join(str(x) for x in artifact.get("validation_errors") or []), param_source

    try:
        valid_pred = _predict_from_canonical_artifact(artifact, dataset.valid.X) if dataset.valid else None
        id_pred = _predict_from_canonical_artifact(artifact, dataset.id_test.X) if dataset.id_test else None
        ood_pred = _predict_from_canonical_artifact(artifact, dataset.ood_test.X) if dataset.ood_test else None
        valid_metrics = _evaluate_prediction(dataset.valid, valid_pred)
        id_metrics = _evaluate_prediction(dataset.id_test, id_pred)
        ood_metrics = _evaluate_prediction(dataset.ood_test, ood_pred)
    except Exception as exc:
        return None, f"metric_error:{exc.__class__.__name__}:{exc}", param_source

    if not (
        id_metrics
        and ood_metrics
        and _finite_number(id_metrics.get("nmse"))
        and _finite_number(ood_metrics.get("nmse"))
    ):
        return None, "metric_nonfinite", param_source

    repaired = dict(result)
    repaired["canonical_artifact"] = artifact
    repaired["canonical_artifact_error"] = None
    repaired["valid"] = valid_metrics
    repaired["id_test"] = id_metrics
    repaired["ood_test"] = ood_metrics
    repaired["posthoc_repair"] = {
        "name": "drsr_signature_arg_param_backfill",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "Recomputed DRSR timeout outputs after function-signature variable mapping and parameter backfill.",
        "parameter_source": param_source,
        "source_script": "check/backfill_e1_drsr_results.py",
    }
    return repaired, "repaired", param_source


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
        "total_targeted": len(rows),
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
    target_relpaths = _load_target_rows()
    all_results_path = E1_ROOT / "all_results.jsonl"
    raw_rows = [json.loads(line) for line in all_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report_rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        if raw.get("tool") != "drsr":
            continue
        result_relpath = str(raw.get("result_relpath") or "").strip()
        if result_relpath not in target_relpaths:
            continue

        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        global_index = int(raw["global_index"])
        base_report = {
            "dataset_id": f"g{global_index:04d}",
            "global_index": global_index,
            "dataset_name": raw.get("dataset_name") or result.get("dataset") or "",
            "host": raw.get("host") or "",
            "result_relpath": result_relpath,
        }

        if _is_valid_output(result):
            report_rows.append(
                {**base_report, "action": "skipped_valid", "reason": "already_valid", "parameter_source": ""}
            )
            continue

        dataset_dir = candidate_paths[global_index]
        repaired, reason, param_source = _repair_result(
            result=result,
            dataset_dir=dataset_dir,
            global_index=global_index,
        )
        if repaired is None:
            report_rows.append(
                {**base_report, "action": "unrepaired", "reason": reason, "parameter_source": param_source}
            )
            continue

        raw["result"] = repaired
        result_path = E1_ROOT / result_relpath
        _write_result_file(result_path, repaired)
        report_rows.append({**base_report, "action": "repaired", "reason": reason, "parameter_source": param_source})

    _write_jsonl(all_results_path, raw_rows)
    _write_report(report_rows)
    print(
        json.dumps(
            {
                "drsr_target_rows": len(report_rows),
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
