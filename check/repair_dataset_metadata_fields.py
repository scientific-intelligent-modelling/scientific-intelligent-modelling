#!/usr/bin/env python3
"""Audit and repair metadata fields that can be derived from local dataset files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


REQUIRED_SPLITS = ("train", "valid", "id_test", "ood_test")
EXCLUDED_PARTS = {".git", "__pycache__", "tools", "tests", "unprocessed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill metadata fields that are mechanically derivable from existing "
            "CSV/formula files, and report fields that require human review."
        )
    )
    parser.add_argument("--data-root", default="sim-datasets-data", help="Dataset repository root.")
    parser.add_argument(
        "--scope-csv",
        action="append",
        default=[],
        help=(
            "CSV containing dataset_dir or dataset_rel. May be passed multiple times. "
            "When omitted, scans normalized directories under data-root."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write metadata.yaml changes. Default is dry-run only.",
    )
    parser.add_argument(
        "--report-dir",
        default="experiment-results/dataset_metadata_repair",
        help="Directory for audit reports.",
    )
    return parser.parse_args()


def is_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec/v1")


def normalized_rel(path: str) -> str:
    value = str(path).strip()
    marker = "sim-datasets-data/"
    if value.startswith("/home/") and marker in value:
        value = value[value.index(marker) :]
    if value.startswith(marker):
        value = value[len(marker) :]
    return value.strip("/")


def dataset_dirs_from_scope_csv(data_root: Path, csv_path: Path) -> list[Path]:
    rows = pd.read_csv(csv_path)
    if "dataset_rel" in rows.columns:
        column = "dataset_rel"
    elif "dataset_dir" in rows.columns:
        column = "dataset_dir"
    else:
        raise ValueError(f"{csv_path} must contain dataset_rel or dataset_dir")
    dirs: list[Path] = []
    for value in rows[column].dropna().astype(str):
        dirs.append(data_root / normalized_rel(value))
    return dirs


def discover_dataset_dirs(data_root: Path, scope_csvs: list[str]) -> list[Path]:
    if scope_csvs:
        dirs: list[Path] = []
        for csv_path in scope_csvs:
            dirs.extend(dataset_dirs_from_scope_csv(data_root, Path(csv_path)))
    else:
        dirs = [path.parent for path in data_root.rglob("metadata.yaml")]
        filtered = []
        for path in dirs:
            rel_parts = set(path.relative_to(data_root).parts)
            if rel_parts & EXCLUDED_PARTS:
                continue
            filtered.append(path)
        dirs = filtered
    return sorted(set(dirs))


def yaml_loader() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 120
    return yaml


def ensure_commented_map(value: Any | None = None) -> CommentedMap:
    mapping = CommentedMap()
    if isinstance(value, dict):
        for key, item in value.items():
            mapping[key] = item
    return mapping


def insert_after(mapping: CommentedMap, after_key: str, key: str, value: Any) -> None:
    if key in mapping:
        mapping[key] = value
        return
    keys = list(mapping.keys())
    index = keys.index(after_key) + 1 if after_key in mapping else len(keys)
    mapping.insert(index, key, value)


def insert_split_field(split_info: CommentedMap, key: str, value: Any) -> None:
    if key == "ratio":
        insert_after(split_info, "samples", key, value)
    elif key == "nmse":
        insert_after(split_info, "ratio", key, value)
    else:
        split_info[key] = value


def clean_number(value: float) -> int | float:
    if not math.isfinite(value):
        return value
    if abs(value - round(value)) < 1e-12:
        return int(round(value))
    return float(f"{value:.12g}")


def load_formula_function(formula_path: Path, target_name: str | None) -> Any | None:
    module_name = f"_sim_formula_{abs(hash(formula_path))}"
    spec = importlib.util.spec_from_file_location(module_name, formula_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    setattr(module, "np", np)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if target_name and callable(getattr(module, target_name, None)):
        return getattr(module, target_name)
    functions = [
        item
        for _, item in vars(module).items()
        if inspect.isfunction(item) and getattr(item, "__module__", None) == module_name
    ]
    return functions[0] if functions else None


def compute_nmse(dataset_dir: Path, split_file: str, feature_names: list[str], target_name: str) -> tuple[float | None, str | None]:
    formula_path = dataset_dir / "formula.py"
    csv_path = dataset_dir / split_file
    if not formula_path.exists() or not csv_path.exists() or is_pointer(csv_path):
        return None, "formula/csv unavailable or csv is LFS pointer"
    try:
        func = load_formula_function(formula_path, target_name)
        if func is None:
            return None, "no callable formula function"
        frame = pd.read_csv(csv_path)
        if target_name not in frame.columns:
            return None, "target column missing"
        if not feature_names:
            feature_names = [col for col in frame.columns if col != target_name]
        if any(col not in frame.columns for col in feature_names):
            return None, "feature column missing"
        args = [frame[col].to_numpy(dtype=float) for col in feature_names]
        y_true = frame[target_name].to_numpy(dtype=float)
        y_pred = np.asarray(func(*args), dtype=float)
        if y_pred.shape == ():
            y_pred = np.full_like(y_true, float(y_pred), dtype=float)
        y_pred = np.reshape(y_pred, y_true.shape)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return None, "no finite predictions"
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        mse = float(np.mean((y_true - y_pred) ** 2))
        var = float(np.mean((y_true - float(np.mean(y_true))) ** 2))
        if var == 0:
            return (0.0, None) if mse == 0 else (None, "zero target variance with nonzero formula error")
        value = mse / var
        return (value, None) if math.isfinite(value) else (None, "non-finite nmse")
    except Exception as exc:
        return None, f"formula evaluation failed: {exc}"


def compute_target_range(dataset_dir: Path, split_file: str, target_name: str) -> list[float] | None:
    csv_path = dataset_dir / split_file
    if not csv_path.exists() or is_pointer(csv_path):
        return None
    try:
        frame = pd.read_csv(csv_path)
        if target_name not in frame.columns:
            return None
        values = pd.to_numeric(frame[target_name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return [clean_number(float(np.min(values))), clean_number(float(np.max(values)))]
    except Exception:
        return None


def is_missing(value: Any) -> bool:
    return value is None or value == ""


def record(rows: list[dict[str, Any]], dataset_rel: str, field: str, old: Any, new: Any, reason: str) -> None:
    rows.append(
        {
            "dataset_dir": dataset_rel,
            "field": field,
            "old_value": json.dumps(old, ensure_ascii=False, default=str),
            "new_value": json.dumps(new, ensure_ascii=False, default=str),
            "reason": reason,
        }
    )


def repair_one(dataset_dir: Path, data_root: Path, apply: bool, yaml: YAML) -> tuple[list[dict[str, Any]], list[dict[str, str]], bool]:
    metadata_path = dataset_dir / "metadata.yaml"
    dataset_rel = str(dataset_dir.relative_to(data_root))
    auto_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, str]] = []
    changed = False
    if not metadata_path.exists():
        manual_rows.append({"dataset_dir": dataset_rel, "field": "metadata.yaml", "reason": "missing_file", "recommendation": "人工确认数据集来源后补 metadata"})
        return auto_rows, manual_rows, changed

    with metadata_path.open("r", encoding="utf-8") as handle:
        meta = yaml.load(handle) or CommentedMap()
    dataset = meta.get("dataset")
    if not isinstance(dataset, CommentedMap):
        manual_rows.append({"dataset_dir": dataset_rel, "field": "dataset", "reason": "invalid_schema", "recommendation": "人工修复 metadata 顶层结构"})
        return auto_rows, manual_rows, changed

    splits = dataset.get("splits")
    target = dataset.get("target")
    features = dataset.get("features") or []
    if not isinstance(splits, CommentedMap) or not isinstance(target, CommentedMap):
        manual_rows.append({"dataset_dir": dataset_rel, "field": "splits/target", "reason": "invalid_schema", "recommendation": "人工修复 splits 或 target 结构"})
        return auto_rows, manual_rows, changed

    target_name = target.get("name")
    feature_names = [feat.get("name") for feat in features if isinstance(feat, dict) and feat.get("name")]

    if is_missing(dataset.get("ground_truth_formula")):
        if (dataset_dir / "formula.py").exists():
            value = CommentedMap({"file": "formula.py"})
            insert_after(dataset, "splits", "ground_truth_formula", value)
            record(auto_rows, dataset_rel, "dataset.ground_truth_formula.file", None, "formula.py", "formula.py exists")
            changed = True
        else:
            manual_rows.append({"dataset_dir": dataset_rel, "field": "dataset.ground_truth_formula", "reason": "formula.py missing", "recommendation": "不要伪造公式；若为 blackbox，显式标注 blackbox"})
    elif isinstance(dataset.get("ground_truth_formula"), CommentedMap):
        gtf = dataset.get("ground_truth_formula")
        if is_missing(gtf.get("file")) and (dataset_dir / "formula.py").exists():
            gtf["file"] = "formula.py"
            record(auto_rows, dataset_rel, "dataset.ground_truth_formula.file", None, "formula.py", "formula.py exists")
            changed = True

    id_samples = None
    if isinstance(splits.get("id_test"), dict):
        id_samples = splits["id_test"].get("samples")
    if isinstance(id_samples, (int, float)) and id_samples:
        for split_name in REQUIRED_SPLITS:
            split_info = splits.get(split_name)
            if not isinstance(split_info, CommentedMap):
                continue
            samples = split_info.get("samples")
            if is_missing(split_info.get("ratio")) and isinstance(samples, (int, float)):
                ratio = clean_number(float(samples) / float(id_samples))
                insert_split_field(split_info, "ratio", ratio)
                record(auto_rows, dataset_rel, f"splits.{split_name}.ratio", None, ratio, "samples / id_test.samples")
                changed = True

    if target_name:
        for split_name in REQUIRED_SPLITS:
            split_info = splits.get(split_name)
            if not isinstance(split_info, CommentedMap):
                continue
            split_file = split_info.get("file")
            if not split_file:
                continue
            if is_missing(split_info.get("nmse")):
                nmse, nmse_error = compute_nmse(dataset_dir, str(split_file), feature_names, str(target_name))
                if nmse is not None:
                    nmse_value = clean_number(float(nmse))
                    insert_split_field(split_info, "nmse", nmse_value)
                    record(auto_rows, dataset_rel, f"splits.{split_name}.nmse", None, nmse_value, "computed from formula.py and split CSV")
                    changed = True
                elif (dataset_dir / "formula.py").exists():
                    manual_rows.append({"dataset_dir": dataset_rel, "field": f"splits.{split_name}.nmse", "reason": nmse_error or "formula evaluation failed", "recommendation": "人工确认该 split 是否应记录 NMSE；单样本且目标方差为 0 时 NMSE 不可定义"})

        if is_missing(target.get("ood_range")):
            value = compute_target_range(dataset_dir, "ood_test.csv", str(target_name))
            if value is not None:
                target["ood_range"] = value
                record(auto_rows, dataset_rel, "target.ood_range", None, value, "computed from ood_test target min/max")
                changed = True

    if is_missing(dataset.get("citation")):
        manual_rows.append({"dataset_dir": dataset_rel, "field": "dataset.citation", "reason": "semantic attribution required", "recommendation": "按来源人工确认引用口径"})

    for feat in features:
        if isinstance(feat, dict) and is_missing(feat.get("description")):
            manual_rows.append({"dataset_dir": dataset_rel, "field": f"features.{feat.get('name', '<unknown>')}.description", "reason": "feature semantics required", "recommendation": "blackbox/真实表格特征不要自动臆造语义"})

    if apply and changed:
        with metadata_path.open("w", encoding="utf-8") as handle:
            yaml.dump(meta, handle)
    return auto_rows, manual_rows, changed


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    report_dir = Path(args.report_dir)
    yaml = yaml_loader()
    dataset_dirs = discover_dataset_dirs(data_root, args.scope_csv)
    all_auto: list[dict[str, Any]] = []
    all_manual: list[dict[str, str]] = []
    changed_count = 0
    for dataset_dir in dataset_dirs:
        auto_rows, manual_rows, changed = repair_one(dataset_dir, data_root, args.apply, yaml)
        all_auto.extend(auto_rows)
        all_manual.extend(manual_rows)
        changed_count += int(changed)

    write_csv(
        report_dir / "auto_metadata_updates.csv",
        all_auto,
        ["dataset_dir", "field", "old_value", "new_value", "reason"],
    )
    write_csv(
        report_dir / "manual_metadata_gaps.csv",
        all_manual,
        ["dataset_dir", "field", "reason", "recommendation"],
    )
    summary = {
        "apply": args.apply,
        "dataset_count": len(dataset_dirs),
        "datasets_changed": changed_count,
        "auto_update_count": len(all_auto),
        "manual_gap_count": len(all_manual),
        "auto_by_field": dict(Counter(row["field"] for row in all_auto)),
        "manual_by_field": dict(Counter(row["field"] for row in all_manual)),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"reports: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
