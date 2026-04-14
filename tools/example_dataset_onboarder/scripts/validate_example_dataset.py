from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


REQUIRED_SPLITS = ("train", "valid", "id_test")
OPTIONAL_SPLIT = "ood_test"


def parse_args():
    parser = argparse.ArgumentParser(description="校验 examples 风格数据集目录")
    parser.add_argument("--dataset-dir", required=True, help="待校验的数据集目录")
    parser.add_argument("--require-formula", action="store_true", help="要求 ground_truth_formula.file 存在")
    parser.add_argument("--require-ood", action="store_true", help="要求 ood_test.csv 存在")
    parser.add_argument("--allow-empty-ood", action="store_true", help="允许 ood_test.csv 为空")
    parser.add_argument("--verify-formula", action="store_true", help="若存在 ground truth formula，则代入特征并验证目标列")
    parser.add_argument("--formula-nmse-threshold", type=float, default=1e-6, help="公式校验允许的最大 NMSE")
    parser.add_argument("--formula-max-abs-threshold", type=float, default=1e-6, help="公式校验允许的最大绝对误差")
    return parser.parse_args()


def read_csv_header_and_count(path: Path) -> Tuple[List[str], int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} 是空文件，至少需要表头")
        row_count = sum(1 for _ in reader)
    if not header:
        raise ValueError(f"{path} 缺少表头")
    return header, row_count


def load_split_arrays(path: Path, feature_names: List[str], target_name: str) -> Tuple[List[np.ndarray], np.ndarray]:
    feature_columns = {name: [] for name in feature_names}
    target_values = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} 缺少表头")
        for row in reader:
            for name in feature_names:
                if name not in row:
                    raise ValueError(f"{path} 缺少特征列 {name}")
                feature_columns[name].append(float(row[name]))
            if target_name not in row:
                raise ValueError(f"{path} 缺少目标列 {target_name}")
            target_values.append(float(row[target_name]))
    feature_arrays = [np.asarray(feature_columns[name], dtype=float) for name in feature_names]
    target_array = np.asarray(target_values, dtype=float)
    return feature_arrays, target_array


def load_formula_callable(dataset_dir: Path, formula_file: str, target_name: str | None):
    formula_path = dataset_dir / formula_file
    if not formula_path.exists():
        raise FileNotFoundError(f"公式文件不存在: {formula_path.name}")

    module_name = f"_example_formula_{dataset_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, formula_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为公式文件创建导入 spec: {formula_path}")

    module = importlib.util.module_from_spec(spec)
    module.__dict__.setdefault("np", np)
    module.__dict__.setdefault("numpy", np)
    module.__dict__.setdefault("math", math)
    spec.loader.exec_module(module)

    candidate_names = []
    if isinstance(target_name, str) and target_name.strip():
        candidate_names.append(target_name.strip())
    candidate_names.extend(["equation", "formula", "ground_truth"])
    for name in candidate_names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj, name

    for name, obj in module.__dict__.items():
        if name.startswith("_"):
            continue
        if callable(obj) and getattr(obj, "__module__", None) == module.__name__:
            return obj, name

    raise ValueError(f"{formula_path.name} 中未找到可调用公式函数")


def verify_formula_against_data(
    dataset_dir: Path,
    split_meta: dict,
    feature_names: List[str],
    target_name: str,
    formula_file: str,
    nmse_threshold: float,
    max_abs_threshold: float,
    allow_empty_ood: bool,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    reports: List[str] = []

    formula_fn, formula_name = load_formula_callable(dataset_dir, formula_file, target_name)
    checked_any = False

    for split_name in REQUIRED_SPLITS + (OPTIONAL_SPLIT,):
        split_info = split_meta.get(split_name) or {}
        csv_name = split_info.get("file") or f"{split_name}.csv"
        csv_path = dataset_dir / csv_name
        if not csv_path.exists():
            continue

        feature_arrays, target_array = load_split_arrays(csv_path, feature_names, target_name)
        if target_array.size == 0:
            if split_name == "ood_test" and allow_empty_ood:
                continue
            continue

        checked_any = True
        pred = formula_fn(*feature_arrays)
        pred_array = np.asarray(pred, dtype=float)
        if pred_array.ndim == 0:
            pred_array = np.full_like(target_array, float(pred_array), dtype=float)
        pred_array = pred_array.reshape(-1)

        if pred_array.shape != target_array.shape:
            errors.append(
                f"公式 {formula_name} 在 split={split_name} 的输出形状不匹配: {pred_array.shape} != {target_array.shape}"
            )
            continue

        if not np.all(np.isfinite(pred_array)):
            errors.append(f"公式 {formula_name} 在 split={split_name} 产生了非有限值")
            continue

        residual = pred_array - target_array
        mse = float(np.mean(np.square(residual)))
        rmse = math.sqrt(mse)
        denom = float(np.mean(np.square(target_array))) if target_array.size else float("nan")
        nmse = float(mse / denom) if denom > 0 and math.isfinite(denom) else mse
        max_abs_err = float(np.max(np.abs(residual))) if target_array.size else 0.0
        reports.append(
            f"公式校验 split={split_name}: rmse={rmse:.6g}, nmse={nmse:.6g}, max_abs_err={max_abs_err:.6g}"
        )
        if nmse > nmse_threshold and max_abs_err > max_abs_threshold:
            errors.append(
                f"公式 {formula_name} 在 split={split_name} 与数据不一致: "
                f"nmse={nmse:.6g} > {nmse_threshold:.6g} 且 max_abs_err={max_abs_err:.6g} > {max_abs_threshold:.6g}"
            )

    if not checked_any:
        errors.append("启用了公式校验，但没有可用于代入的非空 split")

    return errors, reports


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    errors: List[str] = []
    warnings: List[str] = []

    if not dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset_dir}")

    metadata_path = dataset_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise SystemExit(f"缺少 metadata.yaml: {metadata_path}")

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    dataset_meta = metadata.get("dataset")
    if not isinstance(dataset_meta, dict):
        errors.append("metadata.yaml 缺少顶层 dataset 字段")
        dataset_meta = {}

    target_meta = dataset_meta.get("target") or {}
    target_name = target_meta.get("name")
    if not isinstance(target_name, str) or not target_name.strip():
        errors.append("metadata.yaml 缺少 dataset.target.name")
        target_name = None

    feature_meta = dataset_meta.get("features") or []
    feature_names = []
    if not isinstance(feature_meta, list) or not feature_meta:
        errors.append("metadata.yaml 缺少非空 dataset.features")
    else:
        for index, item in enumerate(feature_meta):
            if not isinstance(item, dict):
                errors.append(f"dataset.features[{index}] 不是对象")
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"dataset.features[{index}] 缺少 name")
                continue
            feature_names.append(name.strip())

    split_meta = dataset_meta.get("splits") or {}
    if not isinstance(split_meta, dict):
        errors.append("metadata.yaml 缺少 dataset.splits")
        split_meta = {}

    reference_header = None
    reference_columns = None

    for split_name in REQUIRED_SPLITS:
        split_info = split_meta.get(split_name) or {}
        csv_name = split_info.get("file") or f"{split_name}.csv"
        csv_path = dataset_dir / csv_name

        if not csv_path.exists():
            errors.append(f"缺少 split 文件: {csv_path.name}")
            continue

        try:
            header, row_count = read_csv_header_and_count(csv_path)
        except Exception as exc:
            errors.append(str(exc))
            continue

        if reference_header is None:
            reference_header = header
            reference_columns = set(header)
        elif header != reference_header:
            errors.append(
                f"{csv_path.name} 的列头与 train/首个 split 不一致: {header} != {reference_header}"
            )

        if row_count == 0:
            if split_name == "ood_test" and args.allow_empty_ood:
                warnings.append("ood_test.csv 为空，已按 --allow-empty-ood 放行")
            else:
                errors.append(f"{csv_path.name} 没有数据行")

        declared_samples = split_info.get("samples")
        if declared_samples is not None:
            try:
                declared_samples = int(declared_samples)
            except Exception:
                errors.append(f"metadata 中 {split_name}.samples 不是整数")
            else:
                if declared_samples != row_count:
                    errors.append(
                        f"metadata 中 {split_name}.samples={declared_samples}，但 {csv_path.name} 实际为 {row_count}"
                    )

    ood_info = split_meta.get(OPTIONAL_SPLIT) or {}
    ood_csv_name = ood_info.get("file") or f"{OPTIONAL_SPLIT}.csv"
    ood_csv_path = dataset_dir / ood_csv_name
    has_ood_meta = bool(ood_info)
    has_ood_file = ood_csv_path.exists()

    if args.require_ood and not has_ood_file:
        errors.append(f"缺少 split 文件: {ood_csv_path.name}")
    elif has_ood_meta and not has_ood_file:
        errors.append(f"metadata 声明了 {ood_csv_path.name}，但文件不存在")
    elif has_ood_file:
        try:
            header, row_count = read_csv_header_and_count(ood_csv_path)
        except Exception as exc:
            errors.append(str(exc))
        else:
            if reference_header is None:
                reference_header = header
                reference_columns = set(header)
            elif header != reference_header:
                errors.append(
                    f"{ood_csv_path.name} 的列头与 train/首个 split 不一致: {header} != {reference_header}"
                )
            if row_count == 0:
                if args.allow_empty_ood:
                    warnings.append("ood_test.csv 为空，已按 --allow-empty-ood 放行")
                else:
                    errors.append(f"{ood_csv_path.name} 没有数据行")
            declared_samples = ood_info.get("samples")
            if declared_samples is not None:
                try:
                    declared_samples = int(declared_samples)
                except Exception:
                    errors.append("metadata 中 ood_test.samples 不是整数")
                else:
                    if declared_samples != row_count:
                        errors.append(
                            f"metadata 中 ood_test.samples={declared_samples}，但 {ood_csv_path.name} 实际为 {row_count}"
                        )
    else:
        warnings.append("ood_test.csv 缺失；已按可选 split 处理")

    if reference_columns is not None and target_name is not None and target_name not in reference_columns:
        errors.append(f"目标列 {target_name} 不存在于 CSV 列头中")

    if reference_header is not None and feature_names:
        csv_feature_names = [col for col in reference_header if col != target_name]
        if csv_feature_names != feature_names:
            errors.append(
                f"metadata.features 与 CSV 特征列顺序不一致: metadata={feature_names}, csv={csv_feature_names}"
            )

    formula_info = dataset_meta.get("ground_truth_formula") or {}
    formula_file = formula_info.get("file")
    if args.require_formula:
        if not isinstance(formula_file, str) or not formula_file.strip():
            errors.append("要求 formula.py，但 metadata 未声明 dataset.ground_truth_formula.file")
        else:
            formula_path = dataset_dir / formula_file
            if not formula_path.exists():
                errors.append(f"要求 formula.py，但文件不存在: {formula_path.name}")

    if args.verify_formula:
        if not isinstance(formula_file, str) or not formula_file.strip():
            errors.append("启用了公式校验，但 metadata 未声明 dataset.ground_truth_formula.file")
        elif not feature_names or target_name is None:
            errors.append("启用了公式校验，但 features/target 信息不完整")
        else:
            try:
                formula_errors, formula_reports = verify_formula_against_data(
                    dataset_dir=dataset_dir,
                    split_meta=split_meta,
                    feature_names=feature_names,
                    target_name=target_name,
                    formula_file=formula_file,
                    nmse_threshold=args.formula_nmse_threshold,
                    max_abs_threshold=args.formula_max_abs_threshold,
                    allow_empty_ood=args.allow_empty_ood,
                )
                errors.extend(formula_errors)
                warnings.extend(formula_reports)
            except Exception as exc:
                errors.append(f"公式校验失败: {exc}")

    if errors:
        print("校验失败：")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)

    print("校验通过。")
    print(f"dataset_dir: {dataset_dir}")
    print(f"target: {target_name}")
    print(f"features: {feature_names}")
    if warnings:
        print("警告：")
        for item in warnings:
            print(f"- {item}")


if __name__ == "__main__":
    main()
