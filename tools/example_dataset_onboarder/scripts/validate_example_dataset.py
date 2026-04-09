from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import yaml


REQUIRED_SPLITS = ("train", "valid", "id_test", "ood_test")


def parse_args():
    parser = argparse.ArgumentParser(description="校验 examples 风格数据集目录")
    parser.add_argument("--dataset-dir", required=True, help="待校验的数据集目录")
    parser.add_argument("--require-formula", action="store_true", help="要求 ground_truth_formula.file 存在")
    parser.add_argument("--allow-empty-ood", action="store_true", help="允许 ood_test.csv 为空")
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
