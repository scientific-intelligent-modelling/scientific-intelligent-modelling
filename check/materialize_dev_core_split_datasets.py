#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_manifest(rows: list[dict[str, str]], split_name: str, split_root: Path, output_path: Path) -> None:
    fieldnames = [
        "tool",
        "machine",
        "global_index",
        "slice_index",
        "benchmark_split",
        "family",
        "subgroup",
        "dataset_name",
        "dataset_dir",
        "train_csv",
        "valid_csv",
        "id_test_csv",
        "ood_test_csv",
        "metadata_yaml",
        "has_formula_py",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            relative_dataset = Path(row["dataset_dir"]).relative_to("sim-datasets-data")
            dataset_dir = Path("sim-datasets-data") / "benchmark-splits" / split_name / relative_dataset
            writer.writerow(
                {
                    "tool": "formal-benchmark",
                    "machine": split_name,
                    "global_index": idx,
                    "slice_index": idx,
                    "benchmark_split": split_name,
                    "family": row["family"],
                    "subgroup": row["subgroup"],
                    "dataset_name": row["dataset_name"],
                    "dataset_dir": dataset_dir.as_posix(),
                    "train_csv": (dataset_dir / "train.csv").as_posix(),
                    "valid_csv": (dataset_dir / "valid.csv").as_posix(),
                    "id_test_csv": (dataset_dir / "id_test.csv").as_posix(),
                    "ood_test_csv": (dataset_dir / "ood_test.csv").as_posix(),
                    "metadata_yaml": (dataset_dir / "metadata.yaml").as_posix(),
                    "has_formula_py": row.get("has_formula_py", "0"),
                }
            )


def _materialize_split(rows: list[dict[str, str]], split_name: str, dataset_root: Path) -> dict[str, object]:
    split_root = dataset_root / "benchmark-splits" / split_name
    _safe_mkdir(split_root)

    linked = 0
    checked = 0
    leaves: list[str] = []
    for row in rows:
        src = Path(row["dataset_dir"]).resolve()
        if not src.exists():
            raise FileNotFoundError(f"原始数据集不存在: {src}")
        relative_dataset = Path(row["dataset_dir"]).relative_to("sim-datasets-data")
        dst = split_root / relative_dataset
        _safe_mkdir(dst.parent)
        relative_target = Path(
            Path(
                Path(*([".."] * len(dst.parent.relative_to(dataset_root).parts)))
            )
            / relative_dataset
        )

        if dst.is_symlink():
            if dst.resolve() != src:
                raise RuntimeError(f"已存在但指向不一致的 symlink: {dst} -> {dst.resolve()} (期望 {src})")
        elif dst.exists():
            raise RuntimeError(f"目标路径已存在且不是 symlink，停止覆盖: {dst}")
        else:
            dst.symlink_to(relative_target)
            linked += 1
        checked += 1
        leaves.append(dst.as_posix())

    manifest_path = split_root / f"{split_name}_manifest.csv"
    _write_manifest(rows, split_name, split_root, manifest_path)
    return {
        "split": split_name,
        "split_root": split_root.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "dataset_count": checked,
        "new_links_created": linked,
        "sample_paths": leaves[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="把 Dev-50 / Core-50 物化到 sim-datasets-data/benchmark-splits 下。")
    parser.add_argument(
        "--dev-slice",
        default="experiment-results/benchmark_formal200_20260417/dev_core_split_v1/benchmark_dev50_formal_slice.csv",
        help="Dev-50 正式切片 CSV",
    )
    parser.add_argument(
        "--core-slice",
        default="experiment-results/benchmark_formal200_20260417/dev_core_split_v1/benchmark_core50_formal_slice.csv",
        help="Core-50 正式切片 CSV",
    )
    parser.add_argument(
        "--dataset-root",
        default="sim-datasets-data",
        help="本地 sim-datasets-data 根目录",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 不存在: {dataset_root}")

    dev_rows = _load_rows(Path(args.dev_slice).resolve())
    core_rows = _load_rows(Path(args.core_slice).resolve())

    dev_info = _materialize_split(dev_rows, "dev50", dataset_root)
    core_info = _materialize_split(core_rows, "core50", dataset_root)

    print(json.dumps({"dev50": dev_info, "core50": core_info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
