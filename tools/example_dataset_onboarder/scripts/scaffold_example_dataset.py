from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


REQUIRED_SPLITS = ("train", "valid", "id_test", "ood_test")


def parse_args():
    parser = argparse.ArgumentParser(description="生成 examples 风格数据集目录脚手架")
    parser.add_argument("--dataset-dir", required=True, help="输出数据集目录")
    parser.add_argument("--dataset-name", required=True, help="metadata.yaml 中的 dataset.name")
    parser.add_argument("--target", required=True, help="目标列名")
    parser.add_argument("--features", required=True, help="逗号分隔的特征列名，例如 x0,x1,x2")
    parser.add_argument("--description", default="", help="数据集描述")
    parser.add_argument("--with-formula", action="store_true", help="同时生成 formula.py 模板")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在文件")
    return parser.parse_args()


def write_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"文件已存在，请使用 --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv_with_header(path: Path, header: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"文件已存在，请使用 --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    feature_names = [item.strip() for item in args.features.split(",") if item.strip()]
    if not feature_names:
        raise SystemExit("至少需要一个特征列")

    if args.target in feature_names:
        raise SystemExit("目标列不能同时出现在 features 中")

    header = feature_names + [args.target]

    metadata = {
        "dataset": {
            "name": args.dataset_name,
            "description": args.description,
            "splits": {
                split: {
                    "file": f"{split}.csv",
                    "samples": 0,
                    "notes": "Fill after writing actual rows",
                }
                for split in REQUIRED_SPLITS
            },
            "features": [
                {
                    "name": name,
                    "type": "continuous",
                    "description": "",
                    "train_range": [None, None],
                    "ood_range": [None, None],
                }
                for name in feature_names
            ],
            "target": {
                "name": args.target,
                "type": "continuous",
                "description": "",
                "train_range": [None, None],
                "ood_range": [None, None],
            },
            "resources": [],
        }
    }

    if args.with_formula:
        metadata["dataset"]["ground_truth_formula"] = {"file": "formula.py"}

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in REQUIRED_SPLITS:
        write_csv_with_header(dataset_dir / f"{split}.csv", header, args.overwrite)

    metadata_path = dataset_dir / "metadata.yaml"
    write_text(metadata_path, yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), args.overwrite)

    if args.with_formula:
        formula_content = (
            "import numpy as np\n\n"
            f"def {args.target}({', '.join(feature_names)}):\n"
            f"    raise NotImplementedError('Replace with ground-truth formula if known')\n"
        )
        write_text(dataset_dir / "formula.py", formula_content, args.overwrite)

    print("脚手架生成完成。")
    print(f"dataset_dir: {dataset_dir}")
    print(f"features: {feature_names}")
    print(f"target: {args.target}")


if __name__ == "__main__":
    main()
