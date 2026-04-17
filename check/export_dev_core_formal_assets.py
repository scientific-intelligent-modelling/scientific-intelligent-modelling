#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


SLICE_FIELDS = [
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


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_slice_csv(rows: list[dict[str, str]], benchmark_split: str, repo_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SLICE_FIELDS)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            dataset_dir = row["dataset_dir"]
            formula_path = repo_root / dataset_dir / "formula.py"
            writer.writerow(
                {
                    "tool": "formal-benchmark",
                    "machine": benchmark_split,
                    "global_index": idx,
                    "slice_index": idx,
                    "benchmark_split": benchmark_split,
                    "family": row["family"],
                    "subgroup": row["subgroup"],
                    "dataset_name": row["dataset_name"],
                    "dataset_dir": dataset_dir,
                    "train_csv": f"{dataset_dir}/train.csv",
                    "valid_csv": f"{dataset_dir}/valid.csv",
                    "id_test_csv": f"{dataset_dir}/id_test.csv",
                    "ood_test_csv": f"{dataset_dir}/ood_test.csv",
                    "metadata_yaml": f"{dataset_dir}/metadata.yaml",
                    "has_formula_py": "1" if formula_path.exists() else "0",
                }
            )


def _float(v: Any) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _metric_summary(rows: list[dict[str, str]], key: str) -> tuple[float | None, float | None]:
    vals = [_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return sum(vals) / len(vals), median(vals)


def _render_counter_table(title: str, left: Counter, right: Counter) -> list[str]:
    keys = sorted(set(left) | set(right))
    lines = [f"### {title}", "", "| 项目 | Dev-50 | Core-50 |", "|---|---:|---:|"]
    for key in keys:
        lines.append(f"| `{key}` | {left.get(key, 0)} | {right.get(key, 0)} |")
    lines.append("")
    return lines


def _write_paper_summary(master: list[dict[str, str]], dev: list[dict[str, str]], core: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_family = Counter(r["family"] for r in master)
    dev_family = Counter(r["family"] for r in dev)
    core_family = Counter(r["family"] for r in core)
    dev_mode = Counter(r["selection_mode"] for r in dev)
    core_mode = Counter(r["selection_mode"] for r in core)
    dev_adv = Counter(r["candidate_advantage_side"] for r in dev)
    core_adv = Counter(r["candidate_advantage_side"] for r in core)

    lines = [
        "# Dev-50 / Core-50 切分说明（论文版简表）",
        "",
        "## 设计原则",
        "",
        "- 先基于三 seed 正式结果从 200 个候选中筛出 `Master-100`。",
        "- 再只用非结果信息切分成 `Dev-50` 与 `Core-50`，避免直接按正式成绩量身定制测试集。",
        "- 强约束：`Master-100` 内 `basename <= 1`，避免同 basename 变体跨集合泄漏。",
        "- `Core-50` 中 `one-sided` 样本数限制为 `<= 5`。",
        "",
        "## 集合规模",
        "",
        f"- `Master-100 = {len(master)}`",
        f"- `Dev-50 = {len(dev)}`",
        f"- `Core-50 = {len(core)}`",
        "",
        "## Master-100 顶层 family 分布",
        "",
        "| family | 数量 |",
        "|---|---:|",
    ]
    for family, count in sorted(master_family.items()):
        lines.append(f"| `{family}` | {count} |")
    lines.append("")

    lines.extend(_render_counter_table("Dev/Core 顶层 family 分布", dev_family, core_family))
    lines.extend(_render_counter_table("Dev/Core selection_mode 分布", dev_mode, core_mode))
    lines.extend(_render_counter_table("Dev/Core 候选阶段优势标签分布", dev_adv, core_adv))

    lines.extend(
        [
            "## 静态属性匹配摘要",
            "",
            "| 指标 | Dev-50 mean | Dev-50 median | Core-50 mean | Core-50 median |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ["feature_count", "train_samples", "valid_samples", "id_test_samples", "ood_test_samples", "formula_operator_count"]:
        d_mean, d_median = _metric_summary(dev, key)
        c_mean, c_median = _metric_summary(core, key)
        lines.append(
            f"| `{key}` | {d_mean if d_mean is not None else 'NA'} | {d_median if d_median is not None else 'NA'} | "
            f"{c_mean if c_mean is not None else 'NA'} | {c_median if c_median is not None else 'NA'} |"
        )

    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "- `Dev-50`：用于方法迭代、进化和超参选择。",
            "- `Core-50`：冻结，不参与任何调参，只用于最终汇报。",
            "- 两个集合分布尽量一致，但 `Core-50` 不再根据正式结果做二次微调。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Dev/Core 正式切片 CSV，并生成论文版简表。")
    parser.add_argument(
        "--split-dir",
        default="experiment-results/benchmark_formal200_20260417/dev_core_split_v1",
        help="Master/Dev/Core 切分目录",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="仓库根目录",
    )
    parser.add_argument(
        "--paper-summary",
        default="paper/benchmark/dev_core_split_summary.md",
        help="论文版切分说明表输出路径",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    master = _load_rows(split_dir / "master100_candidates.csv")
    dev = _load_rows(split_dir / "benchmark_dev50.csv")
    core = _load_rows(split_dir / "benchmark_core50.csv")

    dev_slice = split_dir / "benchmark_dev50_formal_slice.csv"
    core_slice = split_dir / "benchmark_core50_formal_slice.csv"
    _write_slice_csv(dev, "dev50", repo_root, dev_slice)
    _write_slice_csv(core, "core50", repo_root, core_slice)

    _write_paper_summary(master, dev, core, Path(args.paper_summary).resolve())

    print("已生成：")
    print(dev_slice)
    print(core_slice)
    print(Path(args.paper_summary).resolve())


if __name__ == "__main__":
    main()
