from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从单个表格中基于特征区间逐维内缩抽取 OOD 样本。"
            "成功时输出 id_pool.csv 和 ood_test.csv；失败时只输出 summary，不生成 ood_test.csv。"
        )
    )
    parser.add_argument("--input-csv", required=True, help="输入总表 CSV 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--target-column",
        default=None,
        help="目标列名；默认使用最后一列",
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="逗号分隔的特征列；默认使用除目标列外的所有列",
    )
    parser.add_argument(
        "--step-percent",
        type=float,
        default=0.0001,
        help="每次内缩比例，按区间长度百分比计；0.0001 表示 0.01%%",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.10,
        help="目标 OOD 占比，默认 10%%",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.20,
        help="允许的最大 OOD 占比，默认 20%%；超过即判失败",
    )
    parser.add_argument(
        "--id-output-name",
        default="id_pool.csv",
        help="非 OOD 样本输出文件名，默认 id_pool.csv",
    )
    parser.add_argument(
        "--ood-output-name",
        default="ood_test.csv",
        help="OOD 样本输出文件名，默认 ood_test.csv",
    )
    parser.add_argument(
        "--summary-name",
        default="ood_split_summary.json",
        help="摘要文件名，默认 ood_split_summary.json",
    )
    return parser.parse_args()


def parse_float(value: str) -> float:
    return float(str(value).strip())


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError(f"{path} 缺少表头")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path} 没有数据行")
    return header, rows


def resolve_columns(
    header: list[str],
    target_column: str | None,
    feature_columns: str | None,
) -> tuple[str, list[str]]:
    target = target_column.strip() if isinstance(target_column, str) and target_column.strip() else header[-1]
    if target not in header:
        raise ValueError(f"目标列不存在: {target}")

    if feature_columns:
        features = [item.strip() for item in feature_columns.split(",") if item.strip()]
    else:
        features = [col for col in header if col != target]
    if not features:
        raise ValueError("至少需要一个特征列")
    missing = [col for col in features if col not in header]
    if missing:
        raise ValueError(f"特征列不存在: {missing}")
    return target, features


def compute_feature_ranges(rows: list[dict[str, str]], feature_names: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for name in feature_names:
        values = [parse_float(row[name]) for row in rows]
        vmin = min(values)
        vmax = max(values)
        ranges[name] = (vmin, vmax)
    return ranges


def row_is_ood(
    row: dict[str, str],
    feature_names: list[str],
    bounds: dict[str, tuple[float, float]],
) -> bool:
    for name in feature_names:
        value = parse_float(row[name])
        lower, upper = bounds[name]
        if value < lower or value > upper:
            return True
    return False


def split_rows(
    rows: list[dict[str, str]],
    feature_names: list[str],
    bounds: dict[str, tuple[float, float]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ood_rows: list[dict[str, str]] = []
    id_rows: list[dict[str, str]] = []
    for row in rows:
        if row_is_ood(row, feature_names, bounds):
            ood_rows.append(row)
        else:
            id_rows.append(row)
    return id_rows, ood_rows


def infer_discrete_like(
    rows: list[dict[str, str]],
    feature_names: list[str],
) -> dict[str, bool]:
    total = len(rows)
    threshold = min(20, max(3, int(total * 0.05)))
    result: dict[str, bool] = {}
    for name in feature_names:
        values = [row[name] for row in rows]
        unique_values = set(values)
        result[name] = len(unique_values) <= threshold
    return result


def build_bounds(
    ranges: dict[str, tuple[float, float]],
    shrink_steps: dict[str, int],
    step_percent: float,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for name, (vmin, vmax) in ranges.items():
        length = vmax - vmin
        shrink = length * step_percent * shrink_steps[name]
        bounds[name] = (vmin + shrink, vmax - shrink)
    return bounds


def dimension_can_shrink(
    name: str,
    ranges: dict[str, tuple[float, float]],
    shrink_steps: dict[str, int],
    step_percent: float,
) -> bool:
    vmin, vmax = ranges[name]
    length = vmax - vmin
    if length <= 0:
        return False
    next_shrink = length * step_percent * (shrink_steps[name] + 1)
    return (vmin + next_shrink) < (vmax - next_shrink)


def score_rows_for_fallback(
    rows: list[dict[str, str]],
    feature_names: list[str],
    ranges: dict[str, tuple[float, float]],
    discrete_like: dict[str, bool],
) -> list[tuple[float, int]]:
    total = len(rows)
    discrete_freqs: dict[str, dict[str, int]] = {}
    for name in feature_names:
        if discrete_like[name]:
            freq: dict[str, int] = {}
            for row in rows:
                value = row[name]
                freq[value] = freq.get(value, 0) + 1
            discrete_freqs[name] = freq

    scored: list[tuple[float, int]] = []
    for idx, row in enumerate(rows):
        parts: list[float] = []
        for name in feature_names:
            value = parse_float(row[name])
            vmin, vmax = ranges[name]
            if discrete_like[name]:
                count = discrete_freqs[name][row[name]]
                parts.append(1.0 - (count / total))
                continue

            half = (vmax - vmin) / 2.0
            if half <= 0:
                parts.append(0.0)
                continue
            mid = (vmin + vmax) / 2.0
            score = abs((value - mid) / half)
            parts.append(max(0.0, score))

        scored.append((sum(parts) / len(parts) if parts else 0.0, idx))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored


def fallback_extract_ood_rows(
    rows: list[dict[str, str]],
    feature_names: list[str],
    ranges: dict[str, tuple[float, float]],
    *,
    target_ratio: float,
    max_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    total = len(rows)
    target_count = max(1, round(total * target_ratio))
    if target_count / total > max_ratio:
        return rows, [], {
            "status": "failed",
            "reason": "target_count_exceeds_max_ratio",
            "method": "fallback_boundary_rarity_ranking",
            "ood_ratio": target_count / total,
        }

    discrete_like = infer_discrete_like(rows, feature_names)
    scored = score_rows_for_fallback(rows, feature_names, ranges, discrete_like)
    selected = {idx for _, idx in scored[:target_count]}

    ood_rows: list[dict[str, str]] = []
    id_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        if idx in selected:
            ood_rows.append(row)
        else:
            id_rows.append(row)

    return id_rows, ood_rows, {
        "status": "success",
        "reason": "fallback_boundary_rarity_ranking",
        "method": "fallback_boundary_rarity_ranking",
        "ood_ratio": len(ood_rows) / total,
        "target_count": target_count,
        "discrete_like_features": [name for name, flag in discrete_like.items() if flag],
    }


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(row_list)
    return len(row_list)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    header, rows = load_rows(input_path)
    target_col, feature_names = resolve_columns(header, args.target_column, args.feature_columns)
    ranges = compute_feature_ranges(rows, feature_names)
    shrink_steps = {name: 0 for name in feature_names}
    bounds = build_bounds(ranges, shrink_steps, args.step_percent)

    summary: dict[str, object] = {
        "status": "failed",
        "input_csv": str(input_path),
        "target_column": target_col,
        "feature_columns": feature_names,
        "step_percent": args.step_percent,
        "target_ratio": args.target_ratio,
        "max_ratio": args.max_ratio,
        "total_rows": len(rows),
        "ood_ratio": 0.0,
        "method": "iterative_shrink",
        "shrink_steps": shrink_steps.copy(),
        "bounds": bounds,
        "reason": "",
    }

    dim_index = 0
    tried = 0
    max_iterations = max(1, len(feature_names) * 100000)

    success = False
    final_id_rows = rows
    final_ood_rows: list[dict[str, str]] = []

    while tried < max_iterations:
        tried += 1
        name = feature_names[dim_index]
        dim_index = (dim_index + 1) % len(feature_names)

        if not dimension_can_shrink(name, ranges, shrink_steps, args.step_percent):
            if not any(
                dimension_can_shrink(candidate, ranges, shrink_steps, args.step_percent)
                for candidate in feature_names
            ):
                summary["reason"] = "all_dimensions_exhausted_before_reaching_target_ratio"
                break
            continue

        shrink_steps[name] += 1
        bounds = build_bounds(ranges, shrink_steps, args.step_percent)
        id_rows, ood_rows = split_rows(rows, feature_names, bounds)
        ood_ratio = len(ood_rows) / len(rows)

        summary["ood_ratio"] = ood_ratio
        summary["shrink_steps"] = shrink_steps.copy()
        summary["bounds"] = bounds

        if ood_ratio > args.max_ratio:
            summary["reason"] = "ood_ratio_exceeded_max_ratio"
            break

        if ood_ratio >= args.target_ratio:
            success = True
            final_id_rows = id_rows
            final_ood_rows = ood_rows
            summary["status"] = "success"
            summary["reason"] = "target_ratio_reached"
            break

    if not success:
        fallback_id_rows, fallback_ood_rows, fallback_info = fallback_extract_ood_rows(
            rows,
            feature_names,
            ranges,
            target_ratio=args.target_ratio,
            max_ratio=args.max_ratio,
        )
        if fallback_info["status"] == "success":
            success = True
            final_id_rows = fallback_id_rows
            final_ood_rows = fallback_ood_rows
            summary["status"] = "success"
            summary["reason"] = str(fallback_info["reason"])
            summary["method"] = str(fallback_info["method"])
            summary["ood_ratio"] = float(fallback_info["ood_ratio"])
            summary["fallback"] = fallback_info
        else:
            summary["fallback"] = fallback_info

    if success:
        id_count = write_csv(output_dir / args.id_output_name, header, final_id_rows)
        ood_count = write_csv(output_dir / args.ood_output_name, header, final_ood_rows)
        summary["id_rows"] = id_count
        summary["ood_rows"] = ood_count
    else:
        summary["id_rows"] = len(rows)
        summary["ood_rows"] = 0

    (output_dir / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if success:
        print(
            f"成功生成 OOD：total={len(rows)}, id={len(final_id_rows)}, "
            f"ood={len(final_ood_rows)}, ratio={len(final_ood_rows)/len(rows):.4f}"
        )
    else:
        print(f"OOD 生成失败：{summary['reason']}")


if __name__ == "__main__":
    main()
