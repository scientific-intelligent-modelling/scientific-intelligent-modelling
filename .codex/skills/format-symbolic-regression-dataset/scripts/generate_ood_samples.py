#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 feature_distributions.csv 中的 OOD 区间，
为每个数据集在特征的 OOD 区间内进行均匀采样，
并使用对应目录下的 formula.py 计算目标值，生成新的 OOD 数据集。

当前版本保留旧 benchmark 兼容逻辑，但去掉了对某个固定目录家族的硬编码：
- 默认处理 summary 中的全部数据集
- 支持按 dataset prefix 过滤
- 支持 target 列名为 'target' / 'y' / 最后一列
- 动态匹配 formula.py 中的函数参数
- 自动重试生成 NaN/Inf 的样本，确保输出数据有效
"""

import argparse

import csv
import importlib.util
import inspect
import os
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


DEFAULT_ROOT_DIR = "."
SAMPLES_PER_DATASET = 10000
MAX_RETRIES_PER_SAMPLE = 100 # 单个样本生成的最大重试次数
MAX_TOTAL_ATTEMPTS_MULTIPLIER = 50 # 数据集总尝试次数 = 样本数 * 倍数


class FeatureMeta:
    """单个特征在一个数据集中的统计与 OOD 信息"""

    def __init__(
        self,
        name: str,
        vmin: Optional[float],
        vmax: Optional[float],
        left_min: Optional[float],
        left_max: Optional[float],
        right_min: Optional[float],
        right_max: Optional[float],
        dist_type: str,
    ) -> None:
        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.left_min = left_min
        self.left_max = left_max
        self.right_min = right_min
        self.right_max = right_max
        self.dist_type = dist_type


def parse_float(value: str) -> Optional[float]:
    """安全解析浮点数，空字符串返回 None。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OOD samples from feature_distributions.csv and formula.py.",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT_DIR,
        help="数据集家族根目录；内部应包含 <group>/<dataset>/train.csv。",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="feature_distributions.csv 路径；默认使用 <root>/feature_distributions.csv。",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=SAMPLES_PER_DATASET,
        help="每个数据集生成的 OOD 样本数。",
    )
    parser.add_argument(
        "--output-file",
        default="ood_test.csv",
        help="输出文件名，默认 ood_test.csv。",
    )
    parser.add_argument(
        "--dataset-prefix",
        action="append",
        default=[],
        help="仅处理指定前缀的数据集键，如 feynman/ 或 firstprinciples/；可重复传参。",
    )
    return parser.parse_args()


def load_feature_meta(summary_csv: str) -> Dict[str, Dict[str, FeatureMeta]]:
    """
    从 feature_distributions.csv 读取每个数据集、每个特征的 OOD 信息。
    返回：dataset_name -> { feature_name -> FeatureMeta }
    """
    dataset_map: Dict[str, Dict[str, FeatureMeta]] = {}

    if not os.path.exists(summary_csv):
        print(f"Error: {summary_csv} not found.")
        return {}

    with open(summary_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            dataset = row[0]
            cols = row[1:]
            feature_map: Dict[str, FeatureMeta] = {}

            for i in range(0, len(cols), 8):
                if i + 7 >= len(cols):
                    break
                name = cols[i]
                if not name:
                    continue
                dist_type = cols[i + 1]
                vmin = parse_float(cols[i + 2])
                vmax = parse_float(cols[i + 3])
                left_min = parse_float(cols[i + 4])
                left_max = parse_float(cols[i + 5])
                right_min = parse_float(cols[i + 6])
                right_max = parse_float(cols[i + 7])

                feature_map[name] = FeatureMeta(
                    name=name,
                    vmin=vmin,
                    vmax=vmax,
                    left_min=left_min,
                    left_max=left_max,
                    right_min=right_min,
                    right_max=right_max,
                    dist_type=dist_type,
                )

            dataset_map[dataset] = feature_map

    return dataset_map


def split_dataset_key(dataset_name: str) -> Tuple[str, str]:
    top, sub = dataset_name.split("/", 1)
    return top, sub


def load_train_header(root_dir: str, dataset_name: str) -> List[str]:
    """读取某个数据集的 train.csv 表头，保持原列顺序。"""
    top, sub = split_dataset_key(dataset_name)
    train_path = os.path.join(root_dir, top, sub, "train.csv")
    with open(train_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header


def load_formula_func(root_dir: str, dataset_name: str):
    """
    加载某个数据集目录下的 formula.py 中的主函数。
    返回函数对象。
    """
    top, sub = split_dataset_key(dataset_name)
    formula_path = os.path.join(root_dir, top, sub, "formula.py")
    
    spec = importlib.util.spec_from_file_location("formula_module", formula_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载公式模块: {formula_path}")
    
    module = importlib.util.module_from_spec(spec)
    module.np = np  # type: ignore[attr-defined]
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    # 查找该模块中定义的所有函数（排除导入的）
    functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__:
            functions.append(obj)
    
    if not functions:
        if hasattr(module, "y"):
            return getattr(module, "y")
        raise RuntimeError(f"公式模块中未找到定义的函数: {formula_path}")
    
    if len(functions) == 1:
        return functions[0]
    
    for f in functions:
        if f.__name__ == 'y':
            return f
            
    return functions[0]


def choose_interval(
    left_min: Optional[float],
    left_max: Optional[float],
    right_min: Optional[float],
    right_max: Optional[float],
) -> Optional[Tuple[float, float]]:
    """
    根据左右 OOD 区间选择一个区间，用区间长度作为权重。
    """
    intervals: List[Tuple[float, float]] = []
    if left_min is not None and left_max is not None and left_max > left_min:
        intervals.append((left_min, left_max))
    if right_min is not None and right_max is not None and right_max > right_min:
        intervals.append((right_min, right_max))

    if not intervals:
        return None

    if len(intervals) == 1:
        return intervals[0]

    lengths = [b - a for a, b in intervals]
    total_len = sum(lengths)
    if total_len <= 0:
        return intervals[0]

    r = random.random() * total_len
    acc = 0.0
    for (a, b), length in zip(intervals, lengths):
        acc += length
        if r <= acc:
            return (a, b)

    return intervals[-1]


def sample_feature_value(meta: FeatureMeta, discrete_values: Optional[List[float]] = None) -> float:
    """
    按 OOD 区间为单个特征采样一个值。
    """
    if discrete_values:
        return random.choice(discrete_values)
        
    interval = choose_interval(
        meta.left_min,
        meta.left_max,
        meta.right_min,
        meta.right_max,
    )
    if interval is not None:
        a, b = interval
        return random.uniform(a, b)

    # OOD 区间不可用时，退回原区间
    if meta.vmin is not None and meta.vmax is not None:
        if meta.vmax > meta.vmin:
            return random.uniform(meta.vmin, meta.vmax)
        else:
            return meta.vmin

    return 0.0


def generate_ood_for_dataset(
    root_dir: str,
    dataset_name: str,
    feature_meta: Dict[str, FeatureMeta],
    *,
    samples_per_dataset: int,
    output_file: str,
) -> bool:
    """为单个数据集生成 OOD 样本并写入 ood_test.csv。"""
    header = load_train_header(root_dir, dataset_name)

    # 识别 Target 列名
    target_col = "y"
    if "target" in header:
        target_col = "target"
    elif "y" in header:
        target_col = "y"
    else:
        target_col = header[-1]

    feature_names = [col for col in header if col != target_col]

    for name in feature_names:
        if name not in feature_meta:
            pass # Warning suppressed to reduce noise, handled implicitly

    # 处理离散特征值
    discrete_values_map: Dict[str, List[float]] = {}
    need_discrete = [name for name in feature_names if name in feature_meta and feature_meta[name].dist_type == "离散"]
    
    if need_discrete:
        top, sub = split_dataset_key(dataset_name)
        train_path = os.path.join(root_dir, top, sub, "train.csv")
        value_sets: Dict[str, set] = {name: set() for name in need_discrete}
        with open(train_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for name in need_discrete:
                    try:
                        value_sets[name].add(float(row[name]))
                    except (KeyError, ValueError):
                        continue
        for name, s in value_sets.items():
            if s:
                discrete_values_map[name] = sorted(s)

    try:
        func = load_formula_func(root_dir, dataset_name)
    except Exception as e:
        print(f"  Skipping {dataset_name}: {e}")
        return False

    sig = inspect.signature(func)
    func_params = list(sig.parameters.keys())

    top, sub = split_dataset_key(dataset_name)
    out_path = os.path.join(root_dir, top, sub, output_file)

    # 统计信息
    generated_count = 0
    total_attempts = 0
    max_total_attempts = samples_per_dataset * MAX_TOTAL_ATTEMPTS_MULTIPLIER

    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        while generated_count < samples_per_dataset and total_attempts < max_total_attempts:
            total_attempts += 1
            feat_values: Dict[str, float] = {}
            y_val = float('nan')
            
            # 尝试生成一个有效样本
            # 采样
            for name in feature_names:
                if name in feature_meta:
                    meta = feature_meta[name]
                    feat_values[name] = sample_feature_value(meta, discrete_values_map.get(name))
                else:
                    feat_values[name] = 1.0

            # 计算
            try:
                args = []
                for param_name in func_params:
                    if param_name in feat_values:
                        args.append(feat_values[param_name])
                    else:
                        # 尝试不区分大小写匹配，或者部分匹配？
                        # 这里保持严格匹配，或者如果 feature_names 里有对应的
                        found = False
                        for fn in feature_names:
                            if fn == param_name:
                                args.append(feat_values[fn])
                                found = True
                                break
                        if not found:
                            # 某些公式参数可能是 k_f 而 CSV 列是 kf，这里不做过于复杂的猜测
                            # 直接抛出异常，本次采样失败
                             raise ValueError(f"Missing param {param_name}")

                val = float(func(*args))
                
                # 检查有效性
                if not np.isnan(val) and not np.isinf(val) and not isinstance(val, complex):
                    y_val = val
            except Exception:
                # 计算错误（如数学域错误），视为无效
                pass
            
            # 如果有效，写入
            if not np.isnan(y_val) and not np.isinf(y_val):
                row: List[str] = []
                for col in header:
                    if col == target_col:
                        row.append(f"{y_val:.15g}")
                    else:
                        row.append(f"{feat_values[col]:.15g}")
                writer.writerow(row)
                generated_count += 1
        
        if generated_count < samples_per_dataset:
            print(
                f"  Warning: {dataset_name} - Only generated "
                f"{generated_count}/{samples_per_dataset} valid samples "
                f"(Attempts: {total_attempts})"
            )
    return generated_count > 0


def dataset_selected(dataset_name: str, dataset_prefixes: List[str]) -> bool:
    if not dataset_prefixes:
        return True
    return any(dataset_name.startswith(prefix) for prefix in dataset_prefixes)


def main() -> None:
    args = parse_args()
    root_dir = os.path.abspath(args.root)
    summary_csv = args.summary_csv or os.path.join(root_dir, "feature_distributions.csv")
    dataset_feature_map = load_feature_meta(summary_csv)

    count = 0
    for dataset_name, features in sorted(dataset_feature_map.items()):
        if not dataset_selected(dataset_name, args.dataset_prefix):
            continue

        top, sub = split_dataset_key(dataset_name)
        train_path = os.path.join(root_dir, top, sub, "train.csv")
        if not os.path.isfile(train_path):
            continue

        print(f"Generating OOD for {dataset_name}...")
        if generate_ood_for_dataset(
            root_dir,
            dataset_name,
            features,
            samples_per_dataset=args.samples_per_dataset,
            output_file=args.output_file,
        ):
            count += 1

    print(f"Done. Generated OOD samples for {count} datasets.")

if __name__ == "__main__":
    main()
