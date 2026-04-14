#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遍历当前目录下的所有数据集（排除 unprocessed），
对每个数据集子目录中的 train.csv 进行分析：

- 忽略 y 列，只分析特征列
- 判断每个特征是：
  * 常数（所有取值几乎相同）
  * 线性空间上的均匀分布（均匀分布）
  * log10 空间上的均匀分布（log10 均匀分布）
  * 离散（取值个数极少的离散特征）
  * 其他（既不像线性均匀，也不像 log10 均匀，且不是明显的离散特征；在设置 OOD 区间时与均匀分布同样处理）

输出一个汇总表格 feature_distributions.csv：
  第 1 列：数据集名称（形如 "srsd-feynman_easy/feynman-i.12.1"）
  后续列按以下模式展开（每个特征占用一组列）：
    feature_i, dist_i, min_i, max_i,
    left_ood_min_i, left_ood_max_i, right_ood_min_i, right_ood_max_i, ...
  分布类型取值：'常数'、'均匀分布'、'log10均匀分布'、'离散'、'其他'

实现仅依赖标准库，便于在任意环境直接运行。
"""

import csv
import math
import os
from typing import Dict, List, Tuple


def read_train_csv(path: str) -> Dict[str, List[float]]:
    """读取单个 train.csv，返回：列名 -> 数值列表（跳过 y 列）"""
    columns: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # 初始化除 y 以外的列
        for name in fieldnames:
            if name in ["y", "target"]:
                continue
            columns[name] = []

        for row in reader:
            for name in columns:
                value_str = row.get(name, "")
                if value_str == "" or value_str is None:
                    continue
                try:
                    value = float(value_str)
                except ValueError:
                    # 非数值内容直接跳过
                    continue
                columns[name].append(value)

    return columns


def is_constant(values: List[float], eps: float = 1e-12) -> bool:
    """判断是否为常数列：最大值与最小值之差极小"""
    if not values:
        return True
    v_min = min(values)
    v_max = max(values)
    return (v_max - v_min) <= eps


def ks_stat_uniform(z: List[float]) -> float:
    """
    计算样本 z（已映射到 [0, 1] 区间）的 Kolmogorov-Smirnov 统计量，
    用来衡量与理想均匀分布 U[0,1] 的偏差。
    """
    n = len(z)
    if n == 0:
        return 0.0
    z_sorted = sorted(z)
    d_plus = 0.0
    d_minus = 0.0
    for i, value in enumerate(z_sorted, start=1):
        # 理论 CDF：F(x) = x
        emp_cdf = i / n
        d_plus = max(d_plus, emp_cdf - value)
        d_minus = max(d_minus, value - (i - 1) / n)
    return max(d_plus, d_minus)


def classify_distribution(values: List[float]) -> str:
    """
    根据数值序列判断分布类型：
      - 所有值几乎相等 -> '常数'
      - 若全部为正数：
          比较原始空间与 log10 空间下的 KS 统计量，
          哪个更接近均匀分布（KS 较小）就判为对应类型
      - 若包含非正数：无法为 log10 均匀，仅考虑原始空间的均匀性
      - 当与上述理想分布偏差过大时，标记为 '其他'
    """
    # 常数列判定
    if is_constant(values):
        return "常数"

    v_min = min(values)
    v_max = max(values)
    if v_max <= v_min:
        # 理论上前面已经排除了常数，这里作为保护
        return "常数"

    # 原始空间映射到 [0, 1]
    lin_z: List[float] = [(v - v_min) / (v_max - v_min) for v in values]
    ks_lin = ks_stat_uniform(lin_z)

    # 设定一个简单阈值，决定是否认为近似均匀
    # 阈值越小，判定越严格；这里选取 0.1 作为经验值
    ks_threshold = 0.1

    # 若存在非正数，则不考虑 log10 均匀，只在原始空间上判断
    if any(v <= 0 for v in values):
        if ks_lin <= ks_threshold:
            return "均匀分布"
        else:
            return "其他"

    # log10 空间映射到 [0, 1]
    log_values: List[float] = []
    for v in values:
        try:
            log_values.append(math.log10(v))
        except (ValueError, OverflowError):
            # 理论上不会发生，因为已经筛掉了非正数
            log_values = []
            break

    if not log_values:
        # 无法在 log 空间分析时，只依赖原始空间
        if ks_lin <= ks_threshold:
            return "均匀分布"
        else:
            return "其他"

    log_min = min(log_values)
    log_max = max(log_values)
    if log_max <= log_min:
        # log 空间退化，退回原始空间判断
        if ks_lin <= ks_threshold:
            return "均匀分布"
        else:
            return "其他"

    log_z: List[float] = [(lv - log_min) / (log_max - log_min) for lv in log_values]
    ks_log = ks_stat_uniform(log_z)

    # 比较两个空间下哪个更接近均匀分布
    best_ks = min(ks_lin, ks_log)
    if best_ks > ks_threshold:
        return "其他"

    tol = 1e-3
    if ks_log + tol < ks_lin:
        return "log10均匀分布"
    else:
        return "均匀分布"


def collect_all_datasets(root: str) -> List[Tuple[str, str]]:
    """
    收集所有数据集：
      - 遍历 root 下的一级目录，排除 'unprocessed'
      - 在每个一级目录下，再遍历其子目录，若存在 train.csv 则认为是一个数据集
    返回 (数据集名称, train.csv 路径) 列表。
    """
    results: List[Tuple[str, str]] = []
    for top in sorted(os.listdir(root)):
        if top == "unprocessed":
            continue
        top_path = os.path.join(root, top)
        if not os.path.isdir(top_path):
            continue

        for sub in sorted(os.listdir(top_path)):
            sub_path = os.path.join(top_path, sub)
            if not os.path.isdir(sub_path):
                continue
            train_path = os.path.join(sub_path, "train.csv")
            if os.path.isfile(train_path):
                dataset_name = f"{top}/{sub}"
                results.append((dataset_name, train_path))

    return results


def load_feature_types_for_dataset(dataset_name: str, train_path: str) -> Dict[str, str]:
    """
    读取同目录下的 metadata.yaml，得到每个特征的类型（continuous / discrete），
    若不存在或无法解析，则返回空映射。
    """
    feature_types: Dict[str, str] = {}
    dir_path = os.path.dirname(train_path)
    meta_path = os.path.join(dir_path, "metadata.yaml")
    if not os.path.isfile(meta_path):
        return feature_types

    in_features = False
    current_feature: Optional[str] = None

    with open(meta_path, "r") as f:
        for line in f:
            stripped = line.lstrip()

            if stripped.startswith("features:"):
                in_features = True
                current_feature = None
                continue

            # 进入其他顶层块则离开 features
            if stripped.startswith("target:") or stripped.startswith("ground_truth_formula:") or stripped.startswith(
                "license:"
            ):
                in_features = False
                current_feature = None
                continue

            if not in_features:
                continue

            # 识别特征名
            if stripped.startswith("- name:"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    name_part = parts[1].strip()
                    if name_part.startswith(("'", '"')) and name_part.endswith(("'", '"')) and len(name_part) >= 2:
                        name_part = name_part[1:-1]
                    current_feature = name_part
                else:
                    current_feature = None
                continue

            # 在当前特征块内读取 type 字段
            if current_feature is not None and stripped.startswith("type:"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    type_part = parts[1].strip()
                    feature_types[current_feature] = type_part

    return feature_types


def main() -> None:
    root = "."
    datasets = collect_all_datasets(root)
    if not datasets:
        print("未找到任何包含 train.csv 的数据集目录。")
        return

    # 第一次遍历：分析每个数据集的特征、分布类型和取值范围
    # 内部列表元素结构：(特征名, 分布类型, 最小值, 最大值)
    dataset_feature_info: List[Tuple[str, List[Tuple[str, str, float, float]]]] = []
    max_feature_count = 0

    for dataset_name, train_path in datasets:
        columns = read_train_csv(train_path)
        feature_types = load_feature_types_for_dataset(dataset_name, train_path)
        feature_results: List[Tuple[str, str, float, float]] = []
        for feature_name in sorted(columns.keys()):
            values = columns[feature_name]
            # 若 metadata 中显式标记为离散，则优先视为离散特征
            f_type = feature_types.get(feature_name, "")
            if f_type == "discrete":
                dist_type = "离散"
            else:
                dist_type = classify_distribution(values)

            if values:
                v_min = min(values)
                v_max = max(values)
            else:
                v_min = float("nan")
                v_max = float("nan")

            feature_results.append((feature_name, dist_type, v_min, v_max))

        dataset_feature_info.append((dataset_name, feature_results))
        if len(feature_results) > max_feature_count:
            max_feature_count = len(feature_results)

    # 构造表头：
    #   dataset,
    #   feature_1, dist_1, min_1, max_1,
    #   left_ood_min_1, left_ood_max_1, right_ood_min_1, right_ood_max_1,
    #   feature_2, dist_2, min_2, max_2,
    #   left_ood_min_2, left_ood_max_2, right_ood_min_2, right_ood_max_2, ...
    header: List[str] = ["dataset"]
    for i in range(1, max_feature_count + 1):
        header.append(f"feature_{i}")
        header.append(f"dist_{i}")
        header.append(f"min_{i}")
        header.append(f"max_{i}")
        header.append(f"left_ood_min_{i}")
        header.append(f"left_ood_max_{i}")
        header.append(f"right_ood_min_{i}")
        header.append(f"right_ood_max_{i}")

    output_path = os.path.join(root, "feature_distributions.csv")
    with open(output_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for dataset_name, feature_results in dataset_feature_info:
            row: List[str] = [dataset_name]
            # 按 feature_1, dist_1, min_1, max_1, left_ood_min_1, ..., 顺序填充
            for feature_name, dist_type, v_min, v_max in feature_results:
                row.append(feature_name)
                row.append(dist_type)
                row.append("" if math.isnan(v_min) else f"{v_min:.8g}")
                row.append("" if math.isnan(v_max) else f"{v_max:.8g}")

                # 计算 OOD 区间
                # 说明：
                #   - 对于 'log10均匀分布'：
                #       R = max / min
                #       左 OOD: [min * R^{-0.05}, min]
                #       右 OOD: [max, max * R^{0.05}]
                #   - 对于 '均匀分布' 和 '其他'：
                #       len = max - min
                #       左 OOD: [min - len * 0.05, min]
                #       右 OOD: [max, max + len * 0.05]
                #   - '常数' 或范围退化时，OOD 留空
                if math.isnan(v_min) or math.isnan(v_max) or v_max <= v_min:
                    # 无法定义合理区间
                    row.extend(["", "", "", ""])
                else:
                    left_min = left_max = right_min = right_max = None
                    if dist_type == "log10均匀分布" and v_min > 0:
                        R = v_max / v_min if v_min != 0 else float("inf")
                        if R > 0 and math.isfinite(R):
                            factor = math.pow(R, 0.05)
                            inv_factor = math.pow(R, -0.05)
                            left_min = v_min * inv_factor
                            left_max = v_min
                            right_min = v_max
                            right_max = v_max * factor
                    elif dist_type in ("均匀分布", "其他", "常数"):
                        length = v_max - v_min
                        if length > 0:
                            delta = length * 0.05
                            # 默认双侧各扩展 5%
                            left_min = v_min - delta
                            left_max = v_min
                            right_min = v_max
                            right_max = v_max + delta

                            # 额外约束：若原区间不跨 0，则 OOD 也不应跨 0
                            #   - 若原区间在正数侧且左扩展会穿过 0，则改为仅向右扩展，
                            #     且向右扩展比例改为 10%
                            #   - 若原区间在负数侧且右扩展会穿过 0，则改为仅向左扩展，
                            #     且向左扩展比例改为 10%
                            if v_min >= 0 and left_min < 0:
                                # 仅向正方向扩展：右侧 10%
                                left_min = v_min
                                left_max = v_min
                                right_min = v_max
                                right_max = v_max + length * 0.10
                            elif v_max <= 0 and right_max > 0:
                                # 仅向负方向扩展：左侧 10%
                                left_min = v_min - length * 0.10
                                left_max = v_min
                                right_min = v_max
                                right_max = v_max

                    # 将 OOD 区间写入表格；无法计算时写空
                    if left_min is None:
                        row.extend(["", "", "", ""])
                    else:
                        row.append(f"{left_min:.8g}")
                        row.append(f"{left_max:.8g}")
                        row.append(f"{right_min:.8g}")
                        row.append(f"{right_max:.8g}")
            # 若该数据集特征数少于最大值，其余列留空
            missing_pairs = max_feature_count - len(feature_results)
            for _ in range(missing_pairs):
                # feature, dist, min, max, left_ood_min, left_ood_max, right_ood_min, right_ood_max
                row.extend([""] * 8)
            writer.writerow(row)

    print(f"分析完成，结果已写入：{output_path}")


if __name__ == "__main__":
    main()
