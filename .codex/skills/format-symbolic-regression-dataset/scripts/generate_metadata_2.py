#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于结构化 YAML 重写的 metadata_2.yaml 生成脚本。

功能：
1. 读取 feynman/srbench_feynman.csv，填充：
   - 数据集整体描述（context）
   - target 的含义（Meaning）
   - 各个变量的自然语言描述（vars）
2. 读取 feature_distributions.csv，更新每个特征的 train_range / ood_range。
3. 利用 formula.py 和各个 split 的 csv，计算并写入 NMSE。
4. 通过 PyYAML 直接读写 YAML，不再使用逐行状态机与缩进解析。
"""

from __future__ import annotations

import csv
import os
import importlib.util
import inspect
from typing import Dict, Optional, Tuple, Any

import numpy as np
import yaml


ROOT_DIR = "."
SUMMARY_CSV = os.path.join(ROOT_DIR, "feature_distributions.csv")
SRBENCH_FEYNMAN_CSV = os.path.join(ROOT_DIR, "feynman", "srbench_feynman.csv")


# === 1. 公共工具函数 ===

def parse_float(value: str) -> Optional[float]:
    """安全地把字符串转成 float，转换失败返回 None。"""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_ood_ranges() -> Dict[str, Dict[str, Tuple]]:
    """
    从 feature_distributions.csv 加载每个数据集、每个特征的区间信息。

    返回结构：
    {
        "feynman/feynman_I_6_2a": {
            "theta": (left_min, left_max, right_min, right_max, dist_type, vmin, vmax),
            ...
        },
        ...
    }
    """
    result: Dict[str, Dict[str, Tuple]] = {}
    if not os.path.exists(SUMMARY_CSV):
        return result

    with open(SUMMARY_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            dataset_key = row[0]
            cols = row[1:]
            feat_map: Dict[str, Tuple] = {}

            # 每 8 列描述一个特征：name, dist_type, vmin, vmax, left_min, left_max, right_min, right_max
            for i in range(0, len(cols), 8):
                if i + 1 >= len(cols):
                    break
                name = cols[i]
                if not name:
                    continue

                dist_type = cols[i + 1] if i + 1 < len(cols) else ""
                vmin = parse_float(cols[i + 2]) if i + 2 < len(cols) else None
                vmax = parse_float(cols[i + 3]) if i + 3 < len(cols) else None
                left_min = parse_float(cols[i + 4]) if i + 4 < len(cols) else None
                left_max = parse_float(cols[i + 5]) if i + 5 < len(cols) else None
                right_min = parse_float(cols[i + 6]) if i + 6 < len(cols) else None
                right_max = parse_float(cols[i + 7]) if i + 7 < len(cols) else None

                feat_map[name] = (left_min, left_max, right_min, right_max, dist_type, vmin, vmax)

            result[dataset_key] = feat_map

    return result


def load_formula_func(dataset_dir: str):
    """
    动态载入 formula.py 中的真值函数。
    - 若只有一个函数，直接使用它；
    - 若存在名为 y 的函数，优先使用 y；
    - 若没有函数，尝试使用 module.y；
    """
    formula_path = os.path.join(dataset_dir, "formula.py")
    if not os.path.exists(formula_path):
        return None

    spec = importlib.util.spec_from_file_location("formula_module", formula_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    # 允许在公式中使用 np
    module.np = np  # type: ignore[attr-defined]
    try:
        spec.loader.exec_module(module)  # type: ignore[assignment]
    except Exception:
        return None

    functions = [
        obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if obj.__module__ == module.__name__
    ]
    if not functions:
        return getattr(module, "y", None)
    if len(functions) == 1:
        return functions[0]
    for f in functions:
        if f.__name__ == "y":
            return f
    return functions[0]


def compute_nmse(dataset_dir: str, csv_file: str, y_func) -> float:
    """按 srbench 的定义，计算一个 split 的 NMSE。"""
    path = os.path.join(dataset_dir, csv_file)
    if not os.path.isfile(path) or y_func is None:
        return 0.0

    sig = inspect.signature(y_func)
    func_params = list(sig.parameters.keys())

    y_true_list = []
    sq_errors = []

    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # target 列名有三种可能："target" / "y" / 最后一列
            if "target" in fieldnames:
                target_col = "target"
            elif "y" in fieldnames:
                target_col = "y"
            else:
                target_col = fieldnames[-1]

            for row in reader:
                try:
                    args = []
                    for param in func_params:
                        if param not in row:
                            raise ValueError("missing param")
                        args.append(float(row[param]))

                    y_pred = float(row[target_col])
                    y_true = float(y_func(*args))

                    if not (np.isfinite(y_pred) and np.isfinite(y_true)):
                        continue

                    y_true_list.append(y_true)
                    sq_errors.append((y_pred - y_true) ** 2)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
    except Exception:
        return 0.0

    if not y_true_list:
        return 0.0

    var_y = float(np.var(y_true_list))
    if var_y < 1e-12:
        return 0.0

    return float(np.mean(sq_errors) / var_y)


def load_srbench_feynman_csv_info() -> Dict[str, Dict[str, Any]]:
    """
    读取 feynman/srbench_feynman.csv，返回：
    {
        key: {
            "context": <数据集上下文描述>,
            "target_desc": <target 含义>,
            "vars": { 变量名: 描述, ... }
        },
        ...
    }
    其中 key 为第一列 Filename（如 "I.6.2a" 或 "test_1"）。
    """
    info_map: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(SRBENCH_FEYNMAN_CSV):
        return info_map

    with open(SRBENCH_FEYNMAN_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        for row in reader:
            if not row:
                continue

            key = row[0].strip()
            target_symbol = row[1].strip()
            target_meaning = row[2].strip()
            context = row[3].strip()

            vars_data: Dict[str, str] = {}
            # 变量名从第 10 列开始（索引 9），最多 10 个；描述从索引 19 开始
            for i in range(10):
                name_idx = 9 + i
                desp_idx = 19 + i
                if name_idx >= len(row) or desp_idx >= len(row):
                    break

                v_name = row[name_idx].strip()
                v_desp = row[desp_idx].strip()
                if not v_name:
                    continue
                # 这里只保留自然语言描述部分（不加符号前缀）
                vars_data[v_name] = v_desp

            info_map[key] = {
                "context": context,
                "target_desc": target_meaning or target_symbol,
                "vars": vars_data,
            }

    return info_map


def derive_key_from_dataset_name(dataset_full_name: str) -> Optional[str]:
    """
    把目录名映射到 srbench_feynman.csv 的 key。
    - feynman/feynman_I_6_2a -> I.6.2a
    - feynman/feynman_test_1 -> test_1
    - 其他类型的数据集返回 None
    """
    parts = dataset_full_name.split("/")
    if len(parts) < 2:
        return None

    dirname = parts[1]
    if not dirname.startswith("feynman_"):
        return None

    core = dirname[8:]  # 去掉前缀 "feynman_"
    if core.startswith("test_"):
        return core
    return core.replace("_", ".")


# === 2. 处理单个数据集 ===

def process_single_metadata(dataset_key: str, ood_map: Dict[str, Dict[str, Tuple]], srbench_csv_data: Dict[str, Dict[str, Any]]) -> None:
    """
    读取某个数据集的 metadata.yaml，按规则生成 metadata_2.yaml。
    """
    top, sub = dataset_key.split("/")
    dir_path = os.path.join(ROOT_DIR, top, sub)
    meta_path = os.path.join(dir_path, "metadata.yaml")
    if not os.path.isfile(meta_path):
        return

    # 1. 读原始 YAML
    with open(meta_path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    dataset = data.get("dataset", {})

    # 2. Feynman 相关描述信息
    csv_key = derive_key_from_dataset_name(dataset_key)
    dataset_info = srbench_csv_data.get(csv_key, {}) if csv_key else {}

    dataset_desc = dataset_info.get("context") or ""
    target_desc = dataset_info.get("target_desc") or ""
    var_descs = dataset_info.get("vars") or {}

    # 2.1 数据集整体描述
    if dataset_desc:
        dataset["description"] = dataset_desc

    # 3. 计算并写入 splits 的 NMSE
    splits = dataset.get("splits", {})
    y_func = load_formula_func(dir_path)
    if isinstance(splits, dict):
        for split_name, split_cfg in splits.items():
            if not isinstance(split_cfg, dict):
                continue
            nmse_val = 0.0
            if y_func is not None:
                csv_file = split_cfg.get("file")
                if csv_file:
                    nmse_val = compute_nmse(dir_path, csv_file, y_func)
            split_cfg["nmse"] = float(nmse_val)
        dataset["splits"] = splits

    # 4. 更新特征区间 + 描述
    feature_ood = ood_map.get(dataset_key, {})
    features = dataset.get("features", [])
    if isinstance(features, list):
        for feat in features:
            if not isinstance(feat, dict):
                continue

            name = feat.get("name")

            # 4.1 区间更新
            tup = feature_ood.get(name)
            if tup:
                left_min, left_max, right_min, right_max, dist_type, vmin, vmax = tup

                # train_range 统一用 [vmin, vmax]，保持与原逻辑一致
                if vmin is not None and vmax is not None:
                    feat["train_range"] = [vmin, vmax]

                # ood_range：离散用 [vmin, vmax]；连续用两段区间
                if dist_type == "离散" and vmin is not None and vmax is not None:
                    feat["ood_range"] = [vmin, vmax]
                else:
                    segments = []
                    if (left_min is not None) or (left_max is not None):
                        segments.append([left_min, left_max])
                    if (right_min is not None) or (right_max is not None):
                        segments.append([right_min, right_max])
                    if segments:
                        feat["ood_range"] = segments

            # 4.2 描述信息
            desc = var_descs.get(name)
            if desc:
                feat["description"] = desc

        dataset["features"] = features

    # 5. 目标变量描述
    target = dataset.get("target")
    if isinstance(target, dict) and target_desc:
        target["description"] = target_desc
        dataset["target"] = target

    # 写回 dataset
    data["dataset"] = dataset

    # 6. 输出到 metadata_2.yaml
    out_path = os.path.join(dir_path, "metadata_2.yaml")
    # 自定义 Dumper：仅对“数值数组”或“数值数组的数组”使用 flow-style（[] / [[...], [...]]）
    class RangeAwareDumper(yaml.SafeDumper):
        pass

    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _is_numeric_list(lst: Any) -> bool:
        return isinstance(lst, list) and all(_is_number(v) for v in lst)

    def represent_list(dumper: yaml.Dumper, data_list: list):
        flow = False
        if data_list:
            # 情况 1：一维数值数组 -> [a, b, c]
            if all(_is_number(v) for v in data_list):
                flow = True
            # 情况 2：二维数值数组 -> [[a, b], [c, d]]
            elif all(_is_numeric_list(v) for v in data_list):
                flow = True
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq",
            data_list,
            flow_style=flow,
        )

    RangeAwareDumper.add_representer(list, represent_list)

    with open(out_path, "w") as f:
        yaml.dump(
            data,
            f,
            Dumper=RangeAwareDumper,
            sort_keys=False,       # 保持字段顺序更接近原始文件
            allow_unicode=True,    # 允许中文输出
            default_flow_style=False,
        )


# === 3. 主入口 ===

def main() -> None:
    """
    主入口：
    1. 读取 feature_distributions.csv 得到所有数据集、特征的区间信息；
    2. 读取 srbench_feynman.csv 得到 Feynman 系列的语义信息；
    3. 仅遍历 feynman 目录下的数据集，生成/更新 metadata_2.yaml。

    说明：
    - 原始版本会遍历根目录下所有数据集（包括 blackbox 等），在当前环境下对某些目录写
      metadata_2.yaml 会触发权限错误。
    - 目前你的需求是“基于新的 srbench_feynman.csv 重新生成 Feynman 描述”，
      因此这里将处理范围收缩为 dataset_key 以 "feynman/" 开头的条目。
    """
    ood_map = load_ood_ranges()
    srbench_csv_data = load_srbench_feynman_csv_info()

    if not ood_map:
        print("No feature_distributions.csv or empty; nothing to do.")
        return

    print(f"Loaded OOD ranges for {len(ood_map)} datasets.")
    print(f"Loaded Feynman CSV info for {len(srbench_csv_data)} formulas.")

    for dataset_key in sorted(ood_map.keys()):
        if not dataset_key.startswith("feynman/"):
            continue
        process_single_metadata(dataset_key, ood_map, srbench_csv_data)

    print("全部 feynman metadata_2.yaml 已生成/更新完毕。")


if __name__ == "__main__":
    main()
