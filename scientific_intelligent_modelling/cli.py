import argparse
import os
import json
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np

from .srkit.regressor import SymbolicRegressor


def _convert_scalar(value: str) -> Any:
    """将字符串尝试转换为 bool/int/float，失败则原样返回。"""
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_key_value_pairs(pairs: List[str]) -> Dict[str, Any]:
    """
    将命令行中的 key=value 形式参数解析为字典。
    尝试自动推断 int/float/bool 类型，失败则保留为字符串。
    """
    out: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"无效参数格式: '{item}'，应为 key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"无效参数键: '{item}'")
        out[key] = _convert_scalar(value)
    return out


def _parse_unknown_to_kwargs(unknown: List[str]) -> Dict[str, Any]:
    """
    将未知参数列表转换为 kwargs，规则：
    - 支持: --key value
    - 支持: --key=value
    - 若只有 --flag 没有值，则视为 True
    - key 会去掉前缀 '-' 并把 '-' 转成 '_'
    短选项（-k）暂不透传，继续由 argparse 处理。
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("-"):
            # 当前位置参数忽略
            i += 1
            continue

        # 只处理长选项 --xxx
        if token.startswith("--"):
            # 处理 --key=value 形式
            if "=" in token:
                key, value = token[2:].split("=", 1)
                key = key.replace("-", "_")
                kwargs[key] = _convert_scalar(value)
                i += 1
                continue

            # 处理 --key value 或 --flag
            key = token[2:].replace("-", "_")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
                kwargs[key] = _convert_scalar(unknown[i + 1])
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            # 短选项跳过，由 argparse 解析
            i += 1
    return kwargs


def _load_dataset(
    path: str,
    target_column: str | None = None,
    delimiter: str = ",",
    has_header: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从文件中加载训练数据。

    支持：
    - CSV 文本：默认首行为表头，可通过 --no-header 禁用
    - Numpy 二进制：.npy / .npz（默认从 arr_0 读取）

    约定：
    - 若提供 target_column，则从表头中按列名选择 y，其余列为 X
    - 否则默认最后一列为 y，其余列为 X
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"训练数据文件不存在: {path}")

    ext = os.path.splitext(path)[1].lower()

    # Numpy 二进制格式
    if ext in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "arr_0" not in data.files:
                raise ValueError(f"npz 文件中未找到 'arr_0'，实际包含: {data.files}")
            arr = data["arr_0"]
        else:
            arr = data
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("期望二维数组且至少包含 2 列 (特征 + 目标)")
        X = arr[:, :-1]
        y = arr[:, -1]
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    # 文本/CSV
    if has_header:
        # 使用 genfromtxt 读取带表头的文本，得到结构化数组
        arr = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float)
        if arr.size == 0:
            raise ValueError(f"文件为空或无法解析为数值: {path}")
        field_names = list(arr.dtype.names or [])
        if len(field_names) < 2:
            raise ValueError("带表头数据至少需要 2 列 (特征 + 目标)")

        if target_column:
            if target_column not in field_names:
                raise ValueError(
                    f"未找到目标列 '{target_column}'，可用列为: {', '.join(field_names)}"
                )
            y = arr[target_column]
            feature_names = [f for f in field_names if f != target_column]
        else:
            # 默认最后一列为 y
            y = arr[field_names[-1]]
            feature_names = field_names[:-1]

        X_cols = [arr[name] for name in feature_names]
        X = np.vstack(X_cols).T
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    # 无表头文本：按纯数值矩阵处理
    arr = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        raise ValueError("不带表头的数据至少需要 2 列 (特征 + 目标)")
    X = arr[:, :-1]
    y = arr[:, -1]
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器（极简版：只保留算法与训练数据路径）。"""
    parser = argparse.ArgumentParser(
        prog="sim-cli",
        description="Scientific Intelligent Modelling 统一命令行入口",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        required=True,
        help="使用的符号回归算法名称，对应 toolbox_config.json 中的 tool_name，例如: gplearn, pysr, drsr, llmsr 等",
    )
    parser.add_argument(
        "-t",
        "--train-path",
        required=True,
        help="训练数据文件路径，支持: CSV 文本、.npy、.npz",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    """命令行入口函数。"""
    parser = build_parser()
    # 使用 parse_known_args：只消费通用参数，其余透传给具体 wrapper 处理
    args, unknown = parser.parse_known_args(argv)

    algorithm = args.algorithm
    train_path = args.train_path

    # 极简 CLI：除算法与训练路径外，其余长选项全部透传给对应 wrapper
    extra_params: Dict[str, Any] = {}
    # 来自未知长选项的参数（例如 --llm_config_path xxx），全部交给对应 wrapper 自行解析
    extra_kwargs = _parse_unknown_to_kwargs(unknown)
    extra_params.update(extra_kwargs)
    # 将 -a/-t 也作为显式参数传入，便于在 meta.json 中完整记录 CLI 调用
    extra_params.setdefault("algorithm", algorithm)
    extra_params.setdefault("train_path", train_path)

    print(f"[sim-cli] 使用算法: {algorithm}")
    print(f"[sim-cli] 训练数据: {os.path.abspath(train_path)}")

    # 加载数据
    X, y = _load_dataset(
        train_path,
        target_column=None,   # 极简约定：若为 CSV，默认最后一列为目标
        delimiter=",",
        has_header=True,
    )
    print(f"[sim-cli] 加载数据完成: X 形状={X.shape}, y 形状={y.shape}")

    # 构造并训练模型：problem_name/experiments_dir/seed 使用 SymbolicRegressor 默认值
    reg = SymbolicRegressor(
        tool_name=algorithm,
        **extra_params,
    )

    print("[sim-cli] 开始训练模型...")
    reg.fit(X, y)
    print("[sim-cli] 训练完成。")

if __name__ == "__main__":
    main()
