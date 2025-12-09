import argparse
import os
import json
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
from contextlib import redirect_stdout, redirect_stderr

import numpy as np

from .srkit.regressor import SymbolicRegressor


def _resolve_dataset_path(path: str) -> str:
    """
    解析数据集路径：
    - 若设置了环境变量 SIM_DATASETS_PATH，且传入路径为相对路径，
      则将其视作数据根目录并进行拼接；
    - 否则直接使用传入路径。
    最终返回绝对路径，便于日志与后续处理。
    """
    env_var_name = "SIM_DATASETS_PATH"
    base = os.environ.get(env_var_name)
    if base:
        base = os.path.expanduser(base)
        # 仅对相对路径做拼接，避免覆盖用户显式指定的绝对路径
        if not os.path.isabs(path):
            path = os.path.join(base, path)
    return os.path.abspath(path)


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
    # 先结合环境变量解析数据路径
    path = _resolve_dataset_path(path)
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

    # 数据集名称（用于 WandB 等记录）
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        type=str,
        help="数据集名称（将记录到 WandB 的 config.dataset.name 中）",
    )

    # 全局随机种子：会透传到具体算法包装器（如 LLMSR），并写入实验元信息
    parser.add_argument(
        "--seed",
        type=int,
        default=1314,
        help="随机种子（默认 1314，用于实验目录命名与下游算法的 seed 参数）",
    )

    # prompts 类型/版本标签（用于区分当前使用的提示词版本）
    parser.add_argument(
        "--prompts-type",
        "--prompts_type",
        dest="prompts_type",
        type=str,
        help="当前使用的 prompts 类型/版本标签（将记录到 WandB 的 config.prompts_type 中）",
    )

    # WandB 参数
    parser.add_argument("--use_wandb", action="store_true", help="是否启用 WandB 实验记录")
    parser.add_argument("--wandb_project", type=str, default="my-awesome-project", help="WandB 项目名称")
    parser.add_argument("--wandb_entity", type=str, default="my-awesome-entity", help="WandB 实体/用户名")
    parser.add_argument("--wandb_name", type=str, default="sim-run", help="WandB 实验名称")
    parser.add_argument("--wandb_group", type=str, default="sim-group", help="WandB 分组")
    parser.add_argument("--wandb_tags", type=str, default="sim,llmsr", help="WandB 标签，逗号分隔")

    # 实验日志重定向开关
    parser.add_argument(
        "--redirect_io",
        action="store_true",
        help="若开启，则将 sim-cli 的标准输出和错误重定向到实验目录中的 std.out/std.err",
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
    # 显式传入数据集名称，便于下游记录到 WandB
    if getattr(args, "dataset_name", None):
        extra_params.setdefault("dataset_name", args.dataset_name)
    # 显式传入 seed，保证下游包装器与 LLMSRRegressor 能够拿到与 manifest 一致的种子
    if getattr(args, "seed", None) is not None:
        extra_params.setdefault("seed", args.seed)
    # 显式传入 prompts_type，便于在 WandB 中记录当前 prompts 类型/版本
    if getattr(args, "prompts_type", None):
        extra_params.setdefault("prompts_type", args.prompts_type)

    # WandB 相关参数仅在开启时透传，避免无谓污染参数空间
    if args.use_wandb:
        wandb_tags = None
        if args.wandb_tags:
            wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        extra_params.update(
            {
                "use_wandb": True,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_name": args.wandb_name,
                "wandb_group": args.wandb_group,
                "wandb_tags": wandb_tags,
            }
        )

    print(f"[sim-cli] 使用算法: {algorithm}")
    resolved_train_path = _resolve_dataset_path(train_path)
    print(f"[sim-cli] 训练数据: {resolved_train_path}")

    # 先构造回归器以创建实验目录，便于重定向日志到实验目录
    reg = SymbolicRegressor(
        tool_name=algorithm,
        **extra_params,
    )

    if args.redirect_io:
        # 将后续 stdout/stderr 重定向到实验目录中的 std.out / std.err
        exp_dir = getattr(reg, "experiment_dir", os.getcwd())
        os.makedirs(exp_dir, exist_ok=True)
        stdout_path = os.path.join(exp_dir, "std.out")
        stderr_path = os.path.join(exp_dir, "std.err")
        with open(stdout_path, "a", encoding="utf-8") as f_out, open(
            stderr_path, "a", encoding="utf-8"
        ) as f_err, redirect_stdout(f_out), redirect_stderr(f_err):
            print(f"[sim-cli] 使用算法: {algorithm}")
            print(f"[sim-cli] 训练数据: {resolved_train_path}")

            # 加载数据
            X, y = _load_dataset(
                resolved_train_path,
                target_column=None,   # 极简约定：若为 CSV，默认最后一列为目标
                delimiter=",",
                has_header=True,
            )
            print(f"[sim-cli] 加载数据完成: X 形状={X.shape}, y 形状={y.shape}")

            print("[sim-cli] 开始训练模型...")
            reg.fit(X, y)
            print("[sim-cli] 训练完成。")
    else:
        # 不开启重定向时保持原有行为
        X, y = _load_dataset(
            resolved_train_path,
            target_column=None,   # 极简约定：若为 CSV，默认最后一列为目标
            delimiter=",",
            has_header=True,
        )
        print(f"[sim-cli] 加载数据完成: X 形状={X.shape}, y 形状={y.shape}")

        print("[sim-cli] 开始训练模型...")
        reg.fit(X, y)
        print("[sim-cli] 训练完成。")

    # WandB 记录
    if args.use_wandb:
        pass

if __name__ == "__main__":
    main()
