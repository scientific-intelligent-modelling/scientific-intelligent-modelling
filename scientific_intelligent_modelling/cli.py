import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from .srkit.regressor import SymbolicRegressor


def _parse_key_value_pairs(pairs: List[str]) -> Dict[str, Any]:
    """
    将命令行中的 key=value 形式参数解析为字典。
    尝试自动推断 int/float/bool 类型，失败则保留为字符串。
    """

    def _convert(value: str) -> Any:
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

    out: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"无效参数格式: '{item}'，应为 key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"无效参数键: '{item}'")
        out[key] = _convert(value)
    return out


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
    """构建命令行解析器。"""
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
    parser.add_argument(
        "--target-column",
        "-y",
        default=None,
        help="目标列名称（仅当 CSV 含表头时生效），未指定则默认最后一列为目标变量",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV 分隔符（默认 ','）",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="指示 CSV/文本数据不包含表头，按纯数值矩阵读取，最后一列作为目标变量",
    )

    parser.add_argument(
        "--problem-name",
        default=None,
        help="问题名称，用于实验目录与元信息记录（可选）",
    )
    parser.add_argument(
        "--experiments-dir",
        default=None,
        help="实验结果保存的根目录（可选，默认使用当前工作目录下的 ./experiments）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1314,
        help="随机种子，用于实验复现与目录命名（默认 1314）",
    )

    parser.add_argument(
        "--param",
        "-p",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "传递给具体算法的额外参数，可多次使用，例如: "
            "--param population_size=1000 --param generations=20 "
            "或 --param api_model=deepseek/deepseek-chat"
        ),
    )

    parser.add_argument(
        "--print-equation",
        action="store_true",
        help="在训练完成后打印最优符号方程",
    )
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="若指定路径，则将训练集预测结果保存为 .npy 文件",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    """命令行入口函数。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    algorithm = args.algorithm
    train_path = args.train_path

    extra_params = _parse_key_value_pairs(args.param or [])

    print(f"[sim-cli] 使用算法: {algorithm}")
    print(f"[sim-cli] 训练数据: {os.path.abspath(train_path)}")

    # 加载数据
    X, y = _load_dataset(
        train_path,
        target_column=args.target_column,
        delimiter=args.delimiter,
        has_header=not args.no_header,
    )
    print(f"[sim-cli] 加载数据完成: X 形状={X.shape}, y 形状={y.shape}")

    # 构造并训练模型
    reg = SymbolicRegressor(
        tool_name=algorithm,
        problem_name=args.problem_name,
        experiments_dir=args.experiments_dir,
        seed=args.seed,
        **extra_params,
    )

    print("[sim-cli] 开始训练模型...")
    reg.fit(X, y)
    print("[sim-cli] 训练完成。")

    if args.print_equation:
        try:
            eq = reg.get_optimal_equation()
            print("\n===== 最优符号方程 =====")
            print(eq)
            print("===== 结束 =====\n")
        except Exception as e:
            print(f"[sim-cli] 获取最优方程时出错: {e}")

    if args.save_predictions:
        try:
            preds = reg.predict(X)
            out_path = os.path.abspath(args.save_predictions)
            np.save(out_path, preds)
            print(f"[sim-cli] 已将训练集预测结果保存到: {out_path}")
        except Exception as e:
            print(f"[sim-cli] 预测或保存结果时出错: {e}")


if __name__ == "__main__":
    main()

