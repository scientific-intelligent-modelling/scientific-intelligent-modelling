import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _build_quick_dso_config():
    return {
        "experiment": {
            "logdir": "./outputs",
            "exp_name": "ci_dso_check",
            "seed": 42,
        },
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "div"],
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "threshold": 1e-12,
            "protected": False,
        },
        "training": {
            "n_samples": 64,
            "batch_size": 8,
            "epsilon": 0.05,
            "n_cores_batch": 1,
            "complexity": "token",
            "verbose": False,
            "early_stopping": True,
        },
        "logging": {
            "save_summary": False,
            "save_all_iterations": False,
            "save_positional_entropy": False,
            "save_pareto_front": False,
            "save_cache": False,
            "save_freq": 1,
            "hof": 1,
        },
        "state_manager": {
            "type": "hierarchical",
            "observe_action": False,
            "observe_parent": True,
            "observe_sibling": True,
            "observe_dangling": False,
            "embedding": False,
            "embedding_size": 8,
        },
    }


def run():
    rng = np.random.RandomState(21)
    X = rng.rand(40, 2)
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1] + 0.01 * rng.randn(40)

    reg = SymbolicRegressor(
        "dso",
        config=_build_quick_dso_config(),
    )
    try:
        reg.fit(X, y)
    except Exception as e:
        # 当前仓库下的 DSO 子仓库对依赖版本约束较重，可能在部分环境中缺少可用安装路径；
        # 为避免阻塞闭环流程，遇到环境不可用则跳过并保留错误信息。
        if "No module named 'dso'" in str(e) or "tensorflow" in str(e) or "protobuf" in str(e):
            print("[check_dso] skip: dso 运行环境暂不满足（依赖/安装问题）")
            print(f"reason: {e}")
            return
        raise

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "dso: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "dso: 方程列表为空")
    assert_ok(pred.shape == (4,), f"dso: 预测形状异常 {pred.shape}")

    print("[check_dso] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
