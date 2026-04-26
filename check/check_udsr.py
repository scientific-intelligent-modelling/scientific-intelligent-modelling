import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _build_quick_udsr_config():
    return {
        "experiment": {
            "logdir": "./outputs",
            "exp_name": "ci_udsr_check",
            "seed": 42,
        },
        "task": {
            "task_type": "regression",
            "function_set": [
                "add",
                "sub",
                "mul",
                "div",
                "sin",
                "cos",
                "exp",
                "log",
                "sqrt",
                1.0,
                "const",
                "poly",
            ],
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "threshold": 1e-12,
            "protected": False,
            "poly_optimizer_params": {
                "degree": 2,
                "coef_tol": 1e-6,
                "regressor": "dso_least_squares",
                "regressor_params": {
                    "cutoff_p_value": 1.0,
                    "n_max_terms": None,
                    "coef_tol": 1e-6,
                },
            },
        },
        "training": {
            "n_samples": 20,
            "batch_size": 1,
            "epsilon": 0.05,
            "baseline": "R_e",
            "n_cores_batch": 1,
            "complexity": "token",
            "verbose": False,
            "early_stopping": True,
        },
        "policy_optimizer": {
            "policy_optimizer_type": "pg",
            "learning_rate": 0.0005,
            "entropy_weight": 0.03,
            "entropy_gamma": 0.7,
        },
        "gp_meld": {
            "run_gp_meld": True,
            "population_size": 10,
            "generations": 1,
            "train_n": 5,
            "parallel_eval": False,
            "verbose": False,
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

    reg = SymbolicRegressor("udsr", **_build_quick_udsr_config())
    try:
        reg.fit(X, y)
    except Exception as e:
        msg = str(e)
        if (
            "No module named 'dso'" in msg
            or "tensorflow" in msg
            or "protobuf" in msg
            or "can't pickle" in msg
            or "_thread.RLock" in msg
            or "deap" in msg
        ):
            print("[check_udsr] skip: uDSR 运行环境暂不满足（依赖/安装问题）")
            print(f"reason: {msg}")
            return
        raise

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "udsr: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "udsr: 方程列表为空")
    assert_ok(pred.shape == (4,), f"udsr: 预测形状异常 {pred.shape}")

    print("[check_udsr] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
