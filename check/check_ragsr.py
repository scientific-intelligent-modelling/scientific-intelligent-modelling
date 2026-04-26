import importlib.util
import os

import numpy as np

from scientific_intelligent_modelling.algorithms.ragsr_wrapper.wrapper import RAGSRRegressor
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _has_evolutionary_forest() -> bool:
    return importlib.util.find_spec("evolutionary_forest") is not None


def _make_smoke_data():
    rng = np.random.RandomState(27)
    X = rng.rand(36, 2)
    y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.01 * rng.randn(36)
    return X, y


def _run_with_symbolic_regressor(X, y):
    reg = SymbolicRegressor(
        "ragsr",
        problem_name="ci_ragsr_check",
        seed=42,
        n_features=2,
        feature_names=["x0", "x1"],
        target_name="y",
        n_gen=1,
        n_pop=10,
        gene_num=2,
        max_height=4,
        max_trees=100,
        verbose=False,
    )
    reg.fit(X, y)
    return reg


def _run_direct_wrapper(X, y):
    reg = RAGSRRegressor(
        seed=42,
        n_features=2,
        feature_names=["x0", "x1"],
        target_name="y",
        n_gen=1,
        n_pop=10,
        gene_num=2,
        max_height=4,
        max_trees=100,
        verbose=False,
    )
    reg.fit(X, y)
    return reg


def run():
    if not _has_evolutionary_forest():
        print("[check_ragsr] skip: 当前 Python 环境未安装 evolutionary_forest")
        return

    X, y = _make_smoke_data()
    if os.getenv("RAGSR_CHECK_DIRECT", "").lower() in {"1", "true", "yes", "on"}:
        reg = _run_direct_wrapper(X, y)
    else:
        reg = _run_with_symbolic_regressor(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])
    artifact = reg.export_canonical_symbolic_program()

    assert_ok(isinstance(eq, str) and len(eq) > 0, "ragsr: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "ragsr: 方程列表为空")
    assert_ok(pred.shape == (4,), f"ragsr: 预测形状异常 {pred.shape}")
    assert_ok(artifact.get("artifact_valid") is True, f"ragsr: canonical artifact 无效 {artifact}")

    print("[check_ragsr] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
