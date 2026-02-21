import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    rng = np.random.RandomState(11)
    X = rng.rand(30, 2)
    y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.05 * rng.randn(30)

    reg = SymbolicRegressor(
        "e2esr",
        force_cpu=True,
        n_trees_to_refine=3,
        stop_refinement_after=20,
        max_input_points=120,
    )
    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "e2esr: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "e2esr: 方程列表为空")
    assert_ok(pred.shape == (4,), f"e2esr: 预测形状异常 {pred.shape}")

    print("[check_e2esr] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
