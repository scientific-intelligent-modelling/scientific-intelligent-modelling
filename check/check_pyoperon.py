import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    rng = np.random.RandomState(202)
    X = rng.rand(35, 3)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.3 * X[:, 2] + 0.01 * rng.randn(35)

    reg = SymbolicRegressor(
        "pyoperon",
        niterations=5,
        population_size=20,
        n_threads=1,
        random_state=42,
    )
    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "pyoperon: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "pyoperon: 方程列表为空")
    assert_ok(pred.shape == (4,), f"pyoperon: 预测形状异常 {pred.shape}")

    print("[check_pyoperon] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
