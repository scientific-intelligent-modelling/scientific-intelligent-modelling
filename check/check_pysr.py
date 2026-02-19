import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    rng = np.random.RandomState(7)
    X = rng.rand(40, 2)
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 0.01 * rng.randn(40)

    reg = SymbolicRegressor(
        "pysr",
        niterations=5,
        population_size=20,
        n_jobs=1,
        verbosity=0,
        progress=False,
        random_state=42,
    )
    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:5])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "pysr: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "pysr: 方程列表为空")
    assert_ok(pred.shape == (5,), f"pysr: 预测形状异常 {pred.shape}")

    print("[check_pysr] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
