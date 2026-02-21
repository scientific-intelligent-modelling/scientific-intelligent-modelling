import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    # 固定随机子保证可复现
    rng = np.random.RandomState(42)
    X = rng.rand(40, 2)
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.01 * rng.randn(40)

    reg = SymbolicRegressor(
        "gplearn",
        n_jobs=1,
        generations=20,
        population_size=20,
        tournament_size=3,
        function_set="add,sub,mul,div",
        random_state=42,
        verbose=0,
    )
    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:5])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "gplearn: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "gplearn: 方程列表为空")
    assert_ok(pred.shape == (5,), f"gplearn: 预测形状异常 {pred.shape}")

    print("[check_gplearn] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
