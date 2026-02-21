import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run():
    rng = np.random.RandomState(2024)
    X = rng.rand(20, 2)
    y = 2.5 * X[:, 0] - 1.0 * X[:, 1] + 0.01 * rng.randn(20)

    reg = SymbolicRegressor(
        "iMCTS",
        max_depth=2,
        max_expressions=20,
        K=16,
        c=1.2,
        gamma=0.3,
        exploration_rate=0.1,
        mutation_rate=0.1,
        verbose=False,
        seed=42,
    )

    try:
        reg.fit(X, y)
    except Exception as e:
        # iMCTS 对环境与依赖较敏感，闭环中先保留可恢复策略
        msg = str(e)
        if "ModuleNotFoundError" in msg or "No module named 'iMCTS'" in msg:
            print("[check_imcts] skip: iMCTS 运行环境暂不满足（依赖或安装问题）")
            print(f"reason: {msg}")
            return
        raise

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "iMCTS: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "iMCTS: 方程列表为空")
    assert_ok(pred.shape == (4,), f"iMCTS: 预测形状异常 {pred.shape}")

    print("[check_imcts] OK")
    print("eq:", eq)
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


if __name__ == "__main__":
    run()
