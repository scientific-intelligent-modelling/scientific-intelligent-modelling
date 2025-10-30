"""
非 pytest 形式的 DRSR 最小可运行用例。

做法：
- 通过统一入口 SymbolicRegressor('drsr') 调用我们新增的 DRSRRegressor 封装。
- 使用随机生成的二维输入（x, v）与合成目标 y，验证 fit/predict 与方程获取接口。
- 尽量保持与 tests/test_new_arch.py 风格一致，直接执行本脚本即可。
"""

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def main():
    # 生成示例数据：二维输入（x, v），非线性目标
    rng = np.random.default_rng(0)
    X = rng.random((100, 2))
    y = X[:, 0] ** 2 + np.sin(X[:, 1] * np.pi) + 0.1 * rng.standard_normal(100)

    # 使用 DRSR（封装默认 fast_mode=True，无需联网与重优化）
    model = SymbolicRegressor(
        'drsr',
        # 可按需覆盖参数：
        # spec_path='./scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/specs/specification_oscillator1_numpy.txt',
        samples_per_prompt=1,
        max_samples=2,
        evaluate_timeout_seconds=10,
        # 可指定日志/工作目录，便于复现输出：
        # log_dir='./outputs/drsr_logs',
        # workdir='./outputs/drsr_workdir',
    )

    print("训练 drsr 模型...")
    model.fit(X, y)

    eq = model.get_optimal_equation()
    print("最优方程:")
    print(eq)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    print(f"DRSR MSE: {mse}")


if __name__ == "__main__":
    main()

