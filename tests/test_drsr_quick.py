import numpy as np
import pandas as pd
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
from sklearn.metrics import mean_squared_error

# 设置随机种子
np.random.seed(42)

# 生成小规模合成数据：5个特征
n_samples = 50
X = np.random.randn(n_samples, 5)
y = 0.3 * X[:, 0] + 2.0 * X[:, 1] - 15.0 * X[:, 2] + 1.5 * X[:, 3] + 0.8 * X[:, 4] + np.random.normal(0, 0.5, n_samples)

print("=" * 60)
print(f"测试数据: {n_samples} 样本, {X.shape[1]} 特征")
print("=" * 60)

# 构造 DRSR 模型
model = SymbolicRegressor(
    'drsr',
    use_api=True,
    api_model='blt/gpt-3.5-turbo',
    background="""
    这是一个简单的线性回归测试问题。
    特征: 5个随机变量
    目标: 线性组合
    """,
    samples_per_prompt=4,
    max_samples=40,  # 极少采样，快速测试
    evaluate_timeout_seconds=10,
)

print("\n开始训练...")
model.fit(X, y)

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
print(model)

print("\n" + "=" * 60)
print("最优方程:")
print("=" * 60)
eq = model.get_optimal_equation()
print(eq)

print("\n" + "=" * 60)
print("测试预测...")
print("=" * 60)
preds = model.predict(X)
mse = mean_squared_error(y, preds)
print(f"训练集 MSE: {mse:.6f}")
print("=" * 60)

print("\n✅ 测试成功！方程可以正常预测。")
