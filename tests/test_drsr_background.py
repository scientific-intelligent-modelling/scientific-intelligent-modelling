import numpy as np
import pandas as pd
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
from sklearn.metrics import mean_squared_error

# 设置随机种子以保证可重复性
np.random.seed(42)

# 生成合成数据：5个特征 + 1个目标变量
# 模拟金融时间序列场景
n_samples = 200

# 特征1: 移动平均 (MA)
ma = np.linspace(100, 150, n_samples) + np.random.normal(0, 5, n_samples)

# 特征2: 动量指标 (Momentum)
momentum = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 10 + np.random.normal(0, 2, n_samples)

# 特征3: 波动率 (Volatility)
volatility = np.abs(np.random.normal(0.5, 0.2, n_samples))

# 特征4: 日历效应 (Calendar effect, 模拟周期性)
calendar = np.cos(np.linspace(0, 10*np.pi, n_samples)) * 5 + np.random.normal(0, 1, n_samples)

# 特征5: 成交量比率 (Volume ratio)
volume_ratio = np.random.uniform(0.5, 2.0, n_samples)

# 组合成特征矩阵
X = np.column_stack([ma, momentum, volatility, calendar, volume_ratio])

# 生成目标变量：基于某种非线性关系
# 例如: y = 0.3*ma + 2*momentum - 15*volatility + 1.5*calendar + 0.8*volume_ratio + 噪声
y = (0.3 * ma + 
     2.0 * momentum - 
     15.0 * volatility + 
     1.5 * calendar + 
     0.8 * volume_ratio + 
     np.random.normal(0, 3, n_samples))

# 打印数据集信息
print("=" * 60)
print("数据集信息:")
print(f"样本数: {n_samples}")
print(f"特征数: {X.shape[1]}")
print(f"特征列名: ['MA', 'Momentum', 'Volatility', 'Calendar', 'VolumeRatio']")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
print("=" * 60)

# 划分训练集和测试集
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print("=" * 60)

# 构造 DRSR 模型（带 background 参数）
model = SymbolicRegressor(
    'drsr',
    use_api=True,
    # api_model='deepseek/deepseek-chat',   # 或 siliconflow/BLT/ollama
    api_model='blt/gpt-3.5-turbo',  # 或 deepseek/siliconflow/ollama 等
    background="""
    这是一个关于金融时间序列建模的问题。
    特征可能包含移动平均、动量、波动率、日历效应等结构。
    数据包含5个特征：
    1. MA: 移动平均值
    2. Momentum: 动量指标
    3. Volatility: 波动率
    4. Calendar: 日历效应（周期性）
    5. VolumeRatio: 成交量比率
    """,
    samples_per_prompt=4,
    max_samples=20,
    evaluate_timeout_seconds=15,
)

print("\n开始训练 DRSR 模型...")
print("=" * 60)
model.fit(X_train, y_train)

# 打印模型信息
print("\n" + "=" * 60)
print("模型训练完成!")
print("=" * 60)
print("\n模型摘要:")
print(model)

# 获取最优方程
eq = model.get_optimal_equation()
print("\n" + "=" * 60)
print("最优方程:")
print("=" * 60)
print(eq)

# 获取前5个方程
print("\n" + "=" * 60)
print("前5个候选方程:")
print("=" * 60)
for i, equation in enumerate(model.get_total_equations(5), 1):
    print(f"\n--- 方程 {i} ---")
    print(equation)

# 训练集预测与评估
preds_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, preds_train)
print("\n" + "=" * 60)
print(f"训练集 MSE: {mse_train:.4f}")

# 测试集预测与评估
preds_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, preds_test)
print(f"测试集 MSE: {mse_test:.4f}")
print("=" * 60)

# 可选：保存数据到 CSV
df_train = pd.DataFrame(X_train, columns=['MA', 'Momentum', 'Volatility', 'Calendar', 'VolumeRatio'])
df_train['Target'] = y_train
df_train.to_csv('tests/drsr_synthetic_train.csv', index=False)

df_test = pd.DataFrame(X_test, columns=['MA', 'Momentum', 'Volatility', 'Calendar', 'VolumeRatio'])
df_test['Target'] = y_test
df_test.to_csv('tests/drsr_synthetic_test.csv', index=False)

print("\n训练和测试数据已保存到:")
print("  - tests/drsr_synthetic_train.csv")
print("  - tests/drsr_synthetic_test.csv")
print("=" * 60)
