import numpy as np
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置随机种子，确保结果可复现
np.random.seed(1234)

# 样本数量
n_samples = 1000

# 生成特征
x1 = np.random.uniform(-3, 3, n_samples)
x2 = np.random.uniform(-3, 3, n_samples)
x3 = np.random.uniform(-3, 3, n_samples)

# 创建目标函数: y = sin(x1) + x2^2 - 0.5*x3 + 0.1*x1*x2*x3
# 这个函数包含非线性变换、交互项和多项式项，是个很好的符号回归挑战
y_true = np.sin(x1) + x2**2 - 0.5*x3 + 0.1*x1*x2*x3

# 添加一些噪声使数据更真实
noise = np.random.normal(0, 0.1, n_samples)
y = y_true + noise

# 创建数据框
df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'y': y
})

# 确保数据目录存在
os.makedirs('datasets', exist_ok=True)

# 保存为CSV
df.to_csv('datasets/example.csv', index=False)

print(f"数据集已创建: datasets/example.csv")
print(f"数据集大小: {n_samples}行, {df.shape[1]}列")
print("目标函数: y = sin(x1) + x2^2 - 0.5*x3 + 0.1*x1*x2*x3")