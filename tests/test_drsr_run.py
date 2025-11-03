import numpy as np
import pandas as pd
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
from sklearn.metrics import mean_squared_error

# 1) 读取 oscillator1 训练数据
train_csv = 'scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/data/oscillator1/train.csv'
df = pd.read_csv(train_csv).astype(float).to_numpy()
X = df[:, :-1]
y = df[:, -1]

# 如需测试集，反注释下面两行替换为 test_id.csv（或 test_ood.csv）
# test_csv = 'scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/data/oscillator1/test_id.csv'
# df_test = pd.read_csv(test_csv).astype(float).to_numpy(); X_test, y_test = df_test[:, :-1], df_test[:, -1]

# 2) 构造 DRSR（联网真实采样）
model = SymbolicRegressor(
    'drsr',
    fast_mode=False,
    use_api=True,
    api_model='blt/gpt-3.5-turbo',  # 或 deepseek/siliconflow/ollama 等
    samples_per_prompt=2,
    max_samples=4,
    evaluate_timeout_seconds=10,
    # 可显式指定 spec_path（默认就是 oscillator1）
    # spec_path='scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/specs/specification_oscillator1_numpy.txt',
)

print('用 oscillator1 训练数据训练 DRSR ...')
model.fit(X, y)

eq = model.get_optimal_equation()
print('最优方程:')
print(eq)

print(model.get_total_equations(5))

preds = model.predict(X)
print('DRSR 训练集 MSE:', mean_squared_error(y, preds))

# 如需测试集评估
# preds_test = model.predict(X_test)
# print('DRSR 测试集 MSE:', mean_squared_error(y_test, preds_test))