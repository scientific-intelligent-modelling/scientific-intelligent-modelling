# tests/compare_methods.py
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

# 生成示例数据
X = np.random.rand(100, 2)
y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.1*np.random.randn(100)

# 使用不同工具进行对比
models = {
    'gplearn': SymbolicRegressor('gplearn', population_size=1000, generations=20),
    'pysr': SymbolicRegressor('pysr', niterations=100),
    'srbench': SymbolicRegressor('srbench', method='operon')
}

results = {}
for name, model in models.items():
    print(f"训练 {name} 模型...")
    model.fit(X, y)
    
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    results[name] = mse
    print(f"{name} MSE: {mse}")

print("\n最佳模型:", min(results, key=results.get))