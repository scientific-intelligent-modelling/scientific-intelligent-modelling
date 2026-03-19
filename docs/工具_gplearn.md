# gplearn 使用文档

## 1. 入口

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

rng = np.random.RandomState(42)
X = rng.rand(64, 2)
y = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.02 * rng.randn(64)

reg = SymbolicRegressor(
    "gplearn",
    population_size=40,
    generations=20,
    tournament_size=5,
    function_set="add,sub,mul,div",
    random_state=42,
)
reg.fit(X, y)
print(reg.get_optimal_equation())
print(reg.get_total_equations()[:3])
print(reg.predict(X[:4]))
```

## 2. 说明

- 工具映射名：`gplearn`
- 运行环境：`sim_base`
- 参数透传到 `gplearn.sklearn.SymbolicRegressor`

## 3. 常用参数

- `population_size`
- `generations`
- `tournament_size`
- `function_set`
- `random_state`
- `n_jobs`

> 其余未写入会抛 `ValueError`，建议只传白名单参数。

## 4. 结果校验

- `get_optimal_equation()` 返回单条最优表达式
- `get_total_equations()` 返回候选表达式列表
- `predict(X)` 返回一维数值数组

## 5. 闭环脚本

```bash
python check/check_gplearn.py
```
