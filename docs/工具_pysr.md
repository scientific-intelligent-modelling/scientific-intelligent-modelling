# PySR 使用文档

## 1. 入口

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

rng = np.random.RandomState(7)
X = rng.rand(60, 2)
y = 0.5 * X[:, 0] + 2.0 * X[:, 1] + 0.01 * rng.randn(60)

reg = SymbolicRegressor(
    "pysr",
    niterations=20,
    population_size=20,
    n_jobs=1,             # 框架会映射到 PySR 的 procs
    progress=False,
    random_state=42,
)
reg.fit(X, y)
print(reg.get_optimal_equation())
print(reg.get_total_equations()[:2])
print(reg.predict(X[:4]))
```

## 2. 特点

- 工具映射名：`pysr`
- 运行环境：`sim_base`
- `n_jobs` 会透传为 `procs`

## 3. 常用参数

- `niterations`, `population_size`
- `binary_operators`, `unary_operators`
- `parsimony`, `maxsize`, `maxdepth`
- `verbosity`, `progress`, `random_state`
- `constraints`, `nested_constraints`
- `complexity_of_operators`, `complexity_of_constants`, `complexity_of_variables`
- `max_evals`, `early_stop_condition`
- `precision`, `deterministic`

> 仅允许白名单内参数，超出参数会直接报错。

## 4. 适用输出

- `get_total_equations()` 兼容 `pandas.DataFrame` 与 list。

## 5. 闭环脚本

```bash
python check/check_pysr.py
```
