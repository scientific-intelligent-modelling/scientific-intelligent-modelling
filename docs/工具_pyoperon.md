# PyOperon 使用文档

## 1. 入口

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

rng = np.random.RandomState(123)
X = rng.rand(50, 3)
y = 1.0 * X[:, 0] - 1.2 * X[:, 1] + 0.8 * X[:, 2] + 0.01 * rng.randn(50)

reg = SymbolicRegressor(
    "pyoperon",
    niterations=20,
    population_size=30,
    n_threads=1,
    random_state=42,
)
reg.fit(X, y)
print(reg.get_optimal_equation())
print(reg.get_total_equations()[:2])
print(reg.predict(X[:3]))
```

## 2. 说明

- 工具映射名：`pyoperon`
- 运行环境：`sim_base`
- `seed` 会被映射成 `random_state`

## 3. 常用参数

- `generations`, `population_size`, `n_threads`
- `max_length`, `max_depth`, `tournament_size`
- `n_iterations` / `niteration` 不支持（请直接用 `generations`/`population_size`）
- `n_jobs` 不在白名单，请改用 `n_threads`

## 4. 注意

- `get_total_equations()` 在反序列化场景下也可回放。

## 5. 闭环脚本

```bash
python check/check_pyoperon.py
```
