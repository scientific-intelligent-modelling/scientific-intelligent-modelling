# QLattice 使用文档

## 1. 示例

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

rng = np.random.RandomState(1)
X = rng.rand(40, 2)
y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.01 * rng.randn(40)

reg = SymbolicRegressor(
    "QLattice",
    n_epochs=20,
    kind="regression",
    signif=4,
)
reg.fit(X, y)
print(reg.get_optimal_equation())
```

## 2. 常用参数

- `n_epochs`, `kind`, `signif`, `output_name`
- 透传参数：`stypes`, `threads`, `max_complexity`, `loss_function`, `criterion`, `query_string`, `starting_models`

## 3. 说明

- 工具映射名：`QLattice`
- 运行环境：`sim_qLattice`

## 4. 闭环脚本

```bash
python check/check_qlattice.py
```
