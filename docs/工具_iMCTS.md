# iMCTS 使用文档

## 1. 示例

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

rng = np.random.RandomState(2024)
X = rng.rand(20, 2)
y = 2.5 * X[:, 0] - 1.0 * X[:, 1] + 0.01 * rng.randn(20)

reg = SymbolicRegressor(
    "iMCTS",
    max_depth=2,
    max_expressions=20,
    K=16,
    c=1.2,
    gamma=0.3,
    exploration_rate=0.1,
    mutation_rate=0.1,
    verbose=False,
    seed=42,
)
reg.fit(X, y)
print(reg.get_optimal_equation())
```

## 2. 允许参数（白名单）

- `ops`, `arity_dict`, `context`, `max_depth`, `K`, `c`, `gamma`
- `gp_rate`, `mutation_rate`, `exploration_rate`
- `max_expressions`, `max_constants`, `reward_func`, `optimization_method`

## 3. 说明

- 工具映射名：`iMCTS`
- 运行环境：`sim_iMCTS`
- 检查脚本对环境问题有软跳过逻辑。

## 4. 闭环脚本

```bash
python check/check_imcts.py
```
