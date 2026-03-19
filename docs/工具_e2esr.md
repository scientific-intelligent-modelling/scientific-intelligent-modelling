# E2ESR 使用文档

## 1. 基础示例

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

rng = np.random.RandomState(11)
X = rng.rand(30, 2)
y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.05 * rng.randn(30)

reg = SymbolicRegressor(
    "e2esr",
    force_cpu=True,
    max_input_points=120,
    n_trees_to_refine=3,
    stop_refinement_after=20,
)
reg.fit(X, y)
```

## 2. 可配参数

- `model_path`: 本地模型路径
- `model_url`: 自动下载地址
- `max_input_points`
- `n_trees_to_refine`
- `rescale`
- `stop_refinement_after`

## 3. 说明

- 工具映射名：`e2esr`
- 运行环境：`sim_e2esr`
- 典型问题：首次模型下载时间较长。

## 4. 闭环脚本

```bash
python check/check_e2esr.py
```
