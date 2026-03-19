# DSO 使用文档

## 1. 示例（配置字典）

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

reg = SymbolicRegressor(
    "dso",
    experiment={
        "logdir": "./outputs",
        "exp_name": "demo_dso",
        "seed": 42,
    },
    task={
        "task_type": "regression",
        "function_set": ["add", "sub", "mul", "div"],
        "metric": "inv_nrmse",
        "metric_params": [1.0],
        "threshold": 1e-12,
        "protected": False,
        "complexity": "token",
    },
    training={
        "batch_size": 1,
        "n_samples": 20,
        "epsilon": 0.05,
    },
)
```

## 2. 训练与预测

- `fit(X, y)` 后可直接 `get_optimal_equation()` 与 `get_total_equations()`。
- `predict(X)` 会在序列化恢复场景下回放表达式。

## 3. 说明

- 工具映射名：`dso`
- 运行环境：`sim_dso`
- 依赖较重，首次运行/环境差异时可出现安装与序列化问题；检查脚本内置兼容回退。

## 4. 闭环脚本

```bash
python check/check_dso.py
```
