# LLMSR 使用文档

## 1. 离线用法（推荐）

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

# 先准备已有 llmsr 实验输出（experiments/** 下存在 meta/samples）
reg = SymbolicRegressor("llmsr", existing_exp_dir="scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/experiments/demo")
# 也可先 reg.fit(X, y) 重放；有历史实验时通常更快
```

## 2. 在线闭环（可选）

```python
reg = SymbolicRegressor(
    "llmsr",
    problem_name="my_llmsr_check",
    background="y 与 x0,x1 的线性关系：y = 2*x0 - 3*x1",
    niterations=20,
    samples_per_iteration=1,
)
reg.fit(X, y)
```

## 3. 常用参数

- `background`
- `max_params`, `niterations`, `samples_per_iteration`
- `exp_path`, `exp_name`, `llm_config_path`
- `use_wandb` 系列（wandb 追踪）

## 4. 说明

- 工具映射名：`llmsr`
- 运行环境：`sim_llm`
- 在线模式需配置 API key（例如 `BLT_API_KEY`）与 `LLMSR_ALLOW_ONLINE=1`

## 5. 闭环脚本

```bash
python check/check_llmsr.py
```
