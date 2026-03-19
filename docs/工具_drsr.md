# DRSR 使用文档

## 1. 离线复用（推荐）

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor

reg = SymbolicRegressor(
    "drsr",
    problem_name="oscillator1_offline",
    existing_exp_dir="scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/experiments/<最近一次实验目录>",
    max_samples=1,
)
```

## 2. 在线闭环（需要 API）

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor

reg = SymbolicRegressor(
    "drsr",
    problem_name="my_drsr_check",
    background="y 与 x0, x1 的线性关系：y = 2*x0 - 3*x1",
    use_api=True,
    api_model="blt/gpt-4o-mini",
    samples_per_prompt=1,
    evaluate_timeout_seconds=10,
    max_samples=20,
)
```

## 3. 说明

- 工具映射名：`drsr`
- 运行环境：`sim_llm`
- 在线使用建议设置：`DRSR_ALLOW_ONLINE=1`，并配置 API key（例如 `BLT_API_KEY`）。

## 4. 常用参数

- `spec_path`
- `background`
- `samples_per_prompt`, `evaluate_timeout_seconds`, `max_samples`
- `use_api`, `api_model`, `api_key`, `api_base`, `temperature`
- `existing_exp_dir`

## 5. 闭环脚本

```bash
python check/check_drsr.py
```
