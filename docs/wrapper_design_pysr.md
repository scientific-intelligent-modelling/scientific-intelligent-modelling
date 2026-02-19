# pysr Wrapper 落地设计（v0.1）

## 1. 目标
- 让 `pysr` 的参数可控、可搜索、可复现，默认对外只暴露可验证参数。
- 适配主流程 `SymbolicRegressor('pysr')` 子进程执行链路，不影响现有 `gplearn` 方案。
- 在主流程和直接 wrapper 两条路径上都能通过闭环测试。

## 2. 当前现状（待改造点）
- 文件：`scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py`
- 现状：`self.params = kwargs` 直接透传到 `pysr.PySRRegressor`。
- 风险：
  - 框架注入的元参数（`exp_name/exp_path/problem_name/seed`）会透传，造成调用失败。
  - 缺少参数类型/范围校验，参数错配时报错不友好。
  - 算法端依赖（pysr/Julia）不稳定，闭环时难定位。

## 3. wrapper 设计约束（对齐统一规范）
### 3.1 参数分层
- `system_params`（不入模型）：`exp_name/exp_path/problem_name/seed`
- `algo_params`：严格白名单 + 类型检查 + 默认值
- `resource params`：`timeout`、`procs/n_jobs` 等执行控制

### 3.2 元参数处理
- `seed` 作为元参数不直接透传。
- 若用户未显式传 `random_state`，则映射 `seed -> random_state`（用于复现）。
- 允许 `n_jobs` 映射到 `procs`（若 `procs` 未显式提供）。

### 3.3 兼容性兜底
- 统一捕获 `import pysr` / `SymbolicRegressor` 异常，抛出可读错误。
- 约束 `binary_operators`、`unary_operators` 的输入格式（字符串、列表、元组均可）。
- `get_total_equations` 返回稳定字符串列表；缺失时返回空列表而不是抛异常（至少保持闭环可观测）。

## 4. 建议参数 schema（v0.1）
> 详细字段建议同步到 `scientific_intelligent_modelling/config/pysr_schema.json`

- 建议暴露参数（高优先）：
  - `niterations:int > 0`
  - `population_size:int > 0`
  - `populations:int > 0`
  - `ncycles_per_iteration:int > 0`
  - `procs:int > 0`（等效 `n_jobs`）
  - `maxsize:int > 0`
  - `maxdepth:int >= 1`
  - `parsimony:float >= 0`
  - `random_state:int`
  - `binary_operators`（`list|tuple|string`）
  - `unary_operators`（`list|tuple|string`）
  - `elementwise_loss:string`
  - `model_selection:string`
  - `loss_function:string`
  - `warm_start:bool`
  - `progress:bool`
  - `verbosity:int >= 0`
- 建议实验默认：
  - `niterations: 100~500`
  - `population_size: 50~200`
  - `ncycles_per_iteration: 200~1500`
  - `maxsize: 10~40`
  - `maxdepth: 4~12`
  - `parsimony: 1e-6~1e-2`

## 5. 闭环测试（主流程）
- 数据建议：`tests/datasets/example.csv`
- 子流程：
  1. 直接 wrapper：`PySRRegressor(...).fit(X, y)`
  2. 主流程：`SymbolicRegressor('pysr', ...).fit(X, y)`
  3. `predict(X[:N])` 成功
  4. `get_optimal_equation` 成功
  5. `get_total_equations` 成功
- 最小通过标准：两条链路均不抛异常，输出方程可序列化字符串。

## 6. 失败归因（建议的错误提示标准化）
- `IMPORT_ERR`：PySR 未安装 / Julia 环境缺失
- `PARAM_INVALID`：参数名未在白名单、类型不合法、范围错误
- `FIT_TIMEOUT`：训练超时
- `EQUATION_MISSING`：模型未返回可解析符号表达式

## 7. 风险与隔离
- `pysr` 与底层 Julia/编译环境耦合度高，优先在 `sim_base` 环境固定版本。
- 高耗时任务默认不放宽 `timeout`，推荐加入执行超时与失败重试策略。
- 超参数搜索建议先固定算子集合，再放开 `niterations/population_size` 等搜索量参数，避免搜索空间过大。
