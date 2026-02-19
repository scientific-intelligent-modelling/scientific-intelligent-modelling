# gplearn Wrapper 落地设计（v0.1）

## 1. 目标
- 让 `gplearn` 在主流程中「可传参、可复现、可闭环」。
- 防止实验元参数（`exp_path`、`exp_name`、`problem_name`、`seed`）错误地透传到 gplearn。
- 兼容当前环境中的 sklearn 版本差异（当前环境 `sim_base` 为 `sklearn 1.7.0`）。

## 2. 已实现策略
### 2.1 参数治理
- 文件：`scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py`
- 白名单：`_ALLOWED_PARAMS`
- 元参数处理：
  - `exp_path`/`exp_name`/`problem_name`：剥离，不透传
  - `seed`：剥离，并在未显式传入 `random_state` 时映射为 `random_state`
- 归一化：
  - `function_set` 支持字符串/列表/元组
  - `init_depth` 支持 `int` 或 `(low, high)`
  - `const_range` 支持 `(low,high)` 或字符串表达式
- 约束检查：
  - 概率参数约束：`p_*` 必须在 `[0,1]` 且非零值和不超过 `1.0`

### 2.2 兼容性
- 文件：同上
- 在 `fit` 之前注入 `_validate_data` 兼容方法到 `gplearn.genetic.SymbolicRegressor`
- 解决报错：`'SymbolicRegressor' object has no attribute '_validate_data'`（`sklearn>=1.7` 与 `gplearn 0.4.3`）

## 3. 对外暴露参数清单（gplearn）
与 `scientific_intelligent_modelling/config/gplearn_schema.json` 保持一致：
- 主要搜索参数：`population_size, generations, tournament_size, parsimony_coefficient, ...`
- 结构参数：`function_set, const_range, init_depth, init_method, max_samples`
- 执行参数：`random_state, n_jobs, verbose, low_memory, warm_start, metric`
- 概率参数约束按 `0~1` 和求和不超过 1

## 4. 闭环建议（主流程）
使用主入口 `SymbolicRegressor('gplearn', ...)` 时，建议至少验证：
1. `fit` 成功
2. `predict` 成功
3. `get_optimal_equation` 成功
4. `get_total_equations` 成功

对不上游 `sim_base` 与 `sklearn/gplearn` 版本时，优先以 `gplearn_schema.json` 里的兼容说明与 `wrapper` 实现为准。

## 5. 本次落地经验沉淀（复用规范）
- `sim_base` 下 `gplearn==0.4.3` + `scikit-learn 1.7.0` 时，主流程最容易踩坑的是：
  - `'_validate_data'` 缺失（需要 wrapper 内注入兼容方法）
  - 子进程反序列化后 `predict` 阶段 `n_features_in_` 可能缺失（兼容 `_validate_data` 时一并同步设置）
- 框架会自动传递的实验元参数（`exp_name/exp_path/problem_name/seed`）不能透传到算法构造，否则会触发“参数不受支持”。
- `seed` 在 wrapper 内统一映射为 `random_state`，避免每次训练链路复现性差异。
- 建议把闭环测试固定为：直接 wrapper 一条 + `SymbolicRegressor('gplearn', ...)` 一条（子进程链路），都必须成功才允许升级参数暴露面。
