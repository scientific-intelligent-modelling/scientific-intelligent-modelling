# AGENTS 约定与执行规范（项目根目录）

## 1. 协作与审批规则

- 任何实现前先给出方案并征求你的确认。
- 需求不明确时，先提出澄清问题。
- 每次完成后如涉及代码/配置修改，需给出可能的风险点与潜在问题。
- 有 bug 时优先补齐复现用例，再修复。

## 2. 开发与文件修改约束

- 尽量一次性完成所需改动，减少频繁打断。
- 涉及 3 个以上文件时，请先拆分为独立小步骤后再执行。
- 禁止回滚或恢复我未修改的他人改动。
- 优先保持现有设计方向，不做不必要的结构重构。

## 3. 环境与依赖管理（关键）

- 项目以 `conda` 环境隔离为主线，不建议直接在 `base` 做常规运行时装包。
- 优先使用官方环境创建器：
  - `python -m scientific_intelligent_modelling.srkit.conda_env_manager`
  - `env_manager.create_environment("sim")`
  - `env_manager.create_environment("sim_base")`
- `envs_config.json` 与 `toolbox_config.json` 是环境/工具映射的真实来源，任何环境/工具变更需同步到这两个文件。
- 新增环境依赖时，先在 `envs_config.json` 声明，再通过创建器同步。
- 遇到 `pysr`/`pyoperon` 等符号回归包，执行导入测试前先确认版本兼容与编译后端是否一致（尤其是 `numpy` 兼容性）。

## 4. 闭环与验收

- 主要任务必须补齐闭环：
  - 变更说明
  - 执行步骤
  - 结果命令输出摘要
- 任何脚本/模型包装器改动要给出最小运行验证。
- 不新增依赖前，先明确是否影响 `conda` 环境一致性。
- gplearn 闭环经验（本项目当前实例）：`_validate_data` 缺失与 `n_features_in_` 缺失属于 `sim_base` 下 sklearn/gplearn 兼容问题，修复时优先在 wrapper 内做兼容垫片，而不是在环境上盲目升级包。
- 参数治理经验（gplearn）：框架会自动注入 `exp_name/exp_path/problem_name/seed` 这类元参数，必须在 wrapper 白名单前剥离，否则会误报为“不受支持参数”。
- `seed` 建议通过 `random_state` 映射进入算法参数，保证子进程链路与主流程可复现性一致。
- gplearn 闭环验证建议：固定一条最小脚本 `fit -> predict -> get_optimal_equation` + 至少一条子进程主链路 `SymbolicRegressor('gplearn', ...)`，两者都通过才算闭环完成。

## 5. 提交规范（必须）

- 每次完成代码生成后，需提交一次。
- 提交信息前缀固定为中文格式：
  - `[update] xxx`：功能/业务更新
  - `[fix] xxx`：问题修复
  - `[beautify] xxx`：结构/格式调整

## 6. 你向我提交或提需求前可默认关注的事项

- `sim` 与 `sim_base` 是否已正确创建并可运行。
- 是否要优先保持原有环境与新功能兼容。
- 是否需要同时更新 `examples`、文档与参数导出接口。
- 超参数搜索与外部算法参数曝光时，需先定义参数白名单与安全范围。

## 7. pysr 落地经验固化（本轮）

- `pysr` 主链路闭环失败的高频原因是参数透传：`SymbolicRegressor` 会注入 `exp_name/exp_path/problem_name/seed`，未剥离会直接喂给 `PySRRegressor` 报错。  
  结论：wrapper 层必须先过滤元参数，再决定是否将 `seed` 映射为 `random_state`。
- `pysr` 与 `sklearn` 风格参数不完全一致：`n_jobs` 需映射到 `procs`。  
  结论：wrapper 应统一适配层，将外部接口风格转成工具实际支持参数。
- `get_total_equations` 返回值在 `pysr` 中常为 `pandas.DataFrame`，子进程落盘时会在 JSON 阶段因类型不可序列化失败。  
  结论：wrapper 必须将方程清单标准化为可序列化字符串/字典列表。
- `pysr` 首次运行会触发 Julia 后端编译/依赖安装，建议在闭环测试记录中将第一次耗时与警告（如 Julia 后端相关）视为正常，不与功能失败混淆。
- `pysr` 新增修复后闭环标准建议：  
  1) `SymbolicRegressor('pysr', ...)` 的 `fit` 成功；  
  2) `get_optimal_equation` 成功；  
  3) `get_total_equations` 可返回 list 并可 JSON 反序列化；  
  4) `predict` 可正常给出数值；  
  5) 直接实例化 `PySRRegressor` 的闭环也需通过。

## 8. 算法验收脚本固化（check）

- 每新增或改造一个算法接入，必须在 `check/` 下新增对应脚本，命名为 `check_<tool>.py`，至少覆盖：
  - `fit`
  - `get_optimal_equation`
  - `get_total_equations`
  - `predict`
- 本次已完成的验收脚本：
  - `check/check_gplearn.py`
  - `check/check_pysr.py`
  - `check/check_pyoperon.py`
- 建议一并提供统一入口用于批量执行（后续可加）：
  - `python check/check_gplearn.py`
  - `python check/check_pysr.py`
  - `python check/check_pyoperon.py`

## 10. DRSR 固化经验（2026-02-20）

- DRSR 的 `drsr_420` 本地代码与统一 LLM 接口存在差异，不能依赖 `set_shared_llm_client` 作为唯一注入方式。
  - 实际兼容流程：
    1. 若 `sampler` / `data_analyse_real` 有 `set_shared_llm_client`，先尝试注入。
    2. 始终通过 `pipeline.main(..., llm_client=client)` 进行强制注入，确保在无旧接口时也能使用统一客户端。
- `config.Config` 不支持 `use_api`/`api_model`，避免按外部默认字段直接透传：
  - 可透传字段主要为 `num_samplers`, `num_evaluators`, `samples_per_prompt`, `evaluate_timeout_seconds`, `results_root`, `wall_time_limit_seconds`.
- 离线复用策略：优先读取 `equation_experiences/experiences.json`，命中即直接恢复方程与参数，避免触发线上 LLM。
- DRSR 在线闭环统一使用模型：
  - `api_model="blt/gpt-4o-mini"`（同 llmsr）
  - 在线运行需加环境变量 `DRSR_ALLOW_ONLINE=1`，并确保 `BLT_API_KEY` 有效。
- 验收脚本：`check/check_drsr.py`
  - 离线：`python check/check_drsr.py`
  - 在线：`DRSR_ALLOW_ONLINE=1 python check/check_drsr.py`
