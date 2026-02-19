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
