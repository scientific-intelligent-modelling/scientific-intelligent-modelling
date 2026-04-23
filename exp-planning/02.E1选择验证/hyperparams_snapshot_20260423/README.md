# 基准算法超参数快照

这份目录用于固定保存当前 benchmark 计划中各阶段采用的算法超参数口径。

## 用途

- 避免后续重新运行 `generate_e1_formal_assets.py` 时覆盖当前参数口径
- 为 `W1~W5` 分发、复查和论文复现实验提供一份稳定快照
- 明确当前 `E1 / E3 / E6` 采用的是一套**可执行的工程基线**，不是最终调优版
- 10 算法总览表见：
  - [hyperparams_overview.md](./hyperparams_overview.md)

## 当前包含的算法

- `gplearn`
- `pysr`
- `pyoperon`
- `llmsr`
- `drsr`
- `dso`
- `tpsr`
- `e2esr`
- `qlattice`
- `imcts`

## 文件说明

- `gplearn.json`
  - `W1` 使用
  - 经典 GP 基线
- `llmsr.json`
  - `W1` 使用
  - 纯 LLM-based SR
- `pyoperon.json`
  - `W2` 使用
  - 现代 EA / tree-search 风格
- `drsr.json`
  - `W2` 使用
  - LLM-hybrid / DSR 风格
- `pysr.json`
  - `W3` 使用
  - 现代演化式主力基线
- `dso.json`
  - `W4` 使用
  - 纯 RL-based SR
- `tpsr.json`
  - `W5` 使用
  - Transformer + planning
- `e2esr.json`
  - `E3 / E6` 使用
  - E2E Transformer-based SR
- `qlattice.json`
  - `E3 / E6` 使用
  - QLattice 图结构搜索
- `imcts.json`
  - `E3 / E6` 使用
  - MCTS-based symbolic regression

## 参数来源

这些文件是从：

- `exp-planning/02.E1选择验证/generated/params/*.json`

复制出来的快照版本，对应当前 `E1` 资产生成器口径。

## 备注

- 当前默认 `timeout_in_seconds = 3600`
- 当前默认 `progress_snapshot_interval_seconds = 60`
- `e2esr / qlattice / imcts` 不属于 `E1` 七算法，而是用于：
  - `E3` 的 10 算法轻量验证
  - `E6` 的最终 leaderboard
- `llmsr / drsr` 使用固定 benchmark 配置：
  - `/home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config`
  - 仓库内只提交：
    - `benchmark_llm.config.example`
  - 真实 `benchmark_llm.config` 仅保留在本地与远端磁盘，不进入 Git
- `tpsr` 当前参数是官方主配置对齐后的 benchmark 安全口径
- `gplearn` 当前不再使用“仅四则运算”的极保守配置，而采用：
  - `add,sub,mul,div,sqrt,log,sin,cos`
  - 目标是在 `E1` 中避免因表达能力过弱而系统性低估 GP 家族
  - 同时不直接放开 `tan / max / min / neg / inv` 这类更激进或冗余 primitive
- `pyoperon` 当前不再使用稀疏占位参数，而采用更接近 benchmark 文献的显式口径：
  - `population_size = 500`
  - `pool_size = 500`
  - `max_length = 50`
  - `tournament_size = 5`
  - `allowed_symbols = add,mul,aq,exp,log,sin,tanh,constant,variable`
  - `max_evaluations = 500000`
  - 目标是在 `E1` 中显式冻结 EA / tree-search 家族代表的搜索空间
- `dso` 当前不再使用 wrapper 中的弱默认值，而采用更接近官方 regression benchmark 的显式口径：
  - `function_set = add,sub,mul,div,sin,cos,exp,log`
  - `training.n_samples = 2000000`
  - `training.batch_size = 1000`
  - `training.epsilon = 0.05`
  - `policy_optimizer.learning_rate = 0.0005`
  - `policy_optimizer.entropy_weight = 0.03`
  - `policy_optimizer.entropy_gamma = 0.7`
  - 以及官方 regression 配置中的核心 `prior`
  - 目标是在 `E1` 中避免因 wrapper 过弱默认值而系统性低估 DSO
- `e2esr` 当前采用的口径是：
  - `max_input_points = 200`
  - `max_number_bags = -1`
  - `stop_refinement_after = 1`
  - `n_trees_to_refine = 100`
  - `rescale = true`
  - `force_cpu = true`
  - 目标是在 CPU 环境下固定 E2E 家族的推理预算与稳定性口径
- `qlattice` 当前采用的口径是：
  - `n_epochs = 100`
  - `kind = regression`
  - `criterion = bic`
  - `signif = 4`
  - `threads = 1`
  - 目标是在最终 10 算法比较中固定图搜索模型的选模与线程口径
- `imcts` 当前采用的口径是：
  - `ops = +,-,*,/,sin,cos,exp,log`
  - `max_depth = 6`
  - `K = 500`
  - `c = 4.0`
  - `gamma = 0.5`
  - `gp_rate = 0.2`
  - `mutation_rate = 0.1`
  - `exploration_rate = 0.2`
  - `max_single_arity_ops = 999`
  - `max_constants = 10`
  - `max_expressions = 2000000`
  - `optimization_method = LN_NELDERMEAD`
  - 目标是在最终 10 算法比较中对齐仓库自带 basic 配置
