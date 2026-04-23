# E1 七算法超参数快照

这份目录用于固定保存 `E1` 选择验证当前采用的 **7 个算法正式跑法超参数**。

## 用途

- 避免后续重新运行 `generate_e1_formal_assets.py` 时覆盖当前参数口径
- 为 `W1~W5` 分发、复查和论文复现实验提供一份稳定快照
- 明确当前 `E1` 采用的是一套**可执行的工程基线**，不是最终调优版

## 当前包含的算法

- `gplearn`
- `pysr`
- `pyoperon`
- `llmsr`
- `drsr`
- `dso`
- `tpsr`

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

## 参数来源

这些文件是从：

- `exp-planning/02.E1选择验证/generated/params/*.json`

复制出来的快照版本，对应当前 `E1` 资产生成器口径。

## 备注

- 当前默认 `timeout_in_seconds = 3600`
- 当前默认 `progress_snapshot_interval_seconds = 60`
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
