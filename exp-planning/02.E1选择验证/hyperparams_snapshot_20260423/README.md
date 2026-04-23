# E1 七算法超参数快照

这份目录用于固定保存 `E1` 选择验证当前采用的 **7 个算法正式跑法超参数**。

## 用途

- 避免后续重新运行 `generate_e1_formal_assets.py` 时覆盖当前参数口径
- 为 `W1~W5` 分发、复查和论文复现实验提供一份稳定快照
- 明确当前 `E1` 采用的是一套**可执行的保守基线**，不是最终调优版

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
- `llmsr / drsr` 使用远端：
  - `/home/zhangziwen/projects/scientific-intelligent-modelling/llm.config`
- `tpsr` 当前参数是官方主配置对齐后的 benchmark 安全口径

