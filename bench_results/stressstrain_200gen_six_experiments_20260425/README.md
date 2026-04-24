# stressstrain 200 代六组实验归档

该目录集中归档 `stressstrain` 数据集上两轮 200 代重跑结果，每轮包含三个算法：

- `rerun200_llm_drsr_20260405_150543`：`drsr`、`llmsr`、`pysr`
- `rerun200_llm_drsr_20260406_111429`：`drsr`、`llmsr`、`pysr`

目录结构：

- `summary.csv`：六组实验的紧凑指标汇总表。
- `experiment_config.json`：完整归档清单，记录源路径和实际生效参数。
- `shared/dataset_config.json`：统一数据集契约。
- `shared/llm.config.redacted.json`：已脱敏的共享 LLM 配置。
- `environment_and_best_formulas.md`：面向阅读的环境配置、依赖版本、关键参数和最优公式记录。
- `environment_and_best_formulas.json`：机器可读的环境配置、依赖版本、关键参数和最优公式记录。
- `runs/<batch>/<tool>/result.json`：复制得到的最终实验结果。
- `runs/<batch>/<tool>/config.json`：从 `result.json` 和 wrapper 默认值抽取的单实验复现配置。
- `runs/<batch>/<tool>/artifacts/`：可用的关键实验产物，例如 specs、progress、top samples 和 PySR `hall_of_fame`。
- `runs/<batch>/<tool>/logs/`：`drsr` / `llmsr` 的可用 `stressstrain` 运行日志。

说明：

- 原始 `llm.config` 包含凭据，因此本归档中的 `api_key` 已有意替换为 `REDACTED`。
- 该目录没有完整复制所有原始实验目录；每个 `config.json` 都记录了对应源路径。
- PySR 使用当前 wrapper 默认参数并设置 `niterations=200` 重新运行。
