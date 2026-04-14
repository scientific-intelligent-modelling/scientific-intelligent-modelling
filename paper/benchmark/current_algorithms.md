# 当前算法整理

本文档按当前仓库实际注册状态整理，不按历史目录或旧笔记估算。

## 总览

- 当前 `toolbox_config.json` 正式注册了 **10 个**算法后端。
- 当前 `algorithms/` 目录下还能看到一个额外的 `operon_wrapper/`。
  - 它不是第 11 个独立工具。
  - 当前正式注册名称是 `pyoperon`。
  - `pyoperon_wrapper` 只是对 `operon_wrapper.OperonRegressor` 的别名导出。
- 当前仓库中：
  - **10 个注册工具**全部有对应 wrapper 目录。
  - **10 个注册工具**全部有对应 `check/check_<tool>.py` 验收脚本。
  - 测试覆盖最重的是 `drsr`，其次是 `pysr`、`llmsr`。
- 当前论文试点实验里真正作为稳定切片汇报的是：
  - `pysr`
  - `llmsr`
  - `drsr`

## 正式注册算法

| 工具名 | Wrapper目录 | Regressor类 | 环境 | Check脚本 | 测试命中数 | 备注 |
|---|---|---|---|---|---:|---|
| `gplearn` | `gplearn_wrapper` | `GPLearnRegressor` | `sim_base` | `check_gplearn.py` | 3 | 经典基线，环境最轻 |
| `pysr` | `pysr_wrapper` | `PySRRegressor` | `sim_base` | `check_pysr.py` | 6 | 当前试点实验主力之一 |
| `pyoperon` | `pyoperon_wrapper` | `OperonRegressor` | `sim_base` | `check_pyoperon.py` | 1 | 正式工具名是 `pyoperon`，实现实际来自 `operon_wrapper` |
| `llmsr` | `llmsr_wrapper` | `LLMSRRegressor` | `sim_llm` | `check_llmsr.py` | 6 | 有离线/在线两种验收路径 |
| `dso` | `dso_wrapper` | `DSORegressor` | `sim_dso` | `check_dso.py` | 3 | 老依赖栈，环境相对脆弱 |
| `tpsr` | `tpsr_wrapper` | `TPSRRegressor` | `sim_tpsr` | `check_tpsr.py` | 3 | Transformer系，环境较重 |
| `e2esr` | `e2esr_wrapper` | `E2ESRRegressor` | `sim_e2esr` | `check_e2esr.py` | 2 | 依赖模型文件下载 |
| `QLattice` | `QLattice_wrapper` | `QLatticeRegressor` | `sim_qLattice` | `check_qlattice.py` | 3 | 工具名大小写敏感 |
| `iMCTS` | `iMCTS_wrapper` | `iMCTSRegressor` | `sim_iMCTS` | `check_imcts.py` | 1 | 工具名大小写敏感 |
| `drsr` | `drsr_wrapper` | `DRSRRegressor` | `sim_llm` | `check_drsr.py` | 14 | 当前测试最密集，试点实验主力之一 |

## 维护信号

### 强维护

- `drsr`
  - `tests/` 中相关测试最多。
  - 最近还做了 prompt 对齐、背景、接口、匿名化处理。
- `llmsr`
  - 有在线/离线 check。
  - 有预算探针脚本与远端消融聚合脚本。
- `pysr`
  - 有默认参数、线程环境、基准运行脚本。
  - 当前试点 benchmark 已实际产出结果。

### 中等维护

- `gplearn`
- `pyoperon`
- `tpsr`
- `dso`
- `QLattice`
- `e2esr`
- `iMCTS`

这些工具已经接入统一执行路径，也有验收脚本，但当前仓库中的实验沉淀和测试密度明显低于 `drsr / llmsr / pysr`。

## 当前风险与注意点

### 1. `pyoperon` 与 `operon_wrapper` 命名容易混淆

- 外部统一入口应使用 `tool_name=\"pyoperon\"`。
- `operon_wrapper/` 是实现承载目录，不应再单独当成一个算法统计。

### 2. 工具名大小写不统一

- `QLattice`
- `iMCTS`

这两个工具名带大写，写实验脚本或配置时需要保持一致。

### 3. 环境负担差异明显

- 轻环境：
  - `gplearn`
  - `pysr`
  - `pyoperon`
- 中/重环境：
  - `llmsr`
  - `drsr`
  - `dso`
  - `tpsr`
  - `e2esr`
  - `iMCTS`

如果后面要做大规模 benchmark，优先级上应先保证轻环境工具稳定批跑，再决定是否把重环境方法纳入全量评测。

## 适合下一步 benchmark 的分层

### 第一层：当前最适合先跑

- `pysr`
- `llmsr`
- `drsr`
- `gplearn`
- `pyoperon`

理由：

- 执行路径最清楚。
- 环境与工具行为已经有一定仓库内验证。
- 其中前三个已经有试点 benchmark 结果沉淀。

### 第二层：可逐步扩展

- `QLattice`
- `tpsr`
- `dso`
- `e2esr`
- `iMCTS`

理由：

- 已接入，但更适合先做单工具 check 或小规模抽样评测。
- 不建议一开始就进入全量 600+ 数据集 benchmark。

## 结论

- 现在仓库里应按 **10 个正式算法后端** 对外表述。
- `operon_wrapper` 应视为 `pyoperon` 的实现目录，不应单独计数。
- 如果你后面要把算法再整理成 benchmark 计划，最合理的排序是：
  - 先 `pysr / llmsr / drsr`
  - 再 `gplearn / pyoperon`
  - 最后再考虑 `QLattice / tpsr / dso / e2esr / iMCTS`
