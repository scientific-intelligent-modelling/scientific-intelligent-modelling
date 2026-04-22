# E1 运行矩阵表

## 目的

`E1` 的任务不是直接做最终 leaderboard，而是先为下面 4 组 `100` 选择策略提供统一评估底座：

1. `Current-100`
2. `Gap-only-100`
3. `Quality-first-100`
4. `Metadata-diverse-100`

因此 `E1` 采用的是：

- `Candidate-200`
- `7` 个代表性算法
- `1` 个 seed

总任务量：

- `200 datasets × 7 algorithms × 1 seed = 1400`

## 7 个算法的家族覆盖

| 算法 | 主要代表的方法家族 | 为什么要放进 E1 |
|---|---|---|
| `gplearn` | 经典 GP | 给出传统符号回归 baseline |
| `pysr` | 现代演化式 SR | 当前最强主力之一，代表现代 EA |
| `pyoperon` | 树搜索 / EA | 补充现代树搜索式演化框架 |
| `llmsr` | 纯 LLM-based SR | 覆盖纯 LLM 提案式方法 |
| `drsr` | LLM-hybrid / DSR | 覆盖带搜索与回写的混合方法 |
| `dso` | 纯 RL-based SR | 显式覆盖强化学习方法族 |
| `tpsr` | Transformer + planning | 覆盖 planning / MCTS / transformer 路线 |

## 运行矩阵

| 算法 | 方法家族 | Conda 环境 | 当前远程验证状态 | `minute_0001.json` | 最终 `result.json` | 当前已知边界 | E1 建议优先级 | 建议机器池 |
|---|---|---|---|---|---|---|---|---|
| `gplearn` | 经典 GP | `sim_base` | 已在 `iaaccn22` 验证通过 | 已验证 | 已验证 | 无明显工程边界 | `P0` | `iaaccn22~25` |
| `pysr` | 现代演化式 SR | `sim_base` | 已完成多轮正式实验 | 不是重点，但结果链已验证 | 已验证 | 大量任务 `timed_out`，但可恢复，不应直接视作失败 | `P0` | `iaaccn22~25` |
| `pyoperon` | 树搜索 / EA | `sim_base` | 已修复 `SIGSEGV`，并同步到 `22~29` | 已验证 | 已验证 | 需要使用修复后的 wrapper；远端代码已同步 | `P0` | `iaaccn22~25` |
| `llmsr` | 纯 LLM-based SR | `sim_llm` | 已完成多轮正式实验 | 已验证 | 已验证 | 依赖在线 LLM 配置与 API key | `P0` | `iaaccn26~29` |
| `drsr` | LLM-hybrid / DSR | `sim_llm` | 已在 `iaaccn22` 验证通过 | 已验证 | 已验证 | 长任务 timeout 收口不够稳，但中间快照与最终落盘是通的 | `P1` | `iaaccn26~29` |
| `dso` | 纯 RL-based SR | `sim_dso` | 已修好周期快照链；`sim_dso` 已同步到 `22~28` | 已验证 | 已验证 | `iaaccn29` 的 `sim_dso` 尚未单独确认 | `P1` | `iaaccn22~28` |
| `tpsr` | Transformer + planning | `sim_tpsr` | 已修复大样本 OOM；默认值已对齐官方，并在 `iaaccn22` 验证 | 已验证 | 已验证 | 最新“官方对齐默认配置”目前只在 `iaaccn22` 做过 smoke；需再做多机确认 | `P1` | 优先 `iaaccn22~25` |

## E1 的实际推荐分层

### 第一批，先跑稳的 `P0`

- `gplearn`
- `pysr`
- `pyoperon`
- `llmsr`

特点：

- 工程链路最稳
- 已有正式实验沉淀
- 适合作为 `E1` 的第一批底座

### 第二批，再补 `P1`

- `drsr`
- `dso`
- `tpsr`

特点：

- 方法家族上必须保留
- 但当前工程边界略多于 `P0`
- 更适合在 `E1` 的第二批推进，或者先做一轮小切片 smoke 再铺满 `Candidate-200`

## 我对 E1 的建议跑法

### 方案 A：一次铺开 1400 个任务

前提：

- `sim_dso / sim_tpsr / sim_llm / sim_base` 在目标机器上都已确认可用
- `drsr / dso / tpsr` 的多机 smoke 先通过

优点：

- 最快得到完整 selection ablation 底座

风险：

- 如果 `P1` 三个算法里任意一个在多机上还没完全稳，会拖慢整轮

### 方案 B：分两批

#### 批次 1：`P0`

- `200 × 4 × 1 = 800 tasks`

#### 批次 2：`P1`

- `200 × 3 × 1 = 600 tasks`

优点：

- 更稳
- 更容易定位环境或运行时问题

我更推荐：

- **先用方案 B**

因为 `E1` 的目标不是抢最快，而是要保证这 `1400` 个任务能形成干净、可复用的底座。

## E1 前的最小检查清单

在正式分发 `E1` 之前，建议先确认：

1. `Clean-Master-100` 已冻结，不再变动；
2. `Candidate-200` 的切片 CSV 已生成；
3. `iaaccn22~25` 的 `sim_base / sim_tpsr` 已就绪；
4. `iaaccn26~29` 的 `sim_llm` 已就绪，且 `drsr/llmsr` 的 runtime config 可用；
5. `iaaccn22~28` 的 `sim_dso` 已就绪；
6. `dso / tpsr / drsr` 再各做一轮多机 smoke，确认分钟级快照与最终落盘都稳定。

## 一句话结论

`E1` 的这 `7` 个算法不是随便拼出来的，而是为了在成本可控的情况下，尽量完整覆盖：

- GP
- 现代 EA
- 纯 RL
- LLM
- LLM-hybrid
- planning / transformer

如果后续 reviewer 问：

> “你这个 100 的筛选是不是只对 `PySR / LLM-SR` 两个 selector 敏感？”

那么 `E1` 的这张矩阵，就是最直接的防守依据。
