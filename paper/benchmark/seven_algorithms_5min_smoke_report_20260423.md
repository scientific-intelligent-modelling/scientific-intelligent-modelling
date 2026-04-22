# 七算法 5 分钟 Smoke 报告（2026-04-23）

## 目的

对 `E1` 使用的 7 个算法各跑一条 **300 秒 smoke**，验证：

1. 能否在远程 benchmark runner 中正常启动
2. 是否能写出 `progress/minute_0001.json`
3. 是否能写出最终 `result.json`
4. 5 分钟预算下 `equation / canonical_artifact / valid / id_test / ood_test` 是否完整

## smoke 设置

- `seed = 1314`
- `timeout_in_seconds = 300`
- `progress_snapshot_interval_seconds = 60`

数据集：

- CPU/搜索类算法：
  - `llm-srbench/lsrtransform/III.13.18_0_0`
- LLM 类算法：
  - `nguyen/Nguyen-11`

机器分配：

- `gplearn` → `iaaccn22`
- `pysr` → `iaaccn23`
- `pyoperon` → `iaaccn24`
- `tpsr` → `iaaccn25`
- `llmsr` → `iaaccn26`
- `drsr` → `iaaccn27`
- `dso` → `iaaccn28`

## 总表

| 算法 | 主机 | minute_0001 | result.json | status | seconds | equation | artifact | valid | id | ood | 结论 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `gplearn` | `iaaccn22` | ✅ | ✅ | `timed_out` | `302.805` | ✅ | ✅ | ✅ | ✅ | ✅ | 通过 |
| `pysr` | `iaaccn23` | ✅ | ✅ | `timed_out` | `303.132` | ✅ | ✅ | ✅ | ✅ | ✅ | 通过 |
| `pyoperon` | `iaaccn24` | ✅ | ✅ | `timed_out` | `302.753` | ✅ | ✅ | ✅ | ✅ | ✅ | 通过 |
| `tpsr` | `iaaccn25` | ✅ | ✅ | `timed_out` | `306.548` | ✅ | ✅ | ❌ | ❌ | ❌ | 链路通过，指标未补齐 |
| `llmsr` | `iaaccn26` | ✅ | ✅ | `ok` | `305.475` | ✅ | ✅ | ✅ | ✅ | ✅ | 通过 |
| `drsr` | `iaaccn27` | ✅ | ✅ | `ok` | `94.510` | ✅ | ✅ | ✅ | ✅ | ✅ | 通过 |
| `dso` | `iaaccn28` | ❌ | ✅ | `ok` | `11.921` | ✅ | ✅ | ✅ | ✅ | ✅ | 通过，但任务太快未跨过 60 秒 |

## 逐个算法结论

### `gplearn`

- 这轮通过加大 `population_size / generations`，成功跨过 60 秒
- `minute_0001.json` 正常
- 最终 `timed_out` 收口后：
  - `equation / artifact / valid / id / ood` 全在

结论：
- 可以按 5 分钟预算进入 benchmark

### `pysr`

- `minute_0001.json` 正常
- 最终 `timed_out` 收口后：
  - `equation / artifact / valid / id / ood` 全在

结论：
- 5 分钟预算下工程链路稳定
- 仍按 `timed_out != 失败` 解释

### `pyoperon`

- 这轮通过加大 `population_size / generations`，成功跨过 60 秒
- `minute_0001.json` 正常
- 最终 `timed_out` 收口后：
  - `equation / artifact / valid / id / ood` 全在

结论：
- 可以按 5 分钟预算进入 benchmark

### `tpsr`

- `minute_0001.json` 正常
- 最终 `result.json` 正常
- `equation / artifact` 已恢复
- 但 `valid / id / ood` 仍未补齐

结论：
- 5 分钟预算已经证明：
  - 链路通
  - 快照通
  - 最终落盘通
- 但 **指标补全还没有被 300 秒 smoke 证明**

### `llmsr`

- `minute_0001.json` 正常
- 最终 `status = ok`
- `equation / artifact / valid / id / ood` 全在

结论：
- 5 分钟预算通过

### `drsr`

- `minute_0001.json` 正常
- 这轮参数收紧后，最终 `status = ok`
- `equation / artifact / valid / id / ood` 全在

结论：
- 这轮 5 分钟 smoke 已通过
- 但这不推翻此前的已知边界：更长预算下 timeout 收口仍需单独观察

### `dso`

- 最终 `status = ok`
- `equation / artifact / valid / id / ood` 全在
- 本轮没有 `minute_0001.json`

原因：
- 任务仅耗时 `11.921s`
- 天然没跨过 60 秒

结论：
- 5 分钟预算下最终落盘无问题
- 本轮不能据此判断 `minute_0001.json` 有问题，只能说明任务太快

## 总结判断

### 已完整通过 5 分钟 smoke 的

- `gplearn`
- `pysr`
- `pyoperon`
- `llmsr`
- `drsr`

这 5 个算法都已同时验证：

- `minute_0001.json`
- `result.json`
- 完整 `valid / id / ood`

### 已通过“最终落盘 + 指标完整”，但本轮未跨过 60 秒的

- `dso`

### 已通过“链路与落盘”，但 300 秒下指标仍未补齐的

- `tpsr`

## 建议

1. `gplearn / pysr / pyoperon / llmsr / drsr` 可直接进入正式 `E1`
2. `dso` 可进入正式 `E1`
   - 无需因为这轮没 `minute_0001.json` 而阻塞
   - 原因只是 smoke 任务太快
3. `tpsr` 若要在正式 `E1` 中使用，建议再补一条：
   - `600s` smoke
   - 专门验证 `valid / id / ood` 是否能稳定补齐
