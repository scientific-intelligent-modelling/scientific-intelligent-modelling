# E1 Wave Smoke Report (2026-04-23)

## 目的

在正式分发 `E1 = Candidate-200 × 7 algorithms × 1 seed` 之前，先按 wave 方案做一轮远程 smoke，验证：

1. 算法能否在对应机器池上真正启动
2. `minute_0001.json` 是否按预期落盘
3. 最终 `result.json` 是否按预期落盘
4. 是否存在会阻塞大规模分发的环境或收口问题

## smoke 设置

- `seed = 1314`
- `timeout_in_seconds = 150`
- `progress_snapshot_interval_seconds = 60`
- 统一 runner：`/tmp/run_benchmark_one.py`
- 统一输出根：
  - `experiments/e1_wave_smoke_v2_20260423`

数据集：

- CPU/搜索类算法：
  - `llm-srbench/lsrtransform/III.13.18_0_0`
- LLM 类算法：
  - `nguyen/Nguyen-11`

## 结果总览

| Wave | 算法 | 机器池 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| `W1` | `gplearn` | `iaaccn22~25` | `4/4 ok` | 可直接分发 |
| `W1` | `llmsr` | `iaaccn26~29` | `4/4 ok`，且 `minute_0001.json` 正常 | 可直接分发 |
| `W2` | `pyoperon` | `iaaccn22~25` | `4/4 ok` | 可直接分发 |
| `W2` | `drsr` | `iaaccn26~29` | `4/4` 启动正常，`minute_0001.json` 正常；最终收口仍受已知 timeout 行为影响 | 可分发，但需接受已知收口边界 |
| `W3` | `pysr` | `iaaccn22~25` | `4/4 timed_out`，但都有 `minute_0001.json` 和完整输出 | 可直接分发 |
| `W4` | `dso` | `iaaccn22~28` | `7/7 ok` | 可直接分发 |
| `W5` | `tpsr` | `iaaccn22`（保守方案） | `timed_out`，`minute_0001.json` 正常，最终结果正常 | 当前可按单机保守方案分发 |

## 分 wave 细节

### W1

#### `gplearn` on `iaaccn22~25`

- `4/4` 都自然结束
- 全部：
  - `status = ok`
  - `equation = true`
  - `canonical_artifact = true`
  - `valid / id_test / ood_test = true`
- 运行时长约：
  - `11.5s ~ 25.3s`

说明：
- 这批太快，`minute_0001.json` 不会出现
- 这不是问题，而是任务天然不到 60 秒

#### `llmsr` on `iaaccn26~29`

- `4/4` 都自然结束
- 全部：
  - `outer minute_0001.json = true`
  - `inner minute_0001.json = true`
  - `status = ok`
  - `equation / artifact / valid / id / ood = true`
- 运行时长约：
  - `155.3s ~ 155.6s`

结论：
- `W1` 已完整通过

### W2

#### `pyoperon` on `iaaccn22~25`

- `4/4` 都自然结束
- 全部：
  - `status = ok`
  - `equation / artifact / valid / id / ood = true`
- 运行时长约：
  - `7.4s ~ 9.8s`

#### `drsr` on `iaaccn26~29`

- `4/4` 都成功启动
- 全部：
  - `root_exists = true`
  - `outer minute_0001.json = true`
- 但这轮 smoke 中：
  - 截止检查时尚未写出最终 `result.json`

解释：
- 这与 `drsr` 已知的长任务 timeout 收口边界一致
- 该问题此前已确认，不再作为这轮 smoke 的 blocker

结论：
- `W2` 的工程链路是通的
- 若纳入正式 E1，需要接受 `drsr` 已知收口边界

### W3

#### `pysr` on `iaaccn22~25`

- `4/4` 都成功运行至预算结束
- 全部：
  - `outer minute_0001.json = true`
  - `result.json = true`
  - `status = timed_out`
  - `equation / artifact / valid / id / ood = true`
- 运行时长约：
  - `153.1s ~ 158.4s`

结论：
- `W3` 已通过
- `pysr` 仍按其既有语义解释：
  - `timed_out != 失败`

### W4

#### `dso` on `iaaccn22~28`

- `7/7` 都自然结束
- 全部：
  - `status = ok`
  - `equation / artifact / valid / id / ood = true`
- 运行时长约：
  - `11.8s ~ 25.1s`

说明：
- 这批同样没有 `minute_0001.json`
- 原因是任务天然不到 60 秒，不是快照链有问题

结论：
- `W4` 已完整通过

### W5

#### `tpsr` on `iaaccn22`（保守方案）

- 当前只验证了保守单机方案
- 结果：
  - `outer minute_0001.json = true`
  - `result.json = true`
  - `status = timed_out`
  - `equation = true`
  - `canonical_artifact = true`
  - `valid / id / ood = false`
- 运行时长约：
  - `154.3s`

解释：
- 这条 smoke 证明：
  - 默认配置下 `TPSR` 能稳定活过 60 秒
  - 不再复现之前的 OOM / `SIGKILL`
  - timeout 后能写最终 `result.json`
- 但在 `150s` 短预算下，指标恢复不保证完整

结论：
- `W5` 当前可以按保守单机方案使用
- 若要验证推荐版 `iaaccn22~25` 四机对称分发，必须先把 `sim_tpsr` 补到 `iaaccn23~25`

## 总结判断

### 当前可以直接推进的 wave

- `W1`
- `W3`
- `W4`

### 可以推进，但要带已知边界说明的 wave

- `W2`
  - `drsr` 收口行为已有已知边界
- `W5`
  - 当前只验证到单机保守方案

## 对 E1 的直接建议

1. `W1 / W3 / W4` 可以直接进入正式分发准备
2. `W2` 可以保留，但文档中要明确：
   - `drsr` 已知 timeout 收口边界不作为 blocker
3. `W5` 若要进入推荐版分发方案：
   - 先补 `sim_tpsr` 到 `iaaccn23~25`
   - 再补一轮四机 smoke
