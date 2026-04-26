# Deprecated Notice

This plan is preserved only as historical reference. It predates the E1
7-algorithm validation workflow and should not be used as the current benchmark
execution plan.

Current benchmark flow is:

- E1: 7 algorithms × 200 candidates × 1 seed
- Clean-Master-100 construction
- Core-50 / Reserve-50 split
- 7-algorithm rank-fidelity pilot
- Follow-up: add remaining algorithms and run frozen Core-50 formal leaderboard

# 200 候选种子下游正式实验执行方案（v1）

## 0. 目标与范围

本方案面向：

- 候选集：`/tmp/candidate_seeds_200_v3.json`
- 数据集数：`200`
- 随机种子：`520`、`521`
- 方法：`pysr`、`llmsr`

总任务数：

- `200 datasets × 2 seeds × 2 methods = 800 tasks`

机器资源：

- `iaaccn22~25`：执行 `pysr`
- `iaaccn26~29`：执行 `llmsr`

本方案重点是：

1. 把 800 个任务稳定铺到 8 台机器
2. 保持方法侧资源约束与 probe 阶段一致
3. 保证结果汇总时不受“同 basename 覆盖”问题影响


## 1. 任务分配方案

### 1.1 总体分配

保持方法分离，不混跑：

- `iaaccn22~25`：只跑 `pysr`
- `iaaccn26~29`：只跑 `llmsr`

原因：

1. 这与当前环境准备和 probe 阶段完全一致
2. `pysr` 依赖 `sim_base + pyjuliapkg_pysr`
3. `llmsr` 依赖 `sim_llm + DeepInfra`
4. 避免同机混合 CPU/Julia 与 API worker，减少诊断复杂度

### 1.2 数据集切片

对 `200` 个候选种子按固定顺序切成 4 片，每片 `50` 个数据集：

1. `slice_01`：第 `1~50`
2. `slice_02`：第 `51~100`
3. `slice_03`：第 `101~150`
4. `slice_04`：第 `151~200`

建议同一片在两种方法上保持一致，便于横向对照：

- `iaaccn22` / `iaaccn26`：`slice_01`
- `iaaccn23` / `iaaccn27`：`slice_02`
- `iaaccn24` / `iaaccn28`：`slice_03`
- `iaaccn25` / `iaaccn29`：`slice_04`

### 1.3 每台机器任务数

每台机器：

- `50 datasets × 2 seeds × 1 method = 100 tasks`

按方法汇总：

- `pysr`：`4 machines × 100 = 400 tasks`
- `llmsr`：`4 machines × 100 = 400 tasks`

### 1.4 每台机器的任务列表

建议每台机器使用**同一个切片 CSV**，分别起两个 seed 的 controller。

例如：

- `iaaccn22`
  - `pysr + seed520 + slice_01`
  - `pysr + seed521 + slice_01`

- `iaaccn23`
  - `pysr + seed520 + slice_02`
  - `pysr + seed521 + slice_02`

- `iaaccn24`
  - `pysr + seed520 + slice_03`
  - `pysr + seed521 + slice_03`

- `iaaccn25`
  - `pysr + seed520 + slice_04`
  - `pysr + seed521 + slice_04`

- `iaaccn26`
  - `llmsr + seed520 + slice_01`
  - `llmsr + seed521 + slice_01`

- `iaaccn27`
  - `llmsr + seed520 + slice_02`
  - `llmsr + seed521 + slice_02`

- `iaaccn28`
  - `llmsr + seed520 + slice_03`
  - `llmsr + seed521 + slice_03`

- `iaaccn29`
  - `llmsr + seed520 + slice_04`
  - `llmsr + seed521 + slice_04`


## 2. 实验配置

### 2.1 batch name 命名规范

建议统一用一个顶层批次名，再按方法、seed、机器分层：

```text
formal200_v1_<YYYYMMDD-HHMMSS>
```

例如：

```text
formal200_v1_20260417-020000
```

输出根目录建议：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal200_v1_20260417-020000
```

内部结构建议：

```text
experiments/formal200_v1_<ts>/
  pysr/
    seed520/
      iaaccn22/
      iaaccn23/
      iaaccn24/
      iaaccn25/
    seed521/
      iaaccn22/
      iaaccn23/
      iaaccn24/
      iaaccn25/
  llmsr/
    seed520/
      iaaccn26/
      iaaccn27/
      iaaccn28/
      iaaccn29/
    seed521/
      iaaccn26/
      iaaccn27/
      iaaccn28/
      iaaccn29/
```

### 2.2 每个任务的参数

#### `pysr`

延续当前 probe launcher 的参数体系，建议在正式实验中使用相同搜索参数，仅调整预算。

建议：

```python
{
  "timeout_in_seconds": 7200,
  "niterations": 10_000_000,
  "population_size": 64,
  "populations": 8,
  "ncycles_per_iteration": 500,
  "maxsize": 30,
  "maxdepth": 10,
  "parsimony": 1e-3,
  "binary_operators": ["+", "-", "*", "/"],
  "unary_operators": ["square", "cube", "exp", "log", "sin", "cos"],
  "constraints": {"/": (-1, 9), "square": 9, "cube": 9, "exp": 7, "log": 7, "sin": 9, "cos": 9},
  "nested_constraints": {
    "exp": {"exp": 0, "log": 1},
    "log": {"exp": 0, "log": 0},
    "square": {"square": 1, "cube": 1, "exp": 0, "log": 0},
    "cube": {"square": 1, "cube": 1, "exp": 0, "log": 0}
  },
  "complexity_of_operators": {"/": 2, "square": 2, "cube": 3, "sin": 2, "cos": 2, "exp": 3, "log": 3},
  "complexity_of_constants": 2,
  "complexity_of_variables": 1,
  "precision": 32,
  "deterministic": true,
  "parallelism": "serial",
  "model_selection": "best",
  "progress": true,
  "verbosity": 1,
  "procs": 1
}
```

说明：

1. `seed` 由框架统一传入，wrapper 会映射到 `random_state`
2. `parallelism="serial"` 必须保留，否则 `deterministic=True` 会报错

#### `llmsr`

延续当前 probe 配置：

```python
{
  "timeout_in_seconds": 7200,
  "niterations": 100000,
  "samples_per_iteration": 4,
  "max_params": 10,
  "inject_prompt_semantics": false,
  "background": "This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.",
  "persist_all_samples": false
}
```

LLM 配置：

```json
{
  "model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
  "base_url": "https://api.deepinfra.com/v1/openai",
  "max_tokens": 1024,
  "temperature": 0.6,
  "top_p": 0.3
}
```

并通过环境变量注入：

```text
DEEPINFRA_API_KEY
```

### 2.3 `dataset_dir / seed / method / timeout` 的任务格式

建议生成一个统一任务清单，字段如下：

```csv
global_index,dataset_dir,dataset_name,method,seed,machine,slice_id
```

例如：

```csv
1,sim-datasets-data/srsd/srsd-feynman_medium_dummy/feynman-iii.15.12,feynman-iii.15.12,pysr,520,iaaccn22,slice_01
2,sim-datasets-data/srsd/srsd-feynman_medium_dummy/feynman-iii.15.12,feynman-iii.15.12,pysr,521,iaaccn22,slice_01
...
```

### 2.4 spec 文件怎么生成

#### `pysr`

- 不需要单独 spec 文件
- 参数通过 launcher 直接传给 `run_benchmark_task(...)`

#### `llmsr`

- 由当前框架在运行时自动生成 spec
- 来源是：
  - 匿名变量 `x0/x1/.../y`
  - 中性 `background`
  - `inject_prompt_semantics=False`

也就是说，正式实验会保持：

- 不注入变量语义描述
- 不注入 metadata 语义背景

建议保留每个实验目录中的：

- `spec_dynamic.txt`
- `best_history/`
- `samples/`

作为审计材料。


## 3. 启动脚本

### 3.1 预处理脚本

建议新增一个预处理步骤，从 `/tmp/candidate_seeds_200_v3.json` 生成：

1. `candidate_datasets.csv`
   - 只含 `200` 个候选数据集
2. `slices/pysr/iaaccn22~25.csv`
3. `slices/llmsr/iaaccn26~29.csv`

切片规则：

- 保持与候选 JSON 顺序一致
- 每片 `50` 个 dataset

### 3.2 每台机器的启动形式

建议每台机器起 **2 个 tmux session**，分别对应两个 seed。

#### `pysr`

每台机器总预算不变，拆成两个 controller 并发跑：

- `seed520`: `32 workers`
- `seed521`: `32 workers`

总计：

- 每机 `64 workers`

#### `llmsr`

保持 DeepInfra 总并发 `200` 不变，按 seed 平分：

- `seed520`: `25 workers`
- `seed521`: `25 workers`

总计：

- 每机 `50 workers`
- 4 台总共 `200`

### 3.3 推荐启动命令模板

#### `pysr`（以 `iaaccn22` 为例）

```bash
tmux new-session -d -s formal_pysr_520_22 \\
  "cd /home/zhangziwen/workplace/scientific-intelligent-modelling && \\
   PYTHONPATH=. conda run -n sim_base python exp-planning/01.双探针实验/launch_pysr_probe.py run \\
   --slice-csv exp-planning/01.双探针实验/slices_formal/pysr/iaaccn22.csv \\
   --output-root /home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal200_v1_<ts>/pysr/seed520/iaaccn22 \\
   --seed 520 \\
   --workers 32"

tmux new-session -d -s formal_pysr_521_22 \\
  "cd /home/zhangziwen/workplace/scientific-intelligent-modelling && \\
   PYTHONPATH=. conda run -n sim_base python exp-planning/01.双探针实验/launch_pysr_probe.py run \\
   --slice-csv exp-planning/01.双探针实验/slices_formal/pysr/iaaccn22.csv \\
   --output-root /home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal200_v1_<ts>/pysr/seed521/iaaccn22 \\
   --seed 521 \\
   --workers 32"
```

#### `llmsr`（以 `iaaccn26` 为例）

```bash
tmux new-session -d -s formal_llmsr_520_26 \\
  "cd /home/zhangziwen/workplace/scientific-intelligent-modelling && \\
   export DEEPINFRA_API_KEY=... && \\
   PYTHONPATH=. conda run -n sim_llm python exp-planning/01.双探针实验/launch_llmsr_probe.py run \\
   --slice-csv exp-planning/01.双探针实验/slices_formal/llmsr/iaaccn26.csv \\
   --output-root /home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal200_v1_<ts>/llmsr/seed520/iaaccn26 \\
   --llm-config-path /home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal200_v1_<ts>/llmsr/seed520/iaaccn26/llm.config \\
   --seed 520 \\
   --workers 25"

tmux new-session -d -s formal_llmsr_521_26 \\
  "cd /home/zhangziwen/workplace/scientific-intelligent-modelling && \\
   export DEEPINFRA_API_KEY=... && \\
   PYTHONPATH=. conda run -n sim_llm python exp-planning/01.双探针实验/launch_llmsr_probe.py run \\
   --slice-csv exp-planning/01.双探针实验/slices_formal/llmsr/iaaccn26.csv \\
   --output-root /home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal200_v1_<ts>/llmsr/seed521/iaaccn26 \\
   --llm-config-path /home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal200_v1_<ts>/llmsr/seed521/iaaccn26/llm.config \\
   --seed 521 \\
   --workers 25"
```

### 3.4 一次性在 8 台机器上启动

推荐做法：

1. 本地先生成：
   - 正式切片 CSV
   - 各机器启动脚本
2. 同步到 `iaaccn22`
3. 由 `iaaccn22` 再向 `23~29` 内网分发
4. 由 `iaaccn22` 统一 `ssh 10.10.100.xx` 执行

不要本地直接 fan-out 到 8 台，原因：

- 本地到远端链路不稳
- `22` 做中继更稳


## 4. 监控进度

### 4.1 机器级监控

每台机器主要看：

- `tmux ls`
- `pgrep -af 'launch_(pysr|llmsr)_probe.py run|run-task'`
- `__launcher__/task_status.jsonl`

最小巡检命令：

```bash
wc -l <output_root>/__launcher__/task_status.jsonl
tail -n 5 <output_root>/__launcher__/task_status.jsonl
```

### 4.2 批次级监控

建议沿用 probe 阶段的监控思路，做一个本地巡检脚本，每 `5~10` 分钟聚合：

- `completed`
- `ok`
- `timed_out`
- `error`
- `running`

并分别统计：

- `pysr / llmsr`
- `seed520 / seed521`
- `machine`

### 4.3 结果收集

**非常重要：不要只扫外层 `.../<dataset>/result.json`。**

原因：

- 同一个 slice 内如果出现同 basename 的数据集变体
- 外层 `result.json` 目录会互相覆盖

**正式汇总必须以：**

- `__launcher__/task_status.jsonl`

为主索引，再读取每条任务对应的：

- `experiment_dir/result.json`

这点是当前方案里最关键的结果收集规则。


## 5. 预估时间

### 5.1 单任务 timeout 建议

#### 推荐值：`7200s`

理由：

1. `llmsr` 在 probe 的 `3600s` 下已经大多数能自然结束
2. `pysr` 在 probe 的 `3600s` 下全部 `timed_out`，虽然可恢复，但正式实验最好少依赖超时回填
3. `7200s` 更接近正式实验语义：
   - 比 `3600s` 明显更稳
   - 对 `pysr` 来说能减少“纯超时壳结果”的占比
   - 对 `llmsr` 来说也能降低少数 hard case 被 1h 截断的概率

### 5.2 总体预计耗时

#### `pysr`

每台机器：

- `50 datasets / seed`
- `32 workers / seed`
- 两个 seed 同时跑

每个 seed 大约是两波：

- `32 + 18`

若 `timeout=7200s`：

- 每个 seed 约 `2 waves × 2h ≈ 4h`
- 两个 seed 并行，所以单机 wall-clock 仍约 `4~4.5h`

#### `llmsr`

每台机器：

- `50 datasets / seed`
- `25 workers / seed`
- 两个 seed 同时跑

每个 seed 也是两波：

- `25 + 25`

若 `timeout=7200s`：

- 单机 wall-clock 约 `4~4.5h`

#### 全局

因为 `pysr` 和 `llmsr` 在不同机器组上并行：

- **整个 800 任务批次预计 wall-clock：`4.5 ~ 5.5h`**

如果改回 `3600s`：

- 预计可压缩到 `2 ~ 3h`

但当前正式实验已明确改为 `7200s`，不建议再降回 `3600s`。


## 6. 注意事项

### 6.1 与 probe 的区别

主要区别有 4 个：

1. 候选集从 `664` 缩到 `200`
2. 每个数据集从 `1 seed` 变成 `2 seeds`
3. 正式实验把 timeout 从 probe 的 `3600s` 提到 `7200s`
4. 汇总必须使用 `task_status.jsonl + experiment_dir/result.json`，不能再偷懒扫外层目录

### 6.2 需要修改/新增的配置或脚本

建议新增：

1. `build_formal_candidate_slices.py`
   - 从 `/tmp/candidate_seeds_200_v3.json` 生成 4 片正式切片

2. `launch_formal_batches.sh`
   - 统一生成 8 台机器的启动命令

3. `collect_formal_results.py`
   - 以 `task_status.jsonl` 为主索引汇总结果

4. `recover_pysr_formal_timeout.py`
   - 对 `pysr timed_out` 做事后回填

5. `recover_pysr_stable_metrics.py`
   - 对“有方程但没指标”的样本沿 `hall_of_fame.csv/.bak` 找第一条数值稳定候选

### 6.3 结果回填策略

#### `pysr`

正式实验结束后建议固定跑两步：

1. **超时恢复回填**
   - 从 `hall_of_fame.csv / .bak` 恢复 `equation / canonical_artifact`

2. **稳定候选回填**
   - 对“有方程但指标为空”的样本
   - 沿 `hall_of_fame` 按 `Loss` 升序寻找第一条数值稳定候选

#### `llmsr`

通常不需要额外回填。  
但对于 `timed_out` 任务，建议单独记录：

- `dataset_dir`
- `seed`
- `experiment_dir`
- `status`

必要时可以做二次分析，不建议立即自动重跑。

### 6.4 结果主键

正式实验所有结果汇总时，建议统一主键：

```text
task_key = dataset_dir | method | seed
```

不要只用：

- `dataset_name`

否则重复 basename 会再次互相覆盖。


## 7. 推荐执行顺序

1. 从 `/tmp/candidate_seeds_200_v3.json` 生成 `200` 数据集正式切片
2. 为 `pysr` 和 `llmsr` 各生成 `4` 份 slice CSV
3. 在 8 台机器上各起 `2` 个 tmux session
4. 定时巡检：
   - 每 `5~10` 分钟一次
5. 批次结束后：
   - 先汇总 `task_status.jsonl`
   - 再做 `pysr` 结果回填
   - 最后产出统一总表


## 8. 最小闭环建议

如果要开始执行，最小闭环是：

1. 先基于 `/tmp/candidate_seeds_200_v3.json` 生成正式切片
2. 先在 `iaaccn22` 和 `iaaccn26` 各跑一条单机 smoke
   - 验证 `7200s + 双 seed 并发` 配置
3. 确认无误后再全量铺开到 `22~29`
