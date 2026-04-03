# 实验超参数清单

这个文件专门记录当前项目里**实验执行时使用或推荐使用的超参数**。

读取规则：

- 需要查实验参数时，优先看这里
- 不要在 `.codex/AGENTS.md` 里重复维护具体数值
- 若参数有更新，优先修改本文件

---

## 1. PySR

### 1.1 快速验证参数

适用于：
- 流程 smoke test
- 远程手工快速验证

```python
niterations = 100
population_size = 64
populations = 32
maxsize = 30
procs = 32
progress = True
verbosity = 1
random_state = 1314
```

## 2. LLM-SR
### 2.2 高预算参数

适用于：
- 当前高预算 `matsci` 批次

```python
niterations = 200
samples_per_iteration = 4
max_params = 12
```

总预算：

```text
200 * 4 = 800
```

### 2.3 推荐统一接口

`llmsr` 默认使用：

- `llm_config_path`
- `background`
- `metadata_path`
- `niterations`
- `samples_per_iteration`
- `seed`
- `problem_name`
- `exp_path`
- `exp_name`

---

## 3. DRSR

### 3.1 当前对齐后的外部接口

`drsr` 对外接口已经对齐到 `llmsr`，推荐使用：

```python
llm_config_path
background
metadata_path
niterations
samples_per_iteration
seed
problem_name
exp_path
exp_name
evaluate_timeout_seconds
wall_time_limit_seconds
```

### 3.2 当前内部预算映射

内部自动映射为：

```text
max_samples = niterations * samples_per_iteration
samples_per_prompt = samples_per_iteration
```

### 3.4 高预算参数

当前推荐的高预算参数是：

```python
niterations = 200
samples_per_iteration = 4
evaluate_timeout_seconds = 20
wall_time_limit_seconds = 900
```

内部等价于：

```python
max_samples = 800
samples_per_prompt = 4
```

### 3.5 旧参数兼容说明

下面这些旧参数仍可作为 fallback 被识别：

```python
max_samples
samples_per_prompt
workdir
```

但不再建议作为主接口使用。

---


## 5. 远程实验约定

### 5.1 matsci 数据路径

```bash
.sim_datasets/llm-srbench/

具体数据集需听指示，如果没有需要使用sim_datasets包去下载，该包存放在项目根目录中
```

### 5.2 运行环境

- `pysr`：`sim_base`
- `llmsr`：`sim_llm`
- `drsr`：`sim_llm`

### 5.3 统一入口

```bash
check/run_phys_osc_task.py
```

### 5.4 会话前缀建议

- `pysr`：自行定义，但建议带预算前缀
- `llmsr`：如 `ml200_`
- `drsr`：如 `mdn200_`

### 5.5 日志与结果

每个数据集：
- 一个独立 `tmux` 会话
- 一个独立日志
- 一个独立 `result.json`

---

## 6. 备注

1. `drsr` 的 prompt 现在应使用：
   - `x0/x1/.../y`
   - metadata 物理语义
   - 不再使用 `with driving force`
   - 不再向模型暴露外层 `col0/col1`

2. `pysr` 如果想看到中间输出，必须显式开启：

```python
progress = True
verbosity = 1
```

3. 如果后续新增预算档位，直接在本文件追加，不要把数值散落到多个文档里。
