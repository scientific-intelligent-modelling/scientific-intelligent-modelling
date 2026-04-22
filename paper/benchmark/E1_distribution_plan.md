# E1 分发方案与机器分配表

## 目标

`E1` 的目标是为 `Current-100 / Gap-only-100 / Quality-first-100 / Metadata-diverse-100` 四种 `100` 选择策略提供统一评估底座。

本阶段固定为：

- 数据集：`Candidate-200`
- 算法：`7`
  - `gplearn`
  - `pysr`
  - `pyoperon`
  - `llmsr`
  - `drsr`
  - `dso`
  - `tpsr`
- seed：`1314`
- 预算：`3600s`

总任务数：

- `200 × 7 × 1 = 1400`

## 本轮远端环境审计结果（2026-04-23）

基于刚刚对 `iaaccn22~29` 的实际检查，当前环境状态如下：

| 机器 | `sim_base` | `sim_llm` | `sim_dso` | `sim_tpsr` | 结论 |
|---|---:|---:|---:|---:|---|
| `iaaccn22` | ✅ | ✅ | ✅ | ✅ | 全环境可用 |
| `iaaccn23` | ✅ | ✅ | ✅ | ❌ | 缺 `sim_tpsr` |
| `iaaccn24` | ✅ | ✅ | ✅ | ❌ | 缺 `sim_tpsr` |
| `iaaccn25` | ✅ | ✅ | ✅ | ❌ | 缺 `sim_tpsr` |
| `iaaccn26` | ✅ | ✅ | ✅ | ❌ | 缺 `sim_tpsr` |
| `iaaccn27` | ✅ | ✅ | ✅ | ❌ | 缺 `sim_tpsr` |
| `iaaccn28` | ✅ | ✅ | ✅ | ❌ | 缺 `sim_tpsr` |
| `iaaccn29` | ✅ | ✅ | ❌ | ❌ | 缺 `sim_dso` / `sim_tpsr` |

## 关键结论

1. `gplearn / pysr / pyoperon`  
   - 可以稳定放在 `iaaccn22~25`

2. `llmsr / drsr`  
   - 可以稳定放在 `iaaccn26~29`

3. `dso`  
   - 可以放在 `iaaccn22~28`
   - 不建议放 `iaaccn29`，因为缺 `sim_dso`

4. `tpsr`  
   - 当前只有 `iaaccn22` 具备 `sim_tpsr`
   - **如果不先补环境，就无法做对称的多机分发**

因此，`E1` 的分发方案必须分成：

- **推荐方案**
- **保守可执行方案**

---

## 推荐方案

### 前置动作

在正式开始 `E1` 前，先补一个动作：

- 把 `sim_tpsr` 同步到：
  - `iaaccn23`
  - `iaaccn24`
  - `iaaccn25`

这样可以让 `tpsr` 和 `gplearn / pysr / pyoperon` 一样，在 `22~25` 上对称分发。

### 推荐切片方式

#### 四机切片（每片 50）

适用于：

- `gplearn`
- `pysr`
- `pyoperon`
- `llmsr`
- `drsr`
- `tpsr`（补好 `sim_tpsr` 后）

切片：

- `slice_01 = 50`
- `slice_02 = 50`
- `slice_03 = 50`
- `slice_04 = 50`

#### 七机切片（29/29/29/29/28/28/28）

适用于：

- `dso`

切片：

- `slice_01 = 29`
- `slice_02 = 29`
- `slice_03 = 29`
- `slice_04 = 29`
- `slice_05 = 28`
- `slice_06 = 28`
- `slice_07 = 28`

### 推荐运行波次

为了减少总 wall-clock，同时避免不同环境互相抢占，推荐按下面 5 个 wave 跑：

| Wave | 机器池 | 算法 | 任务量 | 说明 |
|---|---|---|---:|---|
| `W1` | `22~25` + `26~29` | `gplearn` + `llmsr` | `400` | 两组机器池互不冲突，可以并行 |
| `W2` | `22~25` + `26~29` | `pyoperon` + `drsr` | `400` | 同样互不冲突 |
| `W3` | `22~25` | `pysr` | `200` | `pysr` 单独跑，便于结果回填与巡检 |
| `W4` | `22~28` | `dso` | `200` | 使用 7 台机器，`29` 不参与 |
| `W5` | `22~25` | `tpsr` | `200` | 需要先补 `sim_tpsr` 到 `23~25` |

总计：

- `400 + 400 + 200 + 200 + 200 = 1400`

---

## 保守可执行方案

如果你不想先补 `sim_tpsr`，那当前立即可执行的保守方案是：

### `tpsr` 退化为单机

- `iaaccn22` 独跑 `200` 个 `tpsr` 任务

其余算法保持不变：

- `gplearn / pysr / pyoperon`：`iaaccn22~25`
- `llmsr / drsr`：`iaaccn26~29`
- `dso`：`iaaccn22~28`

### 保守方案的代价

1. `tpsr` 这一路会成为整个 `E1` 的最长尾巴
2. 机器利用率不均衡
3. `E1` 总 wall-clock 会明显变长

因此除非你特别急着先开跑，否则更推荐：

- **先补 `sim_tpsr` 到 `23~25`**

---

## 机器分配原则

### `iaaccn22~25`

主池用途：

- `gplearn`
- `pysr`
- `pyoperon`
- `tpsr`

理由：

- 这些算法都依赖本地/CPU 环境
- 不需要在线 LLM
- `pysr / pyoperon / tpsr` 的已有经验也主要积累在这组机器

### `iaaccn26~29`

主池用途：

- `llmsr`
- `drsr`

理由：

- 已有多轮在线 LLM 跑通经验
- `runtime_llm.config` / API key 复用链更成熟

### `iaaccn22~28`

扩展池用途：

- `dso`

理由：

- `sim_dso` 当前已确认到 `22~28`
- `29` 缺 `sim_dso`

---

## 统一命名规范

建议 `E1` 的批次命名固定成：

```text
e1_candidate200_seed1314_<wave>_<timestamp>
```

例如：

- `e1_candidate200_seed1314_w1_20260423-101500`
- `e1_candidate200_seed1314_w2_20260423-164000`

对应输出目录建议：

```text
experiments/e1_candidate200_seed1314_<wave>_<timestamp>/
```

---

## 运行前必须满足的条件

### 必须满足

1. `Candidate-200` 切片 CSV 已生成
2. `Clean-Master-100` 已冻结，不再回改
3. `dso / drsr / tpsr` 已通过最近一轮 smoke
4. 远端代码已同步到目标机器

### 推荐满足

1. `sim_tpsr` 已同步到 `iaaccn23~25`
2. `dso` 在 `iaaccn29` 若要使用，则需先补 `sim_dso`

---

## 一句话建议

如果你现在就要排 `E1`：

- **最推荐：先补 `sim_tpsr` 到 `23~25`，然后按 5 个 wave 跑 1400 个任务**
- **保守可执行：不补环境也能跑，但 `tpsr` 只能单机，会拖长总时间**
