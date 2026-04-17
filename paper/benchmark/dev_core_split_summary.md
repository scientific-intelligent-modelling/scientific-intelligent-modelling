# Dev-50 / Core-50 切分说明（论文版简表）

## 设计原则

- 先基于三 seed 正式结果从 200 个候选中筛出 `Master-100`。
- 再只用非结果信息切分成 `Dev-50` 与 `Core-50`，避免直接按正式成绩量身定制测试集。
- 强约束：`Master-100` 内 `basename <= 1`，避免同 basename 变体跨集合泄漏。
- `Core-50` 中 `one-sided` 样本数限制为 `<= 5`。

## 为什么这样选

### 1. 为什么先做 `Master-100`

- 原始候选池是 `200` 个数据集，已经经过双探针和三 seed 正式实验筛选，但仍然偏大，不适合直接切成 `Dev/Core` 后长期冻结。
- 因此先用三 seed 正式结果把 `200` 压缩成 `Master-100`，目标是：
  - 保留高区分度样本；
  - 保留足够多的 family / subgroup；
  - 在 `PySR` 优势样本更多的现实前提下，仍给 `LLM-SR` 保留稳定代表。

### 2. 为什么 `Dev/Core` 切分不用正式结果

- 如果切分时继续使用 `priority_score / three_seed_gap / 正式 OOD 指标`，`Core-50` 就会变成按结果量身定制的测试集。
- 因此 `Dev/Core` 只使用非结果信息：
  - `family`
  - `subgroup`
  - `selection_mode`
  - `candidate_advantage_side`
  - `basename`
  - 特征维度
  - 各 split 样本量
  - 静态公式复杂度
- 这样做的目的是：
  - `Master-100` 负责“题选得好不好”；
  - `Dev/Core` 负责“切得干不干净”。

### 3. 为什么要加 `basename <= 1`

- 在原始 `200` 个候选中，存在 `12` 组重复 `basename`，且最大重复次数为 `2`。
- 如果同 basename 的 dummy / hard / medium 变体同时进入 `Dev` 与 `Core`，后续在 `Dev` 上进化时会引入结构泄漏。
- 因此我们在整个 `Master-100` 上施加：
  - `basename <= 1`
- 最终 `Master-100` 满足：
  - `max basename multiplicity = 1`

## 数据支撑

### 原始 200 候选池分布

- family：
  - `srsd = 70`
  - `llm-srbench = 70`
  - `srbench1.0 = 33`
  - `korns = 9`
  - `keijzer = 6`
  - `nguyen = 6`
  - `srbench2025 = 6`
- `selection_mode`：
  - `strict = 126`
  - `mid-gap = 40`
  - `relaxed = 14`
  - `one-sided = 20`
- 候选阶段优势标签：
  - `pysr = 120`
  - `llmsr = 80`

### `Master-100` 的压缩逻辑

- family 目标配额：
  - `srsd = 34`
  - `llm-srbench = 30`
  - `srbench1.0 = 16`
  - `nguyen = 6`
  - `keijzer = 6`
  - `korns = 4`
  - `srbench2025 = 4`
- 这组配额的含义是：
  - 对大族（`srsd` / `llm-srbench`）做近似“减半保留”；
  - 对中小族（`keijzer / nguyen / srbench2025 / korns`）尽量完整保留；
  - 保证最终 100 个样本仍覆盖主要题型。

- `Master-100` 实际结果：
  - family：
    - `srsd = 34`
    - `llm-srbench = 30`
    - `srbench1.0 = 16`
    - `nguyen = 6`
    - `keijzer = 6`
    - `korns = 4`
    - `srbench2025 = 4`
  - 候选阶段优势标签：
    - `pysr = 69`
    - `llmsr = 31`
  - `selection_mode`：
    - `strict = 62`
    - `mid-gap = 23`
    - `relaxed = 5`
    - `one-sided = 10`

- 这里 `selection_mode` 没有完全达到最初目标值，原因不是脚本失败，而是三个约束共同作用：
  - `basename <= 1`
  - family 配额固定
  - 高优先级样本本身更多集中在 `strict`

### `Dev-50 / Core-50` 的匹配证据

- family 分布：
  - `srsd = 17 / 17`
  - `llm-srbench = 14 / 16`
  - `srbench1.0 = 9 / 7`
  - `nguyen = 3 / 3`
  - `keijzer = 3 / 3`
  - `korns = 2 / 2`
  - `srbench2025 = 2 / 2`

- `selection_mode` 分布：
  - `strict = 31 / 31`
  - `mid-gap = 11 / 12`
  - `relaxed = 3 / 2`
  - `one-sided = 5 / 5`

- 候选阶段优势标签：
  - `Dev-50`: `pysr = 34`, `llmsr = 16`
  - `Core-50`: `pysr = 35`, `llmsr = 15`

- 静态属性均值也比较接近：
  - `feature_count`：`3.76 vs 3.62`
  - `train_samples`：`24044.22 vs 23934.44`
  - `ood_test_samples`：`4930.28 vs 4936.16`
  - `formula_operator_count`：`220.92 vs 228.18`

- 这些数字说明：
  - `Dev-50` 与 `Core-50` 在来源、标签结构和静态难度代理上都比较接近；
  - 但 `Core-50` 又没有被正式结果二次定制，因此更适合作最终冻结测试集。

## 集合规模

- `Master-100 = 100`
- `Dev-50 = 50`
- `Core-50 = 50`

## Master-100 顶层 family 分布

| family | 数量 |
|---|---:|
| `keijzer` | 6 |
| `korns` | 4 |
| `llm-srbench` | 30 |
| `nguyen` | 6 |
| `srbench1.0` | 16 |
| `srbench2025` | 4 |
| `srsd` | 34 |

### Dev/Core 顶层 family 分布

| 项目 | Dev-50 | Core-50 |
|---|---:|---:|
| `keijzer` | 3 | 3 |
| `korns` | 2 | 2 |
| `llm-srbench` | 14 | 16 |
| `nguyen` | 3 | 3 |
| `srbench1.0` | 9 | 7 |
| `srbench2025` | 2 | 2 |
| `srsd` | 17 | 17 |

### Dev/Core selection_mode 分布

| 项目 | Dev-50 | Core-50 |
|---|---:|---:|
| `mid-gap` | 11 | 12 |
| `one-sided` | 5 | 5 |
| `relaxed` | 3 | 2 |
| `strict` | 31 | 31 |

### Dev/Core 候选阶段优势标签分布

| 项目 | Dev-50 | Core-50 |
|---|---:|---:|
| `llmsr` | 16 | 15 |
| `pysr` | 34 | 35 |

## 静态属性匹配摘要

| 指标 | Dev-50 mean | Dev-50 median | Core-50 mean | Core-50 median |
|---|---:|---:|---:|---:|
| `feature_count` | 3.76 | 3.0 | 3.62 | 3.5 |
| `train_samples` | 24044.22 | 8000.0 | 23934.44 | 8000.0 |
| `valid_samples` | 2402.08 | 1000.0 | 2301.68 | 1000.0 |
| `id_test_samples` | 11971.32 | 1000.0 | 12152.7 | 1000.0 |
| `ood_test_samples` | 4930.28 | 1000.0 | 4936.16 | 1000.0 |
| `formula_operator_count` | 220.92 | 93.5 | 228.18 | 95.5 |

## 使用建议

- `Dev-50`：用于方法迭代、进化和超参选择。
- `Core-50`：冻结，不参与任何调参，只用于最终汇报。
- 两个集合分布尽量一致，但 `Core-50` 不再根据正式结果做二次微调。
