# Dev-50 / Core-50 切分说明（论文版简表）

## 设计原则

- 先基于三 seed 正式结果从 200 个候选中筛出 `Master-100`。
- 再只用非结果信息切分成 `Dev-50` 与 `Core-50`，避免直接按正式成绩量身定制测试集。
- 强约束：`Master-100` 内 `basename <= 1`，避免同 basename 变体跨集合泄漏。
- `Core-50` 中 `one-sided` 样本数限制为 `<= 5`。

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
