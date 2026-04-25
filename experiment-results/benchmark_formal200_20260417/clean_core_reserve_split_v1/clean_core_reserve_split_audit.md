# Clean-Master-100 / Core-50 / Reserve-50 切分审计

## 口径

- 输入：`Clean-Master-100`。
- 输出：`Core-50` 与 `Reserve-50`。
- 切分时只使用静态/非结果属性：`family / subgroup / dummy 标记 / feature_count / sample size / formula complexity`。
- `selection_mode`、`candidate_advantage_side`、`priority_score`、`quality_score` 等结果派生字段只在本报告中审计，不参与切分优化。

## 总览

- Master 数量：`100`
- Core 数量：`50`
- Reserve 数量：`50`
- 随机搜索次数：`30000`
- 随机种子：`20260426`
- split loss：`28.163206`

## family 分布

| family | Core-50 | Reserve-50 |
|---|---:|---:|
| `keijzer` | 3 | 3 |
| `korns` | 2 | 2 |
| `llm-srbench` | 15 | 15 |
| `nguyen` | 3 | 3 |
| `srbench1.0` | 8 | 8 |
| `srbench2025` | 2 | 2 |
| `srsd` | 17 | 17 |

## 审计字段分布（未参与切分优化）

### selection_mode

| selection_mode | Core-50 | Reserve-50 |
|---|---:|---:|
| `mid-gap` | 11 | 12 |
| `one-sided` | 5 | 5 |
| `relaxed` | 4 | 1 |
| `strict` | 30 | 32 |

### candidate_advantage_side

| side | Core-50 | Reserve-50 |
|---|---:|---:|
| `llmsr` | 17 | 14 |
| `pysr` | 33 | 36 |

## 静态数值字段

| 字段 | Core mean | Core median | Reserve mean | Reserve median |
|---|---:|---:|---:|---:|
| `feature_count` | 3.6739130434782608 | 4.0 | 3.6444444444444444 | 3.0 |
| `train_samples` | 18668.978260869564 | 8000.0 | 23959.11111111111 | 8000.0 |
| `valid_samples` | 1774.1304347826087 | 1000.0 | 2323.9555555555557 | 1000.0 |
| `id_test_samples` | 12027.478260869566 | 1000.0 | 12754.155555555555 | 1000.0 |
| `ood_test_samples` | 4082.717391304348 | 1000.0 | 4944.822222222222 | 1000.0 |
| `formula_line_count` | 10.195652173913043 | 2.0 | 10.466666666666667 | 2.0 |
| `formula_char_count` | 241.04347826086956 | 100.5 | 250.2 | 107.0 |
| `formula_operator_count` | 235.8913043478261 | 99.5 | 245.0 | 109.0 |
