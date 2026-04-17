# Master-100 / Dev-50 / Core-50 切分审计报告

## 方案说明

- `Master-100`：允许使用三 seed 正式结果进行筛选。
- `Dev-50 / Core-50`：只使用非结果信息切分，包括 `family / subgroup / selection_mode / candidate_advantage_side / basename / 特征维度 / 样本量 / 静态公式复杂度`。
- `basename <= 1` 在整个 `Master-100` 上成立，避免 dev/test 结构泄漏。

- Master-100 数量：`100`
- Dev-50 数量：`50`
- Core-50 数量：`50`

## Master-100 配额实现

### family（目标 / 实现 / 实际上限）
- `srsd`: target=`34`, realized=`34`, cap_used=`34`
- `llm-srbench`: target=`30`, realized=`30`, cap_used=`30`
- `srbench1.0`: target=`16`, realized=`16`, cap_used=`16`
- `nguyen`: target=`6`, realized=`6`, cap_used=`6`
- `keijzer`: target=`6`, realized=`6`, cap_used=`6`
- `korns`: target=`4`, realized=`4`, cap_used=`4`
- `srbench2025`: target=`4`, realized=`4`, cap_used=`4`

### selection_mode（目标 / 实现）
- `strict`: target=`52`, realized=`62`
- `mid-gap`: target=`26`, realized=`23`
- `relaxed`: target=`10`, realized=`5`
- `one-sided`: target=`12`, realized=`10`

### candidate_advantage_side（目标 / 实现）
- `pysr`: target=`70`, realized=`69`
- `llmsr`: target=`30`, realized=`31`

- Master-100 最大 subgroup 占用：`16`

## 离散分布

### family
- `keijzer`: dev=`3`, core=`3`
- `korns`: dev=`2`, core=`2`
- `llm-srbench`: dev=`14`, core=`16`
- `nguyen`: dev=`3`, core=`3`
- `srbench1.0`: dev=`9`, core=`7`
- `srbench2025`: dev=`2`, core=`2`
- `srsd`: dev=`17`, core=`17`

### selection_mode
- `mid-gap`: dev=`11`, core=`12`
- `one-sided`: dev=`5`, core=`5`
- `relaxed`: dev=`3`, core=`2`
- `strict`: dev=`31`, core=`31`

### candidate_advantage_side
- `llmsr`: dev=`16`, core=`15`
- `pysr`: dev=`34`, core=`35`

- `Core-50 one-sided` 上限：`5`，实际：`5`

## 连续静态特征摘要
- `feature_count`: dev(mean=`3.76`, median=`3.0`, n=`50`), core(mean=`3.62`, median=`3.5`, n=`50`)
- `train_samples`: dev(mean=`24044.22`, median=`8000.0`, n=`50`), core(mean=`23934.44`, median=`8000.0`, n=`50`)
- `valid_samples`: dev(mean=`2402.08`, median=`1000.0`, n=`50`), core(mean=`2301.68`, median=`1000.0`, n=`50`)
- `id_test_samples`: dev(mean=`11971.32`, median=`1000.0`, n=`50`), core(mean=`12152.7`, median=`1000.0`, n=`50`)
- `ood_test_samples`: dev(mean=`4930.28`, median=`1000.0`, n=`50`), core(mean=`4936.16`, median=`1000.0`, n=`50`)
- `formula_line_count`: dev(mean=`9.48`, median=`2.0`, n=`50`), core(mean=`9.68`, median=`2.0`, n=`50`)
- `formula_char_count`: dev(mean=`225.74`, median=`94.5`, n=`50`), core(mean=`232.44`, median=`97.5`, n=`50`)
- `formula_operator_count`: dev(mean=`220.92`, median=`93.5`, n=`50`), core(mean=`228.18`, median=`95.5`, n=`50`)