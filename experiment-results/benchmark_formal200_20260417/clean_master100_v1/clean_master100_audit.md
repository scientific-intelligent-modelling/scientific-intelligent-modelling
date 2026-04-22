# Clean-Master-100 审计报告

## 总结

- 输入 `Master-100`：`100` 个
- 输出 `Clean-Master-100`：`100` 个
- 在 `Candidate-200` 全局语义审计中发现的重复组数：`26`
- 当前 `Master-100` 内实际命中的重复组数：`9`
- 本次删除并补位条数：`9`

## 删除规则

本次只删除两类：

1. semantic duplicate
2. 结构性坏样本（本轮若存在）

不处理：

- 质量分数偏低
- `strict` 太多
- `srsd` 太多
- 某一算法表现偏强/偏弱

## 清洗前后分布

### family

- before: {'srsd': 34, 'srbench2025': 4, 'srbench1.0': 16, 'keijzer': 6, 'nguyen': 6, 'llm-srbench': 30, 'korns': 4}
- after: {'srsd': 34, 'srbench2025': 4, 'srbench1.0': 16, 'keijzer': 6, 'nguyen': 6, 'llm-srbench': 30, 'korns': 4}

### selection_mode

- before: {'strict': 62, 'relaxed': 5, 'mid-gap': 23, 'one-sided': 10}
- after: {'strict': 62, 'relaxed': 5, 'mid-gap': 23, 'one-sided': 10}

### candidate_advantage_side

- before: {'pysr': 69, 'llmsr': 31}
- after: {'pysr': 69, 'llmsr': 31}

## 被识别出的 `Master-100` 内重复组


### group 1

- `II.34.29b_2_0` | `sim-datasets-data/llm-srbench/lsrtransform/II.34.29b_2_0`
- `II.34.29b_4_0` | `sim-datasets-data/llm-srbench/lsrtransform/II.34.29b_4_0`

### group 2

- `III.15.27_1_0` | `sim-datasets-data/llm-srbench/lsrtransform/III.15.27_1_0`
- `III.15.27_2_0` | `sim-datasets-data/llm-srbench/lsrtransform/III.15.27_2_0`

### group 4

- `feynman_III_15_27` | `sim-datasets-data/srbench1.0/feynman/feynman_III_15_27`
- `feynman-iii.15.27` | `sim-datasets-data/srsd/srsd-feynman_easy_dummy/feynman-iii.15.27`

### group 5

- `feynman_III_17_37` | `sim-datasets-data/srbench1.0/feynman/feynman_III_17_37`
- `feynman-iii.17.37` | `sim-datasets-data/srsd/srsd-feynman_medium_dummy/feynman-iii.17.37`

### group 9

- `feynman_II_34_11` | `sim-datasets-data/srbench1.0/feynman/feynman_II_34_11`
- `feynman-ii.34.11` | `sim-datasets-data/srsd/srsd-feynman_easy_dummy/feynman-ii.34.11`

### group 12

- `feynman_II_8_7` | `sim-datasets-data/srbench1.0/feynman/feynman_II_8_7`
- `feynman-ii.8.7` | `sim-datasets-data/srsd/srsd-feynman_medium_dummy/feynman-ii.8.7`

### group 13

- `feynman_I_14_3` | `sim-datasets-data/srbench1.0/feynman/feynman_I_14_3`
- `feynman-i.14.3` | `sim-datasets-data/srsd/srsd-feynman_easy_dummy/feynman-i.14.3`

### group 15

- `feynman_I_27_6` | `sim-datasets-data/srbench1.0/feynman/feynman_I_27_6`
- `feynman-i.27.6` | `sim-datasets-data/srsd/srsd-feynman_easy_dummy/feynman-i.27.6`

### group 17

- `feynman-iii.12.43` | `sim-datasets-data/srsd/srsd-feynman_easy_dummy/feynman-iii.12.43`
- `feynman-i.34.27` | `sim-datasets-data/srsd/srsd-feynman_medium_dummy/feynman-i.34.27`

## 补位记录

- 删除 `feynman-iii.17.37`，补入 `feynman-bonus.8`；family `srsd -> srsd`，subgroup `srsd/srsd-feynman_medium_dummy -> srsd/srsd-feynman_medium_dummy`，score=`1070.0`
- 删除 `feynman-ii.34.11`，补入 `feynman-i.12.1`；family `srsd -> srsd`，subgroup `srsd/srsd-feynman_easy_dummy -> srsd/srsd-feynman_easy_dummy`，score=`1070.0`
- 删除 `feynman_I_27_6`，补入 `feynman_III_10_19`；family `srbench1.0 -> srbench1.0`，subgroup `srbench1.0/feynman -> srbench1.0/feynman`，score=`1070.0`
- 删除 `feynman-i.34.27`，补入 `feynman-i.10.7`；family `srsd -> srsd`，subgroup `srsd/srsd-feynman_medium_dummy -> srsd/srsd-feynman_medium_dummy`，score=`1070.0`
- 删除 `feynman_I_14_3`，补入 `feynman_III_15_14`；family `srbench1.0 -> srbench1.0`，subgroup `srbench1.0/feynman -> srbench1.0/feynman`，score=`1070.0`
- 删除 `feynman_II_8_7`，补入 `feynman_II_11_20`；family `srbench1.0 -> srbench1.0`，subgroup `srbench1.0/feynman -> srbench1.0/feynman`，score=`1070.0`
- 删除 `II.34.29b_4_0`，补入 `I.11.19_1_0`；family `llm-srbench -> llm-srbench`，subgroup `llm-srbench/lsrtransform -> llm-srbench/lsrtransform`，score=`1070.0`
- 删除 `III.15.27_2_0`，补入 `I.11.19_2_0`；family `llm-srbench -> llm-srbench`，subgroup `llm-srbench/lsrtransform -> llm-srbench/lsrtransform`，score=`1070.0`
- 删除 `feynman_III_15_27`，补入 `feynman_I_43_16`；family `srbench1.0 -> srbench1.0`，subgroup `srbench1.0/feynman -> srbench1.0/feynman`，score=`1070.0`
