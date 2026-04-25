# stressstrain 200 代 RMSE 提升比例

提升比例按下式计算：

```text
提升比例 = (基线 RMSE - DRSR_RMSE) / 基线 RMSE * 100%
```

正值表示 DRSR 的 RMSE 更低。以下主表只使用同为 200 代的结果；`best-of-two` 表示在两轮 200 代重跑中，对每个算法和每个 split 分别取最低 RMSE。

## 主表：200 代 best-of-two

| 数据集 | 对比对象 | ID RMSE 提升 | OOD RMSE 提升 |
|---|---|---:|---:|
| stressstrain | 相比 llmsr 200代 | 21.9961% | 19.6352% |
| stressstrain | 相比 pysr 200代 | 19.6950% | 25.9346% |

使用的 best RMSE：

| 算法 | ID RMSE | OOD RMSE |
|---|---:|---:|
| drsr | 0.0433494665 | 0.0401321623 |
| llmsr | 0.0555734593 | 0.0499374750 |
| pysr | 0.0539810568 | 0.0541847773 |

## 明细：逐轮 200 代 paired comparison

| 批次 | 数据集 | 对比对象 | ID RMSE 提升 | OOD RMSE 提升 |
|---|---|---|---:|---:|
| rerun200_llm_drsr_20260405_150543 | stressstrain | 相比 llmsr 200代 | 22.4225% | 67.3150% |
| rerun200_llm_drsr_20260405_150543 | stressstrain | 相比 pysr 200代 | 25.9277% | 41.0179% |
| rerun200_llm_drsr_20260406_111429 | stressstrain | 相比 llmsr 200代 | 13.3644% | 19.6352% |
| rerun200_llm_drsr_20260406_111429 | stressstrain | 相比 pysr 200代 | 10.8087% | 25.9346% |

逐轮 RMSE 来源：

| 批次 | 算法 | ID RMSE | OOD RMSE |
|---|---|---:|---:|
| rerun200_llm_drsr_20260405_150543 | drsr | 0.0433494665 | 0.0401904881 |
| rerun200_llm_drsr_20260405_150543 | llmsr | 0.0558788980 | 0.1229632315 |
| rerun200_llm_drsr_20260405_150543 | pysr | 0.0585231917 | 0.0681401524 |
| rerun200_llm_drsr_20260406_111429 | drsr | 0.0481463838 | 0.0401321623 |
| rerun200_llm_drsr_20260406_111429 | llmsr | 0.0555734593 | 0.0499374750 |
| rerun200_llm_drsr_20260406_111429 | pysr | 0.0539810568 | 0.0541847773 |
