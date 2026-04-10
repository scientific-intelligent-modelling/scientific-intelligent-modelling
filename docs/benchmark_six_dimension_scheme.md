# 六维 Benchmark 简版方案

## 1. 目标

只保留两层信息：

- 每个维度依赖哪些 `raw` 子指标
- 这些 `raw` 指标如何汇总成 6 个大类分数

## 2. 通用规则

### 2.1 基本记号

```text
clip01(x) = clip(x, 0, 1)
inv_pos(x) = 1 / (1 + max(x, 0))
count_score(c) = 1 / (1 + log(1 + max(c, 0)))
time_budget_score(t, B) = clip(1 - t / B, 0, 1)
```

说明：

- `clip01(x)` 用于本来就在 `[0, 1]` 的指标
- `inv_pos(x)` 用于非负、无界、且无量纲的下降型指标，例如 `nmse`、退化率、变异系数
- `count_score(c)` 用于复杂度这类计数型指标
- `time_budget_score(t, B)` 用于时间成本，其中 `B` 是 benchmark 协议中固定的时间预算

### 2.2 比较切片

本方案不再使用相对排名分。

所有分数都必须是绝对分：

- 同一个 `raw` 输入必须映射到同一个 `score`
- 新方法接入后，旧结果分数不能变化
- 唯一允许依赖外部协议字段的是时间预算 `time_budget_seconds`

### 2.3 时间预算

时间维度要想做成绝对分，必须引入 benchmark 协议中的固定预算：

- `time_budget_seconds`

这不是调参阈值，而是 benchmark 本身的运行约束。

如果某个 benchmark family 还没有固定预算：

- 时间维度先只报 `train_time_seconds`
- 不计算 `time_cost_score`

## 3. 六个维度

## 3.1 时间成本

### raw

- `train_time_seconds`
- `time_budget_seconds`

### 汇总

```text
time_cost_score = time_budget_score(train_time_seconds, time_budget_seconds)
```

## 3.2 ID 数值质量

### raw

- `id_test.rmse`
- `id_test.nmse`
- `id_test.r2`
- `id_test.acc_0_1`

### 预处理

```text
id_nmse_score = inv_pos(id_test.nmse)
id_r2_score   = clip01(id_test.r2)
id_acc_score  = clip01(id_test.acc_0_1)
```

### 汇总

```text
id_quality_score =
    0.40 * id_nmse_score
  + 0.35 * id_acc_score
  + 0.25 * id_r2_score
```

备注：

- `id_test.rmse` 保留为展示值，不进聚合
- 若缺 `acc_0_1`，则其余权重按比例重分配

## 3.3 OOD 泛化能力

### raw

- `ood_test.rmse`
- `ood_test.nmse`
- `ood_test.r2`
- `ood_test.acc_0_1`

### 预处理

```text
ood_nmse_score = inv_pos(ood_test.nmse)
ood_r2_score   = clip01(ood_test.r2)
ood_acc_score  = clip01(ood_test.acc_0_1)
```

### 汇总

```text
ood_generalization_score =
    0.40 * ood_nmse_score
  + 0.35 * ood_acc_score
  + 0.25 * ood_r2_score
```

备注：

- `ood_test.rmse` 保留为展示值，不进聚合

## 3.4 抗噪鲁棒性

### raw

- `noise@0.01.id_test.nmse`
- `noise@0.05.id_test.nmse`
- `noise@0.10.id_test.nmse`
- `noise@0.01.id_test.acc_0_1`
- `noise@0.05.id_test.acc_0_1`
- `noise@0.10.id_test.acc_0_1`
- `id_test.nmse`
- `id_test.acc_0_1`

### 预处理

```text
deg_nmse(σ) = max(0, (nmse_σ - nmse_clean) / (nmse_clean + ε))
deg_acc(σ)  = max(0, (acc_clean - acc_σ) / max(acc_clean, ε))

avg_deg_nmse = mean_σ deg_nmse(σ)
avg_deg_acc  = mean_σ deg_acc(σ)

noise_nmse_score = inv_pos(avg_deg_nmse)
noise_acc_score  = inv_pos(avg_deg_acc)
```

### 汇总

```text
noise_robustness_score =
    0.60 * noise_nmse_score
  + 0.40 * noise_acc_score
```

## 3.5 稳定性

### raw

- `id_test.nmse_mean`
- `id_test.nmse_std`
- `ood_test.nmse_mean`
- `ood_test.nmse_std`
- `complexity_simplified_std`
- `valid_run_rate`

### 预处理

```text
cv_id_nmse  = id_test.nmse_std  / max(abs(id_test.nmse_mean), ε)
cv_ood_nmse = ood_test.nmse_std / max(abs(ood_test.nmse_mean), ε)

stab_id_score         = inv_pos(cv_id_nmse)
stab_ood_score        = inv_pos(cv_ood_nmse)
stab_complexity_score = count_score(complexity_simplified_std)
stab_valid_score      = clip01(valid_run_rate)
```

### 汇总

```text
stability_score =
    0.35 * stab_id_score
  + 0.35 * stab_ood_score
  + 0.15 * stab_complexity_score
  + 0.15 * stab_valid_score
```

## 3.6 符号保真与可解释性

### raw

- `symbolic_solution`
- `solution_rate`
- `symbolic_accuracy`
- `ned`
- `complexity_raw`
- `complexity_simplified`

### 预处理

```text
symbolic_acc_score  = clip01(symbolic_accuracy)
solution_rate_score = clip01(solution_rate)
ned_score           = 1 - clip01(ned)
complexity_score    = count_score(complexity_simplified)
```

### 汇总

```text
symbolic_interpretability_score =
    0.35 * symbolic_acc_score
  + 0.25 * solution_rate_score
  + 0.20 * ned_score
  + 0.20 * complexity_score
```

备注：

- `symbolic_solution` 保留为单次运行标签，不直接进聚合
- `complexity_raw` 保留为展示值，不直接进聚合

## 4. 最小输出结构

```json
{
  "time_cost": {
    "raw": {"train_time_seconds": 12.3},
    "score": 0.62
  },
  "id_quality": {
    "raw": {"rmse": 0.12, "nmse": 0.03, "r2": 0.98, "acc_0_1": 0.91},
    "score": 0.89
  },
  "ood_generalization": {
    "raw": {"rmse": 0.20, "nmse": 0.08, "r2": 0.93, "acc_0_1": 0.84},
    "score": 0.77
  },
  "noise_robustness": {
    "raw": {"avg_deg_nmse": 0.31, "avg_deg_acc": 0.12},
    "score": 0.74
  },
  "stability": {
    "raw": {
      "cv_id_nmse": 0.10,
      "cv_ood_nmse": 0.14,
      "complexity_simplified_std": 1.2,
      "valid_run_rate": 1.0
    },
    "score": 0.83
  },
  "symbolic_interpretability": {
    "raw": {
      "symbolic_accuracy": 1.0,
      "solution_rate": 0.8,
      "ned": 0.1,
      "complexity_simplified": 7
    },
    "score": 0.86
  }
}
```
