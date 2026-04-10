# 六维 Benchmark 量化方案

## 1. 目标

本方案用于将当前仓库中的符号回归 benchmark 结果统一整理为 6 个可量化维度。

这 6 个维度分别是：

1. 时间成本
2. ID 数值质量
3. OOD 泛化能力
4. 抗噪鲁棒性
5. 稳定性
6. 符号保真与可解释性

设计原则：

- 每个维度都必须对应明确的原始字段；
- 每个维度都必须给出可复现的量化公式；
- raw metric 和 normalized score 同时保留；
- 不以单一总分替代分维度报告；
- 只有满足前置条件的任务才允许计算对应维度；
- 完整 6 维总分仅适用于满足 GT、OOD、noise、multi-seed 协议的任务集。

## 2. 通用记号与归一化函数

### 2.1 原始值与方向

- 越小越好：
  - `train_time_seconds`
  - `rmse`
  - `nmse`
  - `complexity_*`
  - `ned`
- 越大越好：
  - `r2`
  - `acc_0_1`
  - `solution_rate`
  - `symbolic_accuracy`
  - `valid_run_rate`

### 2.2 建议使用的标准归一化函数

#### 有界指标的归一化

用于已经天然位于 `[0, 1]` 区间，或可以稳定裁剪到 `[0, 1]` 的量：

```text
up01(x) = clip(x, 0, 1)
```

对于 `r2`，推荐：

```text
r2_score = clip(r2, 0, 1)
```

说明：

- 负 `r2` 统一视为 `0`
- `r2 > 1` 统一截断到 `1`

#### 无界下降型指标的归一化

用于“越小越好”且没有天然上界的量，例如：

- `train_time_seconds`
- `nmse`
- `complexity_*`
- 稳定性里的各类波动量

推荐使用同一比较切片内的分位数/排名归一化：

```text
rank_down(x) = 1 - percentile_rank(x)
```

其中：

- `percentile_rank(x)` 取值位于 `[0, 1]`
- 同一切片内最小值的分数接近 `1`
- 同一切片内最大值的分数接近 `0`

为了避免歧义，建议实现为平均名次版本：

```text
rank_down(x_i) = 1 - (avg_rank_asc(x_i) - 1) / max(n - 1, 1)
```

这里：

- `avg_rank_asc` 表示按升序排名后的平均名次
- `n` 是当前比较切片内的方法数

### 2.3 关于比较切片

所有 `rank_down(*)` 都必须在同一比较切片内计算。

推荐的比较切片定义为：

- 同一个 benchmark family
- 同一批任务
- 同一预算约束
- 同一评测协议

重要：

- 不能把不同 benchmark family 混在一个切片里算排名分
- 不能把不同预算设置混在一个切片里算排名分
- 如果当前切片里只有 1 个方法，则该维度只报告 raw，不报告 score

## 3. 六个维度

## 3.1 时间成本

### 定义

衡量方法完成一次训练/搜索所需的时间开销。

### 原始字段

- `train_time_seconds`

可选扩展字段：

- `evaluated_expressions`
- `api_tokens`
- `api_cost`

### 推荐量化

```text
time_cost_score = rank_down(train_time_seconds)
```

### 报告建议

同时保留：

- raw：`train_time_seconds`
- score：`time_cost_score`

## 3.2 ID 数值质量

### 定义

衡量模型在分布内测试集上的预测精度。

### 原始字段

- `id_test.rmse`
- `id_test.r2`
- `id_test.nmse`
- `id_test.acc_0_1`

说明：

- `RMSE` 适合展示绝对误差
- `NMSE` 更适合跨任务归一化比较
- `Acc0.1` 对齐 `LLM-SRBench` 风格

### 推荐量化

建议使用：

```text
id_quality_score =
    0.40 * rank_down(id_test.nmse)
  + 0.35 * up01(id_test.acc_0_1)
  + 0.25 * clip(id_test.r2, 0, 1)
```

说明：

- `rmse` 保留为展示值，不直接参与跨任务聚合
- 若某任务没有 `acc_0_1`，则剩余权重按比例重分配

## 3.3 OOD 泛化能力

### 定义

衡量模型在分布外测试集上的泛化能力。

### 原始字段

- `ood_test.rmse`
- `ood_test.r2`
- `ood_test.nmse`
- `ood_test.acc_0_1`

### 推荐量化

```text
ood_generalization_score =
    0.40 * rank_down(ood_test.nmse)
  + 0.35 * up01(ood_test.acc_0_1)
  + 0.25 * clip(ood_test.r2, 0, 1)
```

### 报告建议

不要把 OOD 和 ID 混成一个数。

原因：

- 很多方法在 ID 上很好，但 OOD 会明显崩
- `LLM-SRBench` 本身就强调 OOD

## 3.4 抗噪鲁棒性

### 定义

衡量训练/测试数据加入噪声后，模型性能下降有多严重。

### 推荐实验协议

对每个任务构造多个噪声等级：

- `σ ∈ {0.01, 0.05, 0.10}`

推荐对目标值加噪：

```text
y_noisy = y + Normal(0, σ * std(y_train))
```

可选地再做输入噪声实验。

### 原始字段

建议新增：

- `noise@0.01.id_test.nmse`
- `noise@0.05.id_test.nmse`
- `noise@0.10.id_test.nmse`
- `noise@0.01.id_test.acc_0_1`
- `noise@0.05.id_test.acc_0_1`
- `noise@0.10.id_test.acc_0_1`

若任务有 GT，可再加：

- `noise@σ.symbolic_accuracy`
- `noise@σ.solution_rate`

### 推荐量化

先定义退化量：

```text
deg_nmse(σ) = max(0, (nmse_σ - nmse_clean) / (nmse_clean + ε))
deg_acc(σ)  = max(0, (acc_clean - acc_σ) / max(acc_clean, ε))
```

再对多个噪声等级取平均：

```text
avg_deg_nmse = mean_σ deg_nmse(σ)
avg_deg_acc  = mean_σ deg_acc(σ)
```

最后定义抗噪分数：

```text
noise_robustness_score =
    0.60 * rank_down(avg_deg_nmse)
  + 0.40 * rank_down(avg_deg_acc)
```

若有 GT，可增加符号项并重新分配权重。

## 3.5 稳定性

### 定义

衡量换随机种子、多次重复实验时，性能和表达式是否稳定。

### 推荐协议

每个方法、每个任务至少跑：

- `K = 5` 个 seed

### 原始字段

建议保留每个 seed 的明细，并聚合：

- `id_test.nmse_mean`
- `id_test.nmse_std`
- `ood_test.nmse_mean`
- `ood_test.nmse_std`
- `complexity_simplified_mean`
- `complexity_simplified_std`
- `symbolic_accuracy_mean`
- `symbolic_accuracy_std`
- `valid_run_rate`

### 推荐量化

先定义：

```text
cv_id_nmse  = id_test.nmse_std  / max(abs(id_test.nmse_mean), ε)
cv_ood_nmse = ood_test.nmse_std / max(abs(ood_test.nmse_mean), ε)
```

再定义稳定性分数：

```text
stability_score =
    0.35 * rank_down(cv_id_nmse)
  + 0.35 * rank_down(cv_ood_nmse)
  + 0.15 * rank_down(complexity_simplified_std)
  + 0.15 * up01(valid_run_rate)
```

如果任务带 GT，也可以额外引入：

- `symbolic_accuracy_std`

## 3.6 符号保真与可解释性

### 定义

同时衡量：

- 学到的公式在符号意义上是否正确
- 公式结构与真值距离有多远
- 公式是否足够简洁、可读

### 原始字段

- `symbolic_solution`
- `solution_rate`
- `symbolic_accuracy`
- `ned`
- `complexity_raw`
- `complexity_simplified`

说明：

- `symbolic_solution` 是单次判定
- `solution_rate` 是在任务集/多次运行上的成功比例
- `symbolic_accuracy` 更接近 `LLM-SRBench` 的语义等价性
- `ned` 是 `SRSD` 的结构距离核心指标
- `complexity_*` 是可解释性的操作性代理

### 推荐量化

先定义复杂度分数：

```text
complexity_score = rank_down(complexity_simplified)
```

然后定义第 6 维分数：

```text
symbolic_interpretability_score =
    0.35 * up01(symbolic_accuracy)
  + 0.25 * up01(solution_rate)
  + 0.20 * (1 - clip(ned, 0, 1))
  + 0.20 * complexity_score
```

### 为什么不把 `NED` 单独拆成一维

因为：

- `NED` 只衡量结构距离
- 它不能单独代表语义是否等价
- 也不能代表公式是否简洁

因此更合理的做法是把它作为“符号保真与可解释性”的一个子指标。

## 4. 六维总分

### 4.1 原则

默认不建议过早使用单一总分替代 6 个维度。

更推荐：

- 主表报告 6 个维度分数
- 附录再提供总分

### 4.2 若必须给总分

推荐等权平均：

```text
overall_6d_score =
    mean(
        time_cost_score,
        id_quality_score,
        ood_generalization_score,
        noise_robustness_score,
        stability_score,
        symbolic_interpretability_score
    )
```

### 4.3 适用范围

只有同时满足以下条件时，才允许计算完整 `overall_6d_score`：

- 有 `ID` 和 `OOD` split
- 做了 noise protocol
- 做了 multi-seed protocol
- 数据集带 `ground truth` 公式

如果不满足，必须：

- 单独报告可用维度
- 明确标记 `N/A`
- 不能把缺维任务和完整 6 维任务直接做总分比较

## 5. 推荐输出结构

建议结果 JSON 结构如下：

```json
{
  "time_cost": {
    "raw": {
      "train_time_seconds": 12.3
    },
    "score": 0.62
  },
  "id_quality": {
    "raw": {
      "rmse": 0.12,
      "r2": 0.98,
      "nmse": 0.03,
      "acc_0_1": 0.91
    },
    "score": 0.89
  },
  "ood_generalization": {
    "raw": {
      "rmse": 0.20,
      "r2": 0.93,
      "nmse": 0.08,
      "acc_0_1": 0.84
    },
    "score": 0.77
  },
  "noise_robustness": {
    "raw": {
      "avg_deg_nmse": 0.31,
      "avg_deg_acc": 0.12
    },
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
  },
  "overall_6d_score": 0.785
}
```

## 6. 与当前仓库字段的映射

当前仓库已经具备直接支持的字段包括：

- `train_time_seconds`
- `id_test.rmse`
- `id_test.r2`
- `id_test.nmse`
- `id_test.acc_0_1`
- `ood_test.rmse`
- `ood_test.r2`
- `ood_test.nmse`
- `ood_test.acc_0_1`
- `complexity_raw`
- `complexity_simplified`
- `symbolic_solution`
- `solution_rate`
- `ned`
- `symbolic_accuracy`

仍建议后续补充：

- noise protocol 结果字段
- multi-seed 聚合字段
- `valid_run_rate`

## 7. 落地建议

建议按以下顺序实现：

1. 先把已有字段接成 6 个维度中的：
   - 时间成本
   - ID 数值质量
   - OOD 泛化能力
   - 符号保真与可解释性
2. 再补：
   - noise protocol
   - multi-seed stability protocol
3. 最后再实现：
   - `overall_6d_score`

这样做的好处是：

- 先把现有仓库能力最大化利用
- 不会因为抗噪与稳定性协议未完工，就阻塞主 benchmark 路径
