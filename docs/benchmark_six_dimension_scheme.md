# 六维 Benchmark 中文解释版

## 1. 这份方案在做什么

这份方案只回答两件事：

1. 每个大类指标由哪些 `raw` 子指标组成
2. 这些 `raw` 子指标如何汇总成一个最终分数

这里的 6 个大类分别是：

1. 搜索效率
2. ID 数值质量
3. OOD 泛化能力
4. 抗噪鲁棒性
5. 稳定性
6. 符号保真与可解释性

## 2. 基础函数

为了把不同量纲的指标统一到 `0 ~ 1` 区间，先定义 4 个基础函数。

### 2.1 截断函数

公式：

```text
clip01(x) = clip(x, 0, 1)
```

中文解释：

- 如果 `x < 0`，就按 `0` 记
- 如果 `0 <= x <= 1`，就按原值记
- 如果 `x > 1`，就按 `1` 记

这个函数适合本来就在 `0 ~ 1` 范围里的指标，例如：

- `r2`
- `acc_0_1`
- `solution_rate`
- `symbolic_accuracy`

### 2.2 下降型绝对评分函数

公式：

```text
inv_pos(x) = 1 / (1 + max(x, 0))
```

中文解释：

- `x` 越小，分数越高
- `x = 0` 时，分数是 `1`
- `x` 越大，分数越接近 `0`

这个函数适合“越小越好”的非负指标，例如：

- `nmse`
- 噪声退化率
- 波动系数 `cv`

举例：

- 当 `x = 0` 时，`inv_pos(x) = 1`
- 当 `x = 1` 时，`inv_pos(x) = 0.5`
- 当 `x = 4` 时，`inv_pos(x) = 0.2`

也就是说，误差或退化越大，得分就越低。

### 2.3 复杂度评分函数

公式：

```text
count_score(c) = 1 / (1 + log(1 + max(c, 0)))
```

中文解释：

- `c` 表示复杂度或复杂度波动
- `c` 越小，分数越高
- 这里用了 `log`，是为了避免复杂度稍微变大时，分数掉得过猛

这个函数适合：

- `complexity_simplified`
- `complexity_simplified_std`

一句话理解：

复杂度越低越好，但不希望对复杂度做过于激进的惩罚。

### 2.4 命中速度评分函数

公式：

```text
hit_score(t, B) = clip(1 - t / B, 0, 1)
```

中文解释：

- `t` 表示达到某个目标花了多少秒
- `B` 表示总时间预算
- 如果一开始就达到，分数接近 `1`
- 如果刚好在预算结束时达到，分数接近 `0`
- 如果超过预算还没达到，也记 `0`

这个函数适合：

- `time_to_first_valid_seconds`
- `time_to_target_seconds`

一句话理解：

在同样时间预算下，越早达到目标，分数越高。

## 3. 六个大类指标

## 3.1 搜索效率

### raw 子指标

- `wall_clock_budget_seconds`
- `time_to_first_valid_seconds`
- `time_to_target_seconds`
- `anytime_best_score_auc`

这些字段的中文意思是：

- `wall_clock_budget_seconds`：总共允许跑多少秒
- `time_to_first_valid_seconds`：第一次产出合法公式用了多少秒
- `time_to_target_seconds`：第一次达到目标效果用了多少秒
- `anytime_best_score_auc`：在整个搜索过程中，“当前最好结果”的整体表现有多好

### 公式

```text
first_valid_score  = hit_score(time_to_first_valid_seconds, wall_clock_budget_seconds)
target_reach_score = hit_score(time_to_target_seconds, wall_clock_budget_seconds)
anytime_score      = clip01(anytime_best_score_auc)

search_efficiency_score =
    0.35 * first_valid_score
  + 0.35 * target_reach_score
  + 0.30 * anytime_score
```

### 公式中文解释

先看前三项：

- `first_valid_score`：越早找到一个合法公式，分越高
- `target_reach_score`：越早达到预设目标效果，分越高
- `anytime_score`：整个搜索过程中的“随时间持续进步能力”，越高越好

最后总分：

- `35%` 看谁更快找到合法公式
- `35%` 看谁更快达到目标效果
- `30%` 看整个搜索过程是否持续有效

一句话理解：

这个维度不是比“跑了多久”，而是比“在同样时间预算内推进得有多快”。

### 特殊情况

- 如果任务没有定义 `target` 阈值，就去掉 `target_reach_score`
- 其余权重按比例重分配

## 3.2 ID 数值质量

### raw 子指标

- `id_test.rmse`
- `id_test.nmse`
- `id_test.r2`
- `id_test.acc_0_1`

这些字段的中文意思是：

- `id_test.rmse`：分布内测试集上的绝对误差
- `id_test.nmse`：分布内测试集上的归一化误差
- `id_test.r2`：分布内测试集上的拟合优度
- `id_test.acc_0_1`：分布内测试集上，误差落在 `0.1` 容忍范围内的比例

### 公式

```text
id_nmse_score = inv_pos(id_test.nmse)
id_r2_score   = clip01(id_test.r2)
id_acc_score  = clip01(id_test.acc_0_1)

id_quality_score =
    0.40 * id_nmse_score
  + 0.35 * id_acc_score
  + 0.25 * id_r2_score
```

### 公式中文解释

先看前三项：

- `id_nmse_score`：`nmse` 越小，分越高
- `id_r2_score`：`r2` 越接近 `1`，分越高
- `id_acc_score`：`acc_0_1` 越高，分越高

最后总分：

- `40%` 看归一化误差
- `35%` 看容忍误差下的命中比例
- `25%` 看整体拟合优度

一句话理解：

这个维度衡量“在正常测试集上预测得准不准”。

### 备注

- `rmse` 保留展示，但不进入聚合公式
- 如果缺少 `acc_0_1`，其余项权重按比例重分配

## 3.3 OOD 泛化能力

### raw 子指标

- `ood_test.rmse`
- `ood_test.nmse`
- `ood_test.r2`
- `ood_test.acc_0_1`

这些字段的中文意思是：

- `ood_test.rmse`：分布外测试集上的绝对误差
- `ood_test.nmse`：分布外测试集上的归一化误差
- `ood_test.r2`：分布外测试集上的拟合优度
- `ood_test.acc_0_1`：分布外测试集上，误差落在 `0.1` 容忍范围内的比例

### 公式

```text
ood_nmse_score = inv_pos(ood_test.nmse)
ood_r2_score   = clip01(ood_test.r2)
ood_acc_score  = clip01(ood_test.acc_0_1)

ood_generalization_score =
    0.40 * ood_nmse_score
  + 0.35 * ood_acc_score
  + 0.25 * ood_r2_score
```

### 公式中文解释

先看前三项：

- `ood_nmse_score`：OOD 误差越小，分越高
- `ood_r2_score`：OOD 拟合优度越高，分越高
- `ood_acc_score`：OOD 命中率越高，分越高

最后总分：

- `40%` 看 OOD 误差
- `35%` 看 OOD 命中率
- `25%` 看 OOD 拟合优度

一句话理解：

这个维度衡量“换了分布以后，模型还能不能保持效果”。

### 备注

- `rmse` 保留展示，但不进入聚合公式

## 3.4 抗噪鲁棒性

### raw 子指标

- `noise@0.01.id_test.nmse`
- `noise@0.05.id_test.nmse`
- `noise@0.10.id_test.nmse`
- `noise@0.01.id_test.acc_0_1`
- `noise@0.05.id_test.acc_0_1`
- `noise@0.10.id_test.acc_0_1`
- `id_test.nmse`
- `id_test.acc_0_1`

这些字段的中文意思是：

- 前 6 项是不同噪声强度下的表现
- `id_test.nmse` 和 `id_test.acc_0_1` 是无噪声时的基准表现

### 公式

```text
deg_nmse(σ) = max(0, (nmse_σ - nmse_clean) / (nmse_clean + ε))
deg_acc(σ)  = max(0, (acc_clean - acc_σ) / max(acc_clean, ε))

avg_deg_nmse = mean_σ deg_nmse(σ)
avg_deg_acc  = mean_σ deg_acc(σ)

noise_nmse_score = inv_pos(avg_deg_nmse)
noise_acc_score  = inv_pos(avg_deg_acc)

noise_robustness_score =
    0.60 * noise_nmse_score
  + 0.40 * noise_acc_score
```

### 公式中文解释

先看退化量：

- `deg_nmse(σ)`：加噪以后，`nmse` 相对恶化了多少
- `deg_acc(σ)`：加噪以后，`acc` 相对下降了多少

再取平均：

- `avg_deg_nmse`：所有噪声强度下，误差平均恶化程度
- `avg_deg_acc`：所有噪声强度下，命中率平均下降程度

最后总分：

- `60%` 看误差退化
- `40%` 看命中率退化

一句话理解：

这个维度衡量“数据一变脏，模型掉得厉不厉害”。

## 3.5 稳定性

### raw 子指标

- `id_test.nmse_mean`
- `id_test.nmse_std`
- `ood_test.nmse_mean`
- `ood_test.nmse_std`
- `complexity_simplified_std`
- `valid_run_rate`

这些字段的中文意思是：

- `*_mean`：多次运行后的平均水平
- `*_std`：多次运行后的波动大小
- `complexity_simplified_std`：多次运行后，公式复杂度漂不漂
- `valid_run_rate`：多次运行里，有多少比例能成功产出合法公式

### 公式

```text
cv_id_nmse  = id_test.nmse_std  / max(abs(id_test.nmse_mean), ε)
cv_ood_nmse = ood_test.nmse_std / max(abs(ood_test.nmse_mean), ε)

stab_id_score         = inv_pos(cv_id_nmse)
stab_ood_score        = inv_pos(cv_ood_nmse)
stab_complexity_score = count_score(complexity_simplified_std)
stab_valid_score      = clip01(valid_run_rate)

stability_score =
    0.35 * stab_id_score
  + 0.35 * stab_ood_score
  + 0.15 * stab_complexity_score
  + 0.15 * stab_valid_score
```

### 公式中文解释

先看前两项：

- `cv_id_nmse`：ID 误差的相对波动，越小越稳定
- `cv_ood_nmse`：OOD 误差的相对波动，越小越稳定

再看后两项：

- `stab_complexity_score`：公式复杂度波动越小，分越高
- `stab_valid_score`：合法运行比例越高，分越高

最后总分：

- `35%` 看 ID 波动
- `35%` 看 OOD 波动
- `15%` 看复杂度波动
- `15%` 看合法运行率

一句话理解：

这个维度衡量“同一个方法多跑几次，会不会忽好忽坏”。

## 3.6 符号保真与可解释性

### raw 子指标

- `symbolic_solution`
- `solution_rate`
- `symbolic_accuracy`
- `ned`
- `complexity_raw`
- `complexity_simplified`

这些字段的中文意思是：

- `symbolic_solution`：某一次运行是否找到了正确符号解
- `solution_rate`：多次运行中，找到正确符号解的比例
- `symbolic_accuracy`：公式在数学意义上有多接近真值
- `ned`：公式结构与真值结构的距离，越小越好
- `complexity_raw`：原始公式复杂度
- `complexity_simplified`：简化后公式复杂度

### 公式

```text
symbolic_acc_score  = clip01(symbolic_accuracy)
solution_rate_score = clip01(solution_rate)
ned_score           = 1 - clip01(ned)
complexity_score    = count_score(complexity_simplified)

symbolic_interpretability_score =
    0.35 * symbolic_acc_score
  + 0.25 * solution_rate_score
  + 0.20 * ned_score
  + 0.20 * complexity_score
```

### 公式中文解释

先看前四项：

- `symbolic_acc_score`：数学等价性越高，分越高
- `solution_rate_score`：找到正确符号解的比例越高，分越高
- `ned_score`：结构距离越小，分越高
- `complexity_score`：简化后公式越简单，分越高

最后总分：

- `35%` 看数学层面的正确性
- `25%` 看多次运行的成功率
- `20%` 看结构是否接近真公式
- `20%` 看公式是否足够简洁

一句话理解：

这个维度衡量“公式本身是不是对的、像不像真的、好不好读”。

### 备注

- `symbolic_solution` 保留为单次运行标签，不直接进入聚合公式
- `complexity_raw` 保留展示，但聚合时优先使用 `complexity_simplified`
