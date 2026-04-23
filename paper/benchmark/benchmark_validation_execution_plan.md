# Benchmark 验证阶段执行计划表

## 目标

这份计划不是再解释 `664 → 200 → 100` 的原则，而是把后续真正要做的验证实验落成**可执行任务表**。

核心目标有三个：

1. 证明 `664 → 200 → 100` 这条筛选链不是拍脑袋；
2. 证明未来的 `Core-50` 能代表 `Clean-Master-100`；
3. 在不污染测试集的前提下，冻结 `Dev-50 / Core-50` 并跑出最终 benchmark leaderboard。

## 冻结输入

以下输入在进入本计划后默认冻结，不再边跑边改：

- `Candidate-200`
  - 来源：`/tmp/candidate_seeds_200_v3.json`
- 正式三种子结果
  - `experiment-results/benchmark_formal200_20260417/three_seed_formal_task_results.csv`
  - `experiment-results/benchmark_formal200_20260417/three_seed_formal_dataset_method_summary.csv`
  - `experiment-results/benchmark_formal200_20260417/three_seed_formal_dataset_compare.csv`
- 当前 `Master-100 / Dev-50 / Core-50`
  - `experiment-results/benchmark_formal200_20260417/dev_core_split_v1/`

后续允许新生成：

- `Clean-Master-100`
- `benchmark_dev50_v2`
- `benchmark_core50_v2`

但**不允许**在看到下游实验结果后，再回头改 `Candidate-200` 的成员。

## 算法分层

### 第一层：用于 selection ablation 的 7 个代表性算法

这一步不需要 10 个算法全上，但必须覆盖主要方法家族，尤其要显式覆盖**纯强化学习 SR**：

- `gplearn`：经典 GP 基线
- `pysr`：现代演化/符号回归主力
- `pyoperon`：现代树搜索 EA
- `llmsr`：LLM-based SR
- `drsr`：LLM-hybrid / DSR
- `dso`：纯 RL-based symbolic regression
- `tpsr`：Transformer + planning

这 7 个算法足以覆盖：

- GP / EA
- 纯 RL-based SR
- LLM-based
- Hybrid / DSR
- Transformer-planning

对应的执行与分发建议见：

- [E1_run_matrix.md](./E1_run_matrix.md)

#### `gplearn` 参数口径说明

`gplearn` 在 `E1` 中不再使用“仅四则运算”的极保守配置。

原因是当前 `Candidate-200 / Master-100` 中包含大量科学型任务，真实结构经常涉及：

- `sqrt`
- `log`
- `sin`
- `cos`

如果仍将 `gplearn` 的 `function_set` 限制为：

- `add,sub,mul,div`

那么 `gplearn` 在 `E1` 中会因为表达能力过弱而被系统性低估，进而污染：

- `Current-100` 是否优于其它 `100` 选择策略
- `Core-50` 是否真的代表 `Clean-Master-100`

因此，`E1` 采用的 `gplearn` 口径是：

- 保留官方默认的核心搜索强度参数
  - `population_size = 1000`
  - `tournament_size = 20`
  - `parsimony_coefficient = 0.001`
  - 默认 mutation / crossover 比例
- 但将 `function_set` 扩展为更适合科学型数据集的：
  - `add,sub,mul,div,sqrt,log,sin,cos`

这版配置的定位是：

- **scientific baseline**

而不是：

- 严格官方默认 baseline
- 或 14 个内置 primitive 全开的 aggressive baseline

选择这版口径的理由是：

1. 不再明显削弱 `gplearn`
2. 又不把 `tan / max / min / neg / inv` 这类更激进或冗余 primitive 全部放开
3. 便于在 `E1` 阶段把 `gplearn` 当作真正的 GP 家族代表
4. 同时仍保持：
   - `n_jobs = 1`
   - `timeout_in_seconds = 3600`
   - 以统一 wall-clock 预算参与比较

#### `pyoperon` 参数口径说明

`pyoperon` 在 `E1` 中不再使用“仅给 population / iterations / threads”的稀疏占位配置。

原因是 `Operon / PyOperon` 在 benchmark 文献里更常见的写法，是显式冻结其搜索空间和评估预算，例如：

- `population_size`
- `pool_size`
- `max_length`
- `allowed_symbols`
- `tournament_size`
- `max_evaluations`
- `offspring_generator`
- `reinserter`

因此，`E1` 中采用更接近 benchmark 文献的 `pyoperon` 配置：

- `population_size = 500`
- `pool_size = 500`
- `max_length = 50`
- `max_depth = 10`
- `tournament_size = 5`
- `allowed_symbols = add,mul,aq,exp,log,sin,tanh,constant,variable`
- `offspring_generator = basic`
- `reinserter = keep-best`
- `optimizer = lm`
- `local_search_probability = 1.0`
- `max_evaluations = 500000`
- `n_threads = 1`

这版配置的定位是：

- **接近 Operon benchmark 文献常见口径的 explicit baseline**

而不是：

- 只依赖官方默认值的稀疏 baseline
- 或单纯依赖超大 `generations` + `1h timeout` 的预算占位配置

### 第二层：用于最终 leaderboard 的 10 个算法

- `gplearn`
- `pysr`
- `pyoperon`
- `llmsr`
- `drsr`
- `dso`
- `tpsr`
- `e2esr`
- `QLattice`
- `iMCTS`

## 统一运行口径

除非某个算法后续经验证必须例外，否则统一使用：

- `seed`
  - selection / representativeness 阶段：`1 seed`
  - final leaderboard 阶段：`3 seeds`
- `timeout_in_seconds = 3600`
- `progress_snapshot_interval_seconds = 60`
- 统一记录：
  - `task_status.jsonl`
  - `experiment_dir/result.json`
  - 每分钟快照

说明：

- `3600s` 不是因为它绝对最优，而是因为当前已有：
  - `pysr / llmsr` 三种子正式结果
  - 是按统一 `1h` 口径整理过的
- 为了保证和现有材料可比，后续验证优先沿用这套统一预算。

## 实验计划表

| 实验ID | 名称 | 目的 | 输入 | 算法 | seeds | 预算 | 新任务量 | 主要输出 | 通过/停止准则 |
|---|---|---|---|---|---|---|---:|---|---|
| `E0` | `Clean-Master-100` 清洗 | 去掉 semantic duplicate，统一状态语义，补非结果标签 | `Master-100` | 无新训练 | 无 | 无 | `0` | `clean_master100.csv`、`duplicate_report.csv`、状态重标表 | `basename` 全局唯一；semantic duplicate=0；状态标签统一 |
| `E1` | `100` 选择 ablation 数据生成 | 为 `Current-100 / Gap-only-100 / Quality-first-100 / Metadata-diverse-100` 提供统一评估底座 | `Candidate-200` | 7 算法 | `1` | `1h` | `1400` | `candidate200_7algo_1seed_task_results.csv` | 7 算法在 200 上至少 `95%` 任务有有效结果；否则先修工具再继续 |
| `E2` | `100` 选择策略离线比较 | 证明当前 `Master-100` 不是随便挑的，而是在 discrimination / stability / coverage 上 trade-off 最优 | `E1` 输出 + `Clean-Master-100` | 离线 | 无 | 无 | `0` | `selection_ablation_summary.csv`、`selection_ablation.md` | `Current-100` 不被其它策略在主要指标上严格支配 |
| `E3` | `Clean-Master-100` 轻量全算法验证 | 生成 “50 是否代表 100” 的参考母集 | `Clean-Master-100` | 10 算法 | `1` | `1h` | `300*` | `clean_master100_10algo_1seed_*` | 10 算法中大多数可完成；允许少量超时，但必须有可用输出 |
| `E4` | `50` 代表性离线验证 | 比较 `Core-50` 与其它 `50` 选法是否最能代表 `Clean-Master-100` | `E3` 输出 | 离线 | 无 | 无 | `0` | `core50_representativeness.csv`、相关性图、bootstrap CI 图 | `Core-50` 在 `Spearman/Kendall/pairwise agreement` 上优于基线 `50` |
| `E5` | 冻结 `Dev-50 / Core-50` | 基于 `Clean-Master-100` 的非结果属性重新切分并冻结正式集合 | `Clean-Master-100` | 无新训练 | 无 | 无 | `0` | `benchmark_dev50_v2.csv`、`benchmark_core50_v2.csv`、`split_audit_report.md` | family 精确半分或差值 `<=1`；`Core one-sided <=5`；静态属性分布匹配 |
| `E6` | 最终 leaderboard | 在冻结后的 `Core-50` 上跑论文主榜 | `benchmark_core50_v2.csv` | 10 算法 | `3` | `1h` | `1500` | `leaderboard_core50.csv`、radar、Pareto、family breakdown 图 | 所有算法结果收齐；主榜和分解图可复现 |
| `E7` | robustness ablation | 回答 reviewer 会问的“是不是被某类题主导了” | `E6` 输出 | 离线 | 无 | 无 | `0` | leave-one-family-out、去掉 `one-sided`、去掉 `srsd` 的对照表和图 | 排名主结论不应在单一 ablation 下完全翻转 |

> `E3` 的新任务量写成 `300*`，因为它默认复用 `E1` 里已经在 `Candidate-200` 上跑过的 7 算法结果。  
> 对 `Clean-Master-100` 而言，新增只需要补：
> - 剩余 `3` 个算法
> - `3 × 100 × 1 seed = 300` 个新任务  
> 若复用失败，则 `E3` 上限为 `1000` 个新任务。

## 每个实验的详细交付物

### `E0`：`Clean-Master-100`

必须额外交付：

- `canonical_formula_hash`
- `structure_hash`
- `semantic_duplicate_group`
- `status_semantics`
  - `ok_full`
  - `budget_exhausted_with_output`
  - `partial_output`
  - `no_valid_output`
- 静态标签：
  - `family`
  - `subgroup`
  - `feature_count`
  - `train/valid/id/ood samples`
  - `formula_operator_count`
  - `dummy / non-dummy`
  - `source benchmark`

#### `E0` 的方法学边界

`E0` 的默认定位不是：

- **“从 `Candidate-200` 里重新挑一遍 `100`”**

而是：

- **“对当前 `Master-100` 做最小扰动清洗，并在必要时局部补位”**

也就是说，`E0` 的输入固定是当前这版：

- `experiment-results/benchmark_formal200_20260417/dev_core_split_v1/master100_candidates.csv`

`E0` 的职责是把这版 `Master-100` 清洗成：

- `Clean-Master-100`

而不是重新发明一套新的 `200 -> 100` 选择器。

#### 为什么 `E0` 不能直接重选 100

如果 `E0` 一上来就重新从 `Candidate-200` 里整套选 `100`，会导致两个问题：

1. **清洗和重选题边界混淆**
   - 后面无法清楚回答：到底是在“修正坏样本”，还是在“偷偷改 benchmark”。

2. **测试链路不可追溯**
   - 一旦 `Master-100` 的成员大范围变化，后续 `E2 / E4` 的解释都会变弱。

因此 `E0` 采用：

- **最小扰动原则**

即：

1. 先把当前 `Master-100` 当作基线；
2. 只删除明确不该保留的样本；
3. 再从 `Candidate-200 \\ Master-100` 里做等量补位。

#### `E0` 的标准工作流

1. **冻结当前 `Master-100`**
   - 不先动 `Candidate-200`
   - 不先重算 `priority_score`

2. **对整个 `Candidate-200` 做 semantic dedup 审计**
   - 因为后面补位也要从剩余 `Candidate-200` 中挑
   - 所以必须先在全局范围知道哪些题本质上是同一公式

3. **在当前 `Master-100` 中标记必须删除的样本**
   - 这一步默认只删：
     - semantic duplicate
     - 明确坏样本 / 结构性异常样本

4. **在剩余 `Candidate-200` 中做局部补位**
   - 每删一个补一个
   - 优先保持原来的：
     - `family`
     - `subgroup`
     - `selection_mode`
     - `candidate_advantage_side`
     - 静态属性分布

5. **输出 `Clean-Master-100` 并冻结**
   - 之后的 `E1~E7` 都只基于这版 clean 结果继续做

#### 哪些样本应该在 `E0` 阶段删除

默认只删除两类：

1. **semantic duplicate**
   - 同一公式在不同 benchmark 源中的重复出现
   - 例如跨来源 `feynman-i.*` / `feynman_I_*` 这类本质同题样本

2. **结构性坏样本**
   - 元数据损坏
   - 公式无法 canonicalize
   - 数据目录异常
   - 无法进入后续切分与评估流程

#### 哪些理由不属于 `E0` 的删除条件

以下问题不在 `E0` 阶段处理：

- `quality_score` 不够高
- `strict` 太多
- `srsd` 太多
- `PySR` 优势太强
- 某个算法单独在这题上表现差

这些属于：

- `E2` 的 selection ablation
- 或 `E4` 的 representativeness validation

而不是 `E0` 的清洗职责。

#### duplicate group 里保留谁

如果一个 duplicate group 里有多个样本，保留顺序为：

1. `priority_score` 更高
2. `quality_score` 更高
3. `stability_score` 更高
4. `selection_mode` 更优先保留 `strict / mid-gap`
5. 静态属性更规范、元数据更完整
6. 若仍相同，再人工选择 benchmark 来源更规范的一项

#### 补位时如何从 `Candidate-200` 里选替补

替补不是简单选“下一个 `priority_score` 最高”，而是做**约束下最近邻补位**。

优先满足：

1. 同 `family`
2. 同 `subgroup`
3. 同 `selection_mode`
4. 同 `candidate_advantage_side`
5. `feature_count` 接近
6. `train / id / ood` 样本量接近
7. `formula_operator_count` 接近
8. `priority_score` 尽量高
9. 不与当前 `Clean-Master-100` 形成新的 duplicate

#### `E0` 必须产出的文件

- `clean_master100.csv`
- `duplicate_groups.csv`
- `replacement_log.csv`
- `status_semantics_map.csv`
- `clean_master100_audit.md`

这些文件分别回答：

- 哪些题被判成同组重复；
- 哪些题被删、删的理由是什么；
- 用谁补了回来、为什么是它；
- 清洗前后 `family / subgroup / selection_mode / advantage` 分布有没有漂移。

#### 一句话定义

> `E0 = 当前 Master-100 的最小扰动清洗与补位，不是重新从 200 里整套挑一遍 100。`

### `E1`：selection ablation 数据生成

运行对象固定是 `Candidate-200`，因为：

- 不同 `100` 版本都只是 `Candidate-200` 的子集；
- 先在 `200` 上把 7 算法跑齐，后面比较不同 `100` 版本就能完全离线完成。

这一步的关键图表：

- 各算法在 `200` 上的完成率 / 有效率条形图
- family × algorithm 的覆盖热图

### `E2`：`100` 选择策略离线比较

要比较的 `100` 版本：

1. `Current-100`
2. `Gap-only-100`
3. `Quality-first-100`
4. `Metadata-diverse-100`

比较指标：

- 排名区分度
- 排名稳定性
- family / subgroup 覆盖
- 方法家族排序一致性

建议图表：

- 4 个 `100` 版本的 family 覆盖条形图
- 排名相关热图
- 各策略 aggregate score 的误差箱线图

### `E3`：`Clean-Master-100` 轻量全算法验证

这是后续 `50` 代表性验证的母集。

建议输出：

- `dataset × algorithm` 的单 seed 聚合表
- 每个算法在 `100` 上的：
  - `ID/OOD NMSE`
  - `ID/OOD R²`
  - symbolic fidelity
  - complexity
  - time / timeout ratio

### `E4`：`50` 是否代表 `100`

离线比较这些 `50`：

1. `Core-50`
2. `random-50`（`20` 次）
3. `family-stratified random-50`（`20` 次）
4. `gap-top50`
5. `metadata-diverse-50`

主指标：

- `Spearman ρ`
- `Kendall τ`
- `pairwise win agreement`
- `aggregate score error`
- `family-wise leaderboard drift`

建议输出：

- `Core-50` vs 其它 `50` 的相关性箱线图
- `pairwise agreement` 条形图
- `bootstrap 95% CI` 误差条图

### `E5`：冻结 `Dev-50 / Core-50`

切分规则保持：

- 切分时**不用正式结果**
- 只用：
  - `family`
  - `subgroup`
  - `selection_mode`
  - `candidate_advantage_side`
  - `basename`
  - `feature_count`
  - 各 split 样本量
  - 公式静态复杂度

验收标准：

- family 配额精确半分或差值 `<=1`
- `Core-50 one-sided <= 5`
- `basename <= 1` 在整个 `Clean-Master-100` 上成立
- 静态属性的分布匹配报告齐全

### `E6`：最终 leaderboard

最终主榜在 `Core-50` 上跑：

- `10 algorithms × 50 datasets × 3 seeds = 1500 tasks`

建议输出图：

- 总 leaderboard
- 六维雷达图
- Pareto 图
- family / subgroup breakdown
- 稳定性（跨 seed 方差）图

### `E7`：robustness

至少做这 4 组：

1. 去掉 `one-sided`
2. 去掉 `srsd`
3. `leave-one-family-out`
4. 只保留 `quality_score = 1.0`

目的不是追求完全不变，而是证明：

- 主结论不是被某一来源或某一类“异常题”单独驱动的。

## 推荐执行顺序

### Phase A：不新增太多算力，先把 benchmark 设计站稳

1. `E0` 清洗 `Master-100`
2. `E1` 跑 `Candidate-200 × 6 algorithms × 1 seed`
3. `E2` 离线做 `100` 选择 ablation

### Phase B：验证 `50` 是否能代表 `100`

4. `E3` 补齐 `Clean-Master-100 × 10 algorithms × 1 seed`
5. `E4` 离线比较不同 `50`
6. `E5` 冻结 `Dev-50 / Core-50`

### Phase C：跑最终榜单

7. `E6` 在 `Core-50` 上跑 `10 algorithms × 3 seeds`
8. `E7` 做 robustness ablation

## 预计新增任务量

按“能复用 `E1` 结果”的主方案估算：

- `E0`: `0`
- `E1`: `1400`
- `E2`: `0`
- `E3`: `300`
- `E4`: `0`
- `E5`: `0`
- `E6`: `1500`
- `E7`: `0`

总计新增：

- **`3200` 个训练任务**

如果 `E3` 无法有效复用 `E1` 的 7 算法结果，则上限变成：

- **`3900` 个训练任务**

## 最小停止准则

### 可以提前停止并进入下一阶段的条件

- `E0` 完成且 semantic duplicate 清零；
- `E2` 证明 `Current-100` 不被替代策略严格支配；
- `E4` 证明 `Core-50` 在代表性指标上优于随机与简单 top-k 基线；

只要这三条成立，就可以冻结 `Core-50` 并进入最终 leaderboard 阶段。

### 必须回滚重做的条件

- `Clean-Master-100` 仍存在 semantic duplicate；
- `Current-100` 在 selection ablation 中被简单策略系统性击败；
- `Core-50` 无法稳定代表 `Clean-Master-100`（例如排名相关显著低于随机分层基线）。

## 建议配套输出文件

建议最终形成下面这些文件，方便论文和复现实验同时引用：

- `paper/benchmark/selection_ablation.md`
- `paper/benchmark/representative50_validation.md`
- `paper/benchmark/final_core50_leaderboard.md`
- `experiment-results/benchmark_clean_master100_*`
- `experiment-results/core50_validation_*`
- `experiment-results/core50_leaderboard_*`

## 一句话版本

这份计划的核心不是“赶紧定 50”，而是：

> **先证明 `100` 选得合理，再证明 `50` 能代表 `100`，最后才冻结 `Core-50` 跑正式榜单。**
