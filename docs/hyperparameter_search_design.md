# 超参数搜索设计（基于当前SIM框架）

目标：把每个工具“谁可改、改什么、改多少”统一成可自动化搜索的配置策略，先能跑起来，再能放大到 AutoML/Optuna。

> 前提：当前仓库中很多 wrapper 采用 `**kwargs` 透传。若某工具未显式限制白名单（如 gplearn/pysr/pyoperon），建议先做参数校验清单，再加白名单，否则会把无效参数静默下发。

## 1. 各工具重要超参数（适合放进搜索）

以下按“当前wrapper可见能力”给出；对 `gplearn/pysr/pyoperon` 结尾的参数列表按上游官方文档进一步对齐。

### gplearn（当前 `gplearn_wrapper`）

1. `population_size`（优先高）  
2. `generations`（优先高）  
3. `tournament_size`  
4. `p_crossover`、`p_subtree_mutation`、`p_hoist_mutation`、`p_point_mutation`  
5. `parsimony_coefficient`（控制表达式复杂度）  
6. `max_samples`、`const_range`、`function_set`

推荐搜索（初版）：
- `population_size`: [200, 2000]（对数）
- `generations`: [30, 250]（整数）
- `tournament_size`: [10, 50]（整数）
- `parsimony_coefficient`: [1e-6, 0.1]（对数）
- `p_crossover`: [0.6, 0.95]
- `p_point_mutation`/`p_subtree_mutation`/`p_hoist_mutation`: `[0.0, 0.4]`，并且约束和为 1.0 左右（剩余由点替换项补齐）

### pysr（`pysr_wrapper`）

1. `niterations`（高）  
2. `population_size`、`populations`（高）  
3. `ncycles_per_iteration`（中）  
4. `maxsize`、`maxdepth`（高）  
5. `parsimony` / `procs`（高）  
6. `binary_operators`、`unary_operators`（高，结构性搜索）

推荐搜索：
- `niterations`: [100, 3000]（对数/指数步）
- `populations`: [4, 20]（整数）
- `population_size`: [50, 800]
- `ncycles_per_iteration`: [50, 1500]
- `maxsize`: [8, 60]
- `maxdepth`: [3, 12]
- `parsimony`: [1e-6, 0.05]（对数）

### pyoperon（`pyoperon_wrapper`）

当前 wrapper 为 `**kwargs` 透传，建议先做一次接口对齐再大规模搜索。先从以下关键类参数入手：
1. `population_size`/`generations`/`n_generations`（按官方签名确认）
2. `random_state`、`n_threads`
3. `operator_set`（函数集）
4. `initialization`/`crossover`/`mutation` 相关参数（按上游签名确认）

### llmsr（`llmsr_wrapper`）

当前实现已明确支持：
1. `max_params`（关键）  
2. `niterations`（关键）  
3. `samples_per_iteration`（关键）  
4. `anonymize`（实验可控性）  
5. `llm_config_path`（模型/temperature/top_p/max_tokens，放在 llm.config）

推荐搜索：
- `max_params`: [2, 20]（整数）
- `niterations`: [200, 3000]
- `samples_per_iteration`: [1, 8]
- `anonymize`: `[true, false]`
- `llm.config`: 先固定模型做外部参数搜索，再考虑模型级搜索（如 `temperature: 0.2~1.0`, `top_p: 0.2~1.0`）

### dso（`dso_wrapper`）

目前 wrapper 直接透传给 `DeepSymbolicRegressor(config=...)`，`**kwargs` 方案不稳。建议先固定策略：
1. 搜索 `config` 文件（预置模板）而非散参数
2. 在 `dso/dso/dso/config/config_regression.json` + `config_common.json` 中优先搜：
   - `training.n_samples`、`training.batch_size`、`training.epsilon`
   - `policy.max_length`、`policy.num_layers`
   - `policy_optimizer.learning_rate`、`entropy_weight`、`entropy_gamma`
   - `gp_meld.run_gp_meld`（二值）与 `gp_meld.p_crossover`、`gp_meld.p_mutate`
   - `task.function_set`、`task.protected`、`prior` 的若干开关

推荐搜索（先粗后细）：
- `training.n_samples`: [100k, 3M]
- `training.batch_size`: [128, 2000]
- `training.epsilon`: [0.01, 0.3]
- `policy.max_length`: [16, 128]
- `policy.num_layers`: [1, 4]
- `gp_meld.run_gp_meld`: `[true, false]`

### tpsr（`tpsr_wrapper`）

wrapper 当前显式生效参数与子参数（从 `parsers.py`）：
1. `horizon`（关键）  
2. `rollout`（关键）  
3. `num_beams`（关键）  
4. `width`（关键）  
5. `beam_size`、`beam_type`、`backbone_model`
6. `max_input_points`、`n_trees_to_refine`、`rescale`

推荐搜索：
- `width`: [1, 8]
- `num_beams`: [1, 8]
- `rollout`: [1, 12]
- `horizon`: [80, 400]
- `beam_size`: [2, 40]
- `beam_type`: `[sampling, search]`
- `no_seq_cache`: [true, false]
- `no_prefix_cache`: [true, false]
- `backbone_model`: `["e2e"]`（先固定；`nesymres` 兼容性另开分支验证）

### e2esr（`e2esr_wrapper`）

1. `max_input_points`  
2. `n_trees_to_refine`  
3. `rescale`  
4. `model_path`、`model_url`（模型源策略）
5. `SymbolicTransformerRegressor.set_args` 可扩展项（当前实现与 `tpsr` 核心一致）

推荐搜索：
- `max_input_points`: [1000, 50000]（对数）
- `n_trees_to_refine`: [1, 20]
- `rescale`: [true, false]

### QLattice（`QLattice_wrapper`）

1. `n_epochs`（关键）  
2. `max_complexity`（关键）  
3. `threads`（性能关键）  
4. `stypes`、`function_names`（模型空间关键）  
5. `criterion`、`query_string`

推荐搜索：
- `n_epochs`: [20, 500]
- `max_complexity`: [6, 30]
- `threads`: [1, 8] 或机器核数上限
- `criterion`: `["bic","aic","rmse","mae"]`（按版本）
- `query_string`: 先固定空字符串，待稳定后再做启发式网格

### iMCTS（`iMCTS_wrapper`）

1. `K`（关键）  
2. `max_depth`（关键）  
3. `c`、`gamma`  
4. `gp_rate`、`mutation_rate`、`exploration_rate`
5. `max_constants`、`max_single_arity_ops`、`max_expressions`

推荐搜索：
- `max_depth`: [3, 10]
- `K`: [100, 5000]（对数）
- `c`: [0.5, 8]
- `gamma`: [0.0, 1.0]
- `gp_rate`: [0.0, 0.9]
- `mutation_rate`: [0.0, 0.7]
- `exploration_rate`: [0.0, 0.8]
- `max_constants`: [2, 30]
- `ops`: 运算集版本（如 `['+','-','*','/','sin','cos','exp','log']` vs 增加 `tan`,`sqrt`）

### drsr（`drsr_wrapper`）

1. `max_samples`（关键，硬上限）  
2. `samples_per_prompt`（关键，批大小）  
3. `evaluate_timeout_seconds`（关键，稳定性）  
4. `use_api`、`api_model`  
5. `api_params`（温度、top_p、max_tokens 等可按 `api_model` 分类）

推荐搜索：
- `max_samples`: [20, 500]
- `samples_per_prompt`: [1, 6]
- `evaluate_timeout_seconds`: [5, 120]
- `api_model`: 固定单模型，待稳定后再做列表切换
- `temperature`（如使用 wrapper 直传）: [0.1, 1.0]
- `seed`: 固定为每轮复现的几组离散值

## 2. 建议的搜索策略（落地）

### 2.1 统一“预算优先”策略
- 每个算法先做 **3~5 组固定先验**（小试）确认可运行范围。
- 再做 **随机搜索/贝叶斯搜索**（每算法建议 30~100 试验）。
- 对随机性算法（gplearn/pysr/tpsr/iMCTS/LLM类）每个配置建议 3 个种子后取中位数。
- 时间预算约束：`trial_timeout` 强制上限（例如 10~30 分钟），超时直接 `failed`。

### 2.2 评价函数
- 标准任务：`score = α*NMSE + β*log(1+时间) + γ*复杂度罚项`  
- 复杂度罚项：
  - GP类：`parsimony`/树深度/长度相关 proxy（如表达式长度）
  - LLM类：可用采样数、长度、AIC/BIC 近似

### 2.3 自动化落地建议
- 用 `run-pipeline` 的 `--params-json` 做参数注入（推荐）。
- 新增配置文件 `configs/hpo_search_space.yaml`（自定义每算法搜索域）。
- 每条试验执行命令统一化：
  - `python -m scientific_intelligent_modelling.cli run-pipeline --dataset-dir ... --tool-name XXX --params-json '{...}' --seed ...`
- 结果落表字段建议：`tool, seed, params_hash, train_time, val_rmse, val_r2, n_eq, best_eq_len, error_code`

## 3. 直接可执行的下一步

1. 我先给你一版 `configs/hpo_space.json`，把上面的建议范围编码成机器可读 schema。  
2. 再给 `--params-json` 做输入校验（字段白名单 + 类型检查），避免无效参数导致子进程崩溃。  
3. 先在 2~3 个算法做试验模板（gplearn/pysr/tpsr），跑通后批量迁移到全量 10 个工具。
