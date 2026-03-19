# 工具接口与参数汇总

本文档以当前仓库代码实现为准，目标是回答两件事：

1. 统一入口 `SymbolicRegressor` 对外暴露了什么接口
2. 10 个已接入工具各自真正消费哪些参数、返回哪些结果

适用范围：

- `gplearn`
- `pysr`
- `pyoperon`
- `llmsr`
- `dso`
- `tpsr`
- `e2esr`
- `QLattice`
- `iMCTS`
- `drsr`


## 1. 统一入口

主入口类：

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
```

统一构造方式：

```python
reg = SymbolicRegressor(
    tool_name="gplearn",
    problem_name="demo",
    experiments_dir="./experiments",
    seed=1314,
    **tool_kwargs,
)
```

统一公共参数：

- `tool_name`
  - 工具名，必须出现在 `scientific_intelligent_modelling/config/toolbox_config.json` 的 `tool_mapping` 中。
- `problem_name`
  - 问题名，用于实验目录命名。
- `experiments_dir`
  - 实验根目录，默认是当前工作目录下的 `./experiments`。
- `seed`
  - 全局随机种子。
- `**tool_kwargs`
  - 透传给具体 wrapper 的参数。

框架会自动向 wrapper 注入这些元参数：

- `exp_path`
- `exp_name`
- `problem_name`
- `seed`


## 2. 统一公共方法

所有工具都通过下面 4 个主方法暴露统一能力：

### `fit(X, y)`

- 输入：
  - `X`: 二维特征矩阵
  - `y`: 一维目标向量
- 返回：
  - `self`

### `predict(X)`

- 输入：
  - `X`: 二维特征矩阵
- 返回：
  - `numpy.ndarray`

### `get_optimal_equation()`

- 返回：
  - `str`
- 语义：
  - 返回当前模型的“最佳方程”字符串。

### `get_total_equations()`

- 返回：
  - `list`
- 语义：
  - 返回候选方程列表。

额外方法：

### `get_fitted_params()`

- 不是所有工具都支持。
- 当前主实现里，实际可靠支持的是 `drsr`。

### `get_total_equations_with_params(n=None)`

- 不是所有工具都支持。
- 当前主实现里，实际可靠支持的是 `drsr`。

接口注意事项：

- 所有方法实际都是通过子进程调用对应 wrapper。
- `get_total_equations(n)` 虽然 `SymbolicRegressor` 定义了 `n` 参数，但当前主控层没有把它透传到子进程。
  - 结论：当前不要依赖 `SymbolicRegressor.get_total_equations(n)` 的截断行为。
  - 若某个 wrapper 原生支持 `n`，那是 wrapper 内部能力，不等于统一入口已完整暴露。


## 3. CLI 参数注入方式

CLI 入口：

```bash
python -m scientific_intelligent_modelling.cli -a <algorithm> -t <train_path> ...
```

参数注入规则：

- 已声明公共参数由 CLI 直接解析：
  - `-a/--algorithm`
  - `-t/--train-path`
  - `--seed`
  - `--dataset-name`
  - `--prompts-type`
  - `--use_wandb`
  - `--wandb_*`
- 未声明的长选项会透传到具体 wrapper：
  - `--key value`
  - `--key=value`
  - `--flag` 会被解析为 `True`
- 短选项不会透传。


## 4. 工具与环境映射

来自 `scientific_intelligent_modelling/config/toolbox_config.json`：

- `gplearn` -> `sim_base`
- `pysr` -> `sim_base`
- `pyoperon` -> `sim_base`
- `llmsr` -> `sim_llm`
- `dso` -> `sim_dso`
- `tpsr` -> `sim_tpsr`
- `e2esr` -> `sim_e2esr`
- `QLattice` -> `sim_qLattice`
- `iMCTS` -> `sim_iMCTS`
- `drsr` -> `sim_llm`


## 5. 各工具接口细节

### 5.1 `gplearn`

Wrapper：

- `scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py`
- 类名：`GPLearnRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

参数策略：

- 显式白名单
- 会剥离元参数：`exp_name`、`exp_path`、`problem_name`、`seed`
- 若用户未传 `random_state`，则自动执行 `seed -> random_state`

允许参数：

- `population_size`
- `generations`
- `tournament_size`
- `stopping_criteria`
- `parsimony_coefficient`
- `p_crossover`
- `p_subtree_mutation`
- `p_hoist_mutation`
- `p_point_mutation`
- `p_point_replace`
- `const_range`
- `function_set`
- `init_depth`
- `init_method`
- `metric`
- `max_samples`
- `random_state`
- `n_jobs`
- `verbose`
- `low_memory`
- `warm_start`

当前返回特征：

- `get_optimal_equation()` 返回 `str(self.model)`
- `get_total_equations()` 当前返回单元素列表，只包含最优方程


### 5.2 `pysr`

Wrapper：

- `scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py`
- 类名：`PySRRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

参数策略：

- 显式白名单
- 会剥离元参数：`exp_name`、`exp_path`、`problem_name`、`seed`
- 若用户未传 `random_state`，则自动执行 `seed -> random_state`
- 若用户传入 `n_jobs`，会映射为 `procs`

允许参数：

- `niterations`
- `population_size`
- `populations`
- `ncycles_per_iteration`
- `maxsize`
- `maxdepth`
- `parsimony`
- `binary_operators`
- `unary_operators`
- `elementwise_loss`
- `model_selection`
- `loss_function`
- `warm_start`
- `random_state`
- `procs`
- `n_jobs`
- `verbosity`
- `progress`

当前返回特征：

- `get_optimal_equation()` 返回 `self.model.sympy()` 的字符串
- `get_total_equations()` 优先返回表达式列：
  - `sympy_format`
  - `equation`
  - `expr`
  - `expression`
- 若拿不到表达式列，则兜底为字符串字典列表

额外说明：

- `pysr` 首次运行通常会触发 Julia bootstrap，不适合作为首次 quickstart 工具。


### 5.3 `pyoperon`

Wrapper：

- `scientific_intelligent_modelling/algorithms/operon_wrapper/wrapper.py`
- 对外工具名：`pyoperon`
- 类名：`OperonRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

参数策略：

- 显式白名单
- 会剥离元参数：`exp_name`、`exp_path`、`problem_name`、`seed`
- 若用户未传 `random_state`，则自动执行 `seed -> random_state`

别名映射：

- `n_jobs` -> `n_threads`
- `niterations` -> `generations`
- `niteration` -> `generations`
- `population` -> `population_size`

允许参数：

- `allowed_symbols`
- `symbolic_mode`
- `crossover_probability`
- `crossover_internal_probability`
- `mutation`
- `mutation_probability`
- `offspring_generator`
- `reinserter`
- `objectives`
- `optimizer`
- `optimizer_likelihood`
- `optimizer_batch_size`
- `optimizer_iterations`
- `local_search_probability`
- `lamarckian_probability`
- `sgd_update_rule`
- `sgd_learning_rate`
- `sgd_beta`
- `sgd_beta2`
- `sgd_epsilon`
- `sgd_debias`
- `max_length`
- `max_depth`
- `initialization_method`
- `initialization_max_length`
- `initialization_max_depth`
- `female_selector`
- `male_selector`
- `population_size`
- `pool_size`
- `generations`
- `max_evaluations`
- `max_selection_pressure`
- `comparison_factor`
- `brood_size`
- `tournament_size`
- `irregularity_bias`
- `epsilon`
- `model_selection_criterion`
- `add_model_scale_term`
- `add_model_intercept_term`
- `uncertainty`
- `n_threads`
- `max_time`
- `random_state`
- `n_jobs`
- `niterations`
- `niteration`
- `population`

当前返回特征：

- `get_optimal_equation()` 返回最佳模型字符串
- `get_total_equations()` 返回 Pareto front 模型列表


### 5.4 `llmsr`

Wrapper：

- `scientific_intelligent_modelling/algorithms/llmsr_wrapper/wrapper.py`
- 类名：`LLMSRRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

参数策略：

- 没有显式白名单
- 但代码中明确消费了下面这些参数

核心参数：

- `problem_name`
- `background`
- `llm_config_path`
- `exp_path`
- `exp_name`
- `existing_exp_dir`
- `max_params`
- `niterations`
- `samples_per_iteration`
- `seed`

WandB 参数：

- `use_wandb`
- `wandb_project`
- `wandb_entity`
- `wandb_name`
- `wandb_group`
- `wandb_tags`
- `dataset_name`
- `train_path`
- `prompts_type`

当前返回特征：

- `get_optimal_equation()` 从实验目录 `samples/top*.json` 里读取最优 `function`
- `get_total_equations(n=None)` 从 `top*.json` 中按误差排序返回所有候选 `function`

当前恢复能力：

- 支持 `existing_exp_dir`
- 序列化后恢复依赖实验目录，而不是依赖 pickle 的大模型对象


### 5.5 `dso`

Wrapper：

- `scientific_intelligent_modelling/algorithms/dso_wrapper/wrapper.py`
- 类名：`DSORegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

参数策略：

- 无白名单
- `**kwargs` 会整体作为 `config` 传给 `DeepSymbolicRegressor(config=self.params)`

建议理解方式：

- 这里暴露的不是“单个算法参数”
- 而是 DSO 的配置字典结构

常见顶层配置块：

- `experiment`
- `task`
- `training`
- `logging`
- `state_manager`

当前返回特征：

- `get_optimal_equation()` 返回 `program_.pretty()`
- `get_total_equations()` 当前返回单元素列表

当前恢复能力：

- 反序列化后，若底层对象不可恢复，会回退到“表达式字符串 + lambdify 预测”


### 5.6 `tpsr`

Wrapper：

- `scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py`
- 类名：`TPSRRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

后端模式：

- `backbone_model="e2e"`
- `backbone_model="nesymres"`

默认参数：

- `backbone_model="e2e"`
- `max_input_points=10000`
- `max_number_bags=-1`
- `stop_refinement_after=1`
- `n_trees_to_refine=1`
- `rescale=True`
- `beam_size=10`
- `beam_type="sampling"`
- `no_seq_cache=False`
- `no_prefix_cache=False`
- `width=3`
- `num_beams=1`
- `rollout=3`
- `horizon=200`
- `seed=23`
- `cpu=False`
- `train_value=False`
- `lam=0.1`
- `nesymres_eq_setting_path="nesymres/jupyter/100M/eq_setting.json"`
- `nesymres_cfg_path="nesymres/jupyter/100M/config.yaml"`
- `nesymres_model_path=None`
- `symbolicregression_model_path="symbolicregression/weights/model.pt"`
- `symbolicregression_model_url="https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"`

代码里实际注入 parser 的参数：

- `backbone_model`
- `beam_size`
- `beam_type`
- `no_seq_cache`
- `no_prefix_cache`
- `width`
- `num_beams`
- `rollout`
- `horizon`
- `seed`
- `debug`
- `beam_length_penalty`
- `train_value`
- `ucb_constant`
- `uct_alg`
- `ucb_base`
- `cpu`
- `lam`
- `max_input_points`
- `max_number_bags`
- `rescale`
- `sample_only`

当前返回特征：

- `get_optimal_equation()` 返回归一化后的最佳表达式
- `get_total_equations()` 返回所有候选表达式列表

额外说明：

- `e2e` 与 `nesymres` 两个 backbone 都需要本地权重文件
- 当前工具比较适合作为高级用法，不适合作为最短 quickstart


### 5.7 `e2esr`

Wrapper：

- `scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py`
- 类名：`E2ESRRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

显式构造参数：

- `model_path`
- `model_url`
- `max_input_points`
- `n_trees_to_refine`
- `rescale`
- `force_cpu`
- `**kwargs`

实际传给 `SymbolicTransformerRegressor` 的参数：

- `max_input_points`
- `max_number_bags`
- `stop_refinement_after`
- `n_trees_to_refine`
- `rescale`

当前返回特征：

- `get_optimal_equation()` 返回最佳树解析得到的 SymPy 字符串
- `get_total_equations()` 当前只返回单元素列表

额外说明：

- 模型会在 `__init__` 阶段立即尝试加载
- 若未命中本地模型，会按 `model_url` 下载


### 5.8 `QLattice`

Wrapper：

- `scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py`
- 类名：`QLatticeRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

核心参数：

- `n_epochs`
- `kind`
- `signif`
- `output_name`

透传到 `feyn.QLattice.auto_run` 的参数：

- `stypes`
- `threads`
- `max_complexity`
- `query_string`
- `loss_function`
- `criterion`
- `sample_weights`
- `function_names`
- `starting_models`

当前返回特征：

- `get_optimal_equation()` 返回最佳表达式字符串
- `get_total_equations(n=None)` 在 wrapper 层支持 `n`

额外说明：

- 训练阶段依赖在线 `feyn.QLattice()` 社区版服务
- 反序列化后可通过已保存表达式离线预测


### 5.9 `iMCTS`

Wrapper：

- `scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py`
- 类名：`iMCTSRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

允许参数白名单：

- `ops`
- `arity_dict`
- `context`
- `max_depth`
- `K`
- `c`
- `gamma`
- `gp_rate`
- `mutation_rate`
- `exploration_rate`
- `max_single_arity_ops`
- `max_constants`
- `max_expressions`
- `verbose`
- `reward_func`
- `optimization_method`

特殊参数：

- `seed` 不在底层初始化白名单里，而是在 `fit(seed=...)` 时下发

当前返回特征：

- `get_optimal_equation()` 返回简化后的最佳表达式
- `get_total_equations()` 当前只返回单元素列表

当前恢复能力：

- 预测不依赖底层回归器对象，可用持久化的向量表达式重建


### 5.10 `drsr`

Wrapper：

- `scientific_intelligent_modelling/algorithms/drsr_wrapper/wrapper.py`
- 类名：`DRSRRegressor`

公共方法：

- `fit`
- `predict`
- `get_optimal_equation`
- `get_total_equations`

额外支持方法：

- `get_fitted_params`
- `get_total_equations_with_params`

代码里实际消费的参数：

- `existing_exp_dir`
- `exp_dir`
- `workdir`
- `problem_name`
- `background`
- `spec_path`
- `api_model`
- `api_key`
- `api_base`
- `temperature`
- `api_params`
- `samples_per_prompt`
- `evaluate_timeout_seconds`
- `wall_time_limit_seconds`
- `max_samples`
- `log_dir`

当前返回特征：

- `get_optimal_equation()` 返回包装后的完整 `def equation(..., params):` 函数字符串
- `get_total_equations(n=None)` 返回候选方程列表
- `get_fitted_params()` 返回最佳方程训练期参数
- `get_total_equations_with_params(n=None)` 返回结构化候选列表：
  - `equation`
  - `params`
  - `score`
  - `category`
  - `sample_order`

额外说明：

- 支持从 `existing_exp_dir` / `exp_dir` 离线复用已有实验
- 若没有历史拟合参数，wrapper 会尝试重新拟合参数
- 当提供 `background` 时会自动生成 specification；否则走 `spec_path`


## 6. 当前最值得对外公开的稳定接口

如果是面向外部使用者，建议优先强调下面这套最小公共面：

- `SymbolicRegressor(tool_name, **kwargs)`
- `fit(X, y)`
- `predict(X)`
- `get_optimal_equation()`
- `get_total_equations()`

按稳定性推荐的首选工具：

- 入门：`gplearn`
- 次选：`pyoperon`
- 进阶：`e2esr`、`QLattice`、`iMCTS`
- LLM 类：`llmsr`、`drsr`
- 重依赖：`pysr`、`tpsr`


## 7. 文档维护建议

当前仓库里“接口暴露”和“实现细节”有两个容易漂移的地方：

- wrapper 白名单与文档不同步
- `SymbolicRegressor` 公共方法签名与子进程透传行为不同步

建议后续维护时同时检查：

- `scientific_intelligent_modelling/srkit/regressor.py`
- `scientific_intelligent_modelling/srkit/subprocess_runner.py`
- `scientific_intelligent_modelling/algorithms/*_wrapper/wrapper.py`
- 本文档
