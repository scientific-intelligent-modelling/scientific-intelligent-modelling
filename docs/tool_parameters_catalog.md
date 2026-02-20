# 工具参数汇总（当前仓库可暴露参数）

目标：把“谁来改参数”这件事变成可控接口，不需要改源码即可给每个算法器配置超参数。

## 一、参数如何注入（两层接口）

1. CLI 外层（统一入口）
   - `python -m scientific_intelligent_modelling.cli -a <algorithm> -t <train_path> ...`
   - 已明确参数：`-a/--algorithm`、`-t/--train-path`、`--seed`、`--dataset-name`、`--prompts-type` 等。
   - 未声明参数通过 `argparse.parse_known_args` 和 `_parse_unknown_to_kwargs` 解析并透传：
     - `--key value`
     - `--key=value`
     - 仅长选项 `--`；短选项 `-k` 不会透传。
     - `--flag` 会透为布尔 `True`。
   - 透传后形成 `extra_params`，直接给 `SymbolicRegressor(..., **extra_params)`。
   - `SymbolicRegressor` 会附加：`problem_name`、`seed`、`exp_path`、`exp_name` 等。

2. 子进程执行链
   - `SymbolicRegressor` 的 `fit` 会把 `params` 放入命令 JSON，交给 `srkit/subprocess_runner.py`。
   - `subprocess_runner` 根据 `tool_name` 动态导入 `.../<tool>_wrapper/wrapper.py`，按 `regressor_class(**params)` 初始化后 `fit(X, y)`。

3. 实际可改参数的边界
   - 由每个 wrapper 定义。若 wrapper 使用 `**kwargs` 直传外层，实际参数名要看对应算法库（或子仓库文档）。

## 二、集成工具列表（当前仓库）

来自 `scientific_intelligent_modelling/config/toolbox_config.json` 的 `tool_mapping`：

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

## 三、逐工具参数清单（当前 wrapper 层）

### 1) `gplearn`
- Wrapper：`gplearn_wrapper/wrapper.py::GPLearnRegressor.__init__(**kwargs)`
- 方式：`self.params = kwargs`，`fit()` 时 `SymbolicRegressor(**self.params)`。
- 框架层可见参数：无显式白名单（全部透传到 gplearn）。
- 对外建议：请按 gplearn 官方 `SymbolicRegressor` 参数文档提供白名单（例如 `population_size`、`generations`、`function_set`、`parsimony_coefficient` 等，按你当前版本确认）。

### 2) `pysr`
- Wrapper：`pysr_wrapper/wrapper.py::PySRRegressor.__init__(**kwargs)`
- 方式：`self.params = kwargs`，`PySRRegressor(**self.params)`。
- 框架层可见参数：无显式白名单（全部透传到 pysr）。
- 对外建议：参数名按 pysr 官方文档（如 `niterations`、`population_size`、`binary_operators`、`unary_operators` 等）定义与发布版本一致的配置名。

### 3) `pyoperon`
- Wrapper：`operon_wrapper/wrapper.py::OperonRegressor.__init__(**kwargs)`
- 方式：`self.params = kwargs`，`pyoperon.sklearn.SymbolicRegressor(**self.params)`。
- 框架层可见参数：无显式白名单（全部透传到 pyoperon）。
- 对外建议：按 pyoperon 的 `SymbolicRegressor` 参数规范对外对齐。

### 4) `llmsr`
- Wrapper：`llmsr_wrapper/wrapper.py::LLMSRRegressor.__init__(**kwargs)`
- 已明确处理的参数（当前版本）：
  - `problem_name`
  - `background`
  - `llm_config_path`（默认子仓库 `llm.config`）
  - `exp_path`（默认 `./experiments`）
  - `exp_name`
  - `max_params`（默认 10）
  - `niterations`（默认 2500）
  - `samples_per_iteration`（默认 4）
  - `seed`
  - WandB：`use_wandb`、`wandb_project`、`wandb_entity`、`wandb_name`、`wandb_group`、`wandb_tags`
  - 元信息：`dataset_name`、`train_path`、`prompts_type`
- 使用方式（例）：
  - `--background "x in [0,1]" --max_params 20 --niterations 2000 --use_wandb`

### 5) `dso`
- Wrapper：`dso_wrapper/wrapper.py::DSORegressor.__init__(**kwargs)`（直接透传给 `DeepSymbolicRegressor`）
- 子仓库内部签名为 `DeepSymbolicRegressor(config=None)`，因此当前 wrapper 不稳健支持任意参数透传。
- 建议对外暴露：
  - 仅建议公开 `config`（JSON/路径）为第一入口。
  - 其他算法行为参数建议后续包装层显式化白名单再放开。

### 6) `tpsr`
- Wrapper：`tpsr_wrapper/wrapper.py::TPSRRegressor`
- 初始化设置默认参数（若未传则写入）：
  - `max_input_points=10000`
  - `max_number_bags=-1`
  - `stop_refinement_after=1`
  - `n_trees_to_refine=1`
  - `rescale=True`
  - `beam_size=10`
  - `beam_type='sampling'`
  - `backbone_model='e2e'`
  - `no_seq_cache=False`
  - `no_prefix_cache=False`
  - `width=3`
  - `num_beams=1`
  - `rollout=3`
  - `horizon=200`
- `fit()` 中明确读取并可覆盖的参数：
  - `backbone_model`, `beam_size`, `beam_type`, `no_seq_cache`, `no_prefix_cache`, `width`, `num_beams`, `rollout`, `horizon`, `debug`
- 提示：wrapper 只消费上述项；其余 submodule parser 字段当前不对外透传，若需开放需改 wrapper。

### 7) `e2esr`
- Wrapper：`e2esr_wrapper/wrapper.py::E2ESRRegressor.__init__(model_path=None, model_url=..., max_input_points=200, n_trees_to_refine=100, rescale=True, **kwargs)`
- 本层可见参数：
  - `model_path`
  - `model_url`
  - `max_input_points`
  - `n_trees_to_refine`
  - `rescale`
  - `**kwargs`（继续透传给 `SymbolicTransformerRegressor`）
- 对外建议：固定上层公开字段 + 需要时补充 `SymbolicTransformerRegressor` 的参数清单（位于子仓库实现处）。

### 8) `QLattice`
- Wrapper：`QLattice_wrapper/wrapper.py::QLatticeRegressor.__init__(**kwargs)`
- 当前已处理/记录的关键参数：
  - `n_epochs`（默认 100）
  - `kind`（默认 `regression`）
  - `signif`（默认 4）
  - `output_name`（默认 `y`）
- 可透传到 `feyn.QLattice.auto_run` 的参数：
  - `stypes`, `threads`, `max_complexity`, `query_string`, `loss_function`, `criterion`, `sample_weights`, `function_names`, `starting_models`
- `get_total_equations(n=None)` 支持 `n` 上限。

### 9) `iMCTS`
- Wrapper：`iMCTS_wrapper/wrapper.py::iMCTSRegressor`
- 显式透传到底层 `iMCTS.regressor.Regressor` 的参数白名单：
  - `ops`, `arity_dict`, `context`, `max_depth`, `K`, `c`, `gamma`, `gp_rate`, `mutation_rate`,
    `exploration_rate`, `max_single_arity_ops`, `max_constants`, `max_expressions`,
    `verbose`, `reward_func`, `optimization_method`
- 特殊：`seed` 通过 `fit(seed=...)` 下发（训练阶段参数）。
- 其它未在白名单内的参数当前不会透传。

### 10) `drsr`
- Wrapper：`drsr_wrapper/wrapper.py::DRSRRegressor`
- 关键参数（当前可见且已消费）：
  - `workdir`
  - `problem_name`
  - `background`
  - `spec_path`
  - `use_api`
  - `api_model`
  - `api_key`
  - `api_base`
  - `temperature`
  - `api_params`
  - `samples_per_prompt`（默认 1）
  - `evaluate_timeout_seconds`（默认 10）
  - `max_samples`（默认 2）
  - `log_dir`
  - `seed`（用于流程一致性）
- 说明：`spec_path` 在未提供 `background` 时使用；否则会自动由 `background` 生成临时 specification 文本。

## 四、给别人暴露参数的建议（可直接落地）

1. 对“可改哪些参数”定义为白名单（按上面 10 节）。
2. 暴露时统一返回 JSON：
   - `{"tool":"tpsr","params":{"beam_size":16,"rollout":5}}`
3. 统一走两种接口之一：
   - `python -m scientific_intelligent_modelling.cli -a tpsr -t data.csv --beam_size 16 --rollout 5`
   - `python -m scientific_intelligent_modelling.cli run-pipeline --tool-name tpsr --dataset-dir ... --iterations 1 --params-json '{"beam_size":16,"rollout":5}'`
4. 若你希望我继续往前走，我建议再加一张：
   - 你自己的“对外 API”白名单文件（例如 `configs/tool_parameter_schema.json`），和一份 `build_parser` 自动校验列表，避免未知参数静默进入算法却无效。
