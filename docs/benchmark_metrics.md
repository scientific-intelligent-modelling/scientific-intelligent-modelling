# Benchmark 指标接入说明

这份文档把三个原始 benchmark 的核心评测指标收敛成当前仓库可复用的统一口径。

## 1. SRBench

来源：
- 论文：La Cava et al., 2021
- 链接：https://cavalab.org/assets/papers/La%20Cava%20et%20al.%20-%202021%20-%20Contemporary%20Symbolic%20Regression%20Methods%20and%20their.pdf

原始指标分两组：

### 1.1 Black-box regression

- `R2 test`
  - 原论文主准确率指标。
- `Model size`
  - 定义为数学运算符、特征、常数数量之和。
  - 同时统计 raw expression 和 sympy simplify 后的 expression。
- `Training time`
  - 原论文结果图直接报告。

### 1.2 Ground-truth regression

- `Symbolic Solution`
  - 预测式不能退化成常数；
  - 满足 `phi* - phi_hat = a` 或 `phi* / phi_hat = b, b != 0`，其中 `a/b` 为常数。
- `Solution rate`
  - 上述 symbolic solution 的比例统计。

## 2. SRSD

来源：
- 论文：Matsubara et al., 2024
- 链接：https://openreview.net/pdf?id=qrUdrXsiXX

SRSD 明确批评了仅靠预测误差或 binary solution rate 的粗糙性，并引入了 `NED`。

原始指标：

- `R2 > 0.999`
  - 作为 accuracy 视角，继承自 SRBench 风格。
- `Solution rate`
  - 继承自 SRBench 的 binary symbolic correctness。
- `NED`
  - `normalized edit distance`，对简化后的 equation tree 使用 tree edit distance，再按真值树大小归一化。
  - 论文公式是：
    - `min(1, d(f_pred, f_true) / |f_true|)`

额外协议点：
- 模型选择使用 validation split 上的几何/相对误差；
- 最终评估使用 best model 计算 solution rate 和 NED；
- dummy variables 是 SRSD 的重要 stress test。

## 3. LLM-SRBench

来源：
- 论文：Shojaee et al., 2025
- 链接：https://proceedings.mlr.press/v267/shojaee25a.html
- 官方仓库：https://github.com/deep-symbolic-mathematics/llm-srbench

原始指标强调“符号正确性 + 数值泛化”：

- `SA` / `Symbolic Accuracy`
  - 论文主表核心指标。
- `Acc0.1`
  - 数值精度指标，记作 `Acc_tau` 在 `tau=0.1` 下的特例。
  - 当前工具集采用绝对误差阈值定义：
    - `mean(abs(y_true - y_pred) <= 0.1)`
- `NMSE`
  - 标准 normalized mean squared error。
- `ID / OOD`
  - 论文特别强调 OOD generalization。
  - 因此推荐在当前工具集中显式区分：
    - `id_test.acc_0_1`
    - `ood_test.acc_0_1`
    - `id_test.nmse`
    - `ood_test.nmse`

## 4. 当前仓库中的统一映射

统一 profile 已落在：

- `scientific_intelligent_modelling/benchmarks/profiles.py`
- `scientific_intelligent_modelling/config/benchmark_metric_profiles.json`

统一指标实现已落在：

- `scientific_intelligent_modelling/benchmarks/metrics.py`

当前提供：

- `regression_metrics(...)`
  - `mse / rmse / mae / r2 / nmse / acc_tau`
- `srbench_model_size(...)`
- `srbench_symbolic_solution(...)`
- `normalized_tree_edit_distance(...)`

## 5. 推荐接入方式

建议后续 benchmark 结果统一按以下槽位落表：

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

## 6. 注意事项

- `SRBench` 的 `symbolic solution` 是一个较宽松的等价定义，允许加性常数或非零标量倍。
- `SRSD` 的 `NED` 比 binary solution rate 更细粒度，更适合 scientific discovery 语境。
- `LLM-SRBench` 的核心不是只看数值误差，而是把 `SA + ID/OOD numerical metrics` 一起看。
- `srbench_model_size / srbench_symbolic_solution / normalized_tree_edit_distance` 依赖 `sympy`。若运行环境未安装 `sympy`，当前工具集仍可使用 profile 与数值指标，但符号类指标会在调用时报错。
- 若后续引入更完整的 benchmark runner，建议把 metric profile 当作 registry，而不是散落在脚本里手写。
