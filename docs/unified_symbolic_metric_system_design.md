# 统一符号指标系统实施设计

## 1. 目标

本设计用于将当前仓库中的全部符号回归算法接入**同一套完整符号指标系统**。

这里的“完整符号指标系统”指：

- `symbolic_validity`
- `symbolic_complexity`
- `symbolic_solution`
- `solution_rate`
- `normalized_tree_edit_distance` (`ned`)
- `symbolic_accuracy`

约束前提：

1. 接受部分算法通过**受控转换**进入统一表示。
2. 接受只有带 `ground truth` 公式的数据集，才能计算完整符号指标。
3. 不允许不同算法各自用不同表达式格式直接算指标，必须先收敛到统一工件协议。
4. 默认不依赖 `sympy.simplify` 作为主路径；`sympy` 只负责解析和结构化。
5. LLM 只作为 `fallback` 和 `semantic equivalence judge`，不作为全量主翻译器。

## 2. 为什么必须引入统一符号工件协议

当前各算法对外暴露的方程形式并不统一：

- 标准表达式风格：
  - `pysr`
  - `QLattice`
- 近似标准表达式，但 token/格式不稳定：
  - `pyoperon`
  - `e2esr`
  - `tpsr`
- 前缀树/pretty string 风格：
  - `gplearn`
  - `dso`
- Python 函数/程序体风格：
  - `llmsr`
  - `drsr`
- 执行语境依赖较强：
  - `iMCTS`

如果继续让 benchmark 直接消费这些原始输出，那么“同一套符号指标”只是表面统一，实际输入对象完全不同，结果不公平，也无法稳定复现。

因此，系统必须改为：

```text
原始算法输出
  -> 受控归一化
  -> CanonicalSymbolicProgram
  -> 统一符号指标计算
```

## 3. 统一符号工件协议

建议新增统一工件协议：`CanonicalSymbolicProgram`。

建议字段如下：

```python
{
    "version": "csp_v1",
    "tool_name": "pysr",
    "raw_equation": "...",
    "raw_equation_kind": "plain_expression",
    "python_function_source": "def equation(x0, x1, params):\n    return ...\n",
    "return_expression_source": "params[0] + x0 * x1",
    "normalized_expression": "c0 + x0*x1",
    "variables": ["x0", "x1"],
    "parameter_symbols": ["c0"],
    "parameter_values": [1.234],
    "operator_set": ["add", "mul"],
    "ast_node_count": 5,
    "tree_depth": 3,
    "normalization_mode": "direct|rule|ast|llm_fallback",
    "normalization_notes": [],
    "fidelity_check": {
        "passed": true,
        "rmse": 0.0,
        "nmse": 0.0,
        "sample_size": 32
    }
}
```

### 3.1 强约束

所有算法最终都必须导出：

```python
def equation(x0, x1, ..., params):
    return ...
```

该函数必须是**表达式型函数**，不允许任意程序逻辑。

允许的 AST 节点建议限制为：

- `Module`
- `FunctionDef`
- `arguments`
- `arg`
- `Return`
- `Expr`
- `BinOp`
- `UnaryOp`
- `Call`
- `Name`
- `Load`
- `Constant`
- `Subscript`
- `Tuple`
- `List`

禁止：

- `Import`
- `ImportFrom`
- `For`
- `While`
- `If`
- `With`
- `Try`
- `Lambda`
- `Attribute` 的任意链式访问
- `exec`
- `eval`
- 文件 IO

### 3.2 数据集约束

完整符号指标仅对具有 `ground_truth_formula` 的数据集运行。

建议在 `metadata.yaml` 中统一要求：

```yaml
ground_truth_formula:
  available: true
  file: formula.py
  function_name: equation
```

如果数据集没有真值公式，则仍可运行数值指标，但不能标记为“完整符号指标已完成”。

## 4. 现有仓库的最小侵入改造点

### 4.1 方法侧

当前方法接口定义在：

- [base_wrapper.py](../scientific_intelligent_modelling/algorithms/base_wrapper.py)

当前子进程动作分发在：

- [regressor.py](../scientific_intelligent_modelling/srkit/regressor.py)
- [subprocess_runner.py](../scientific_intelligent_modelling/srkit/subprocess_runner.py)

建议新增以下接口。

#### BaseWrapper 新接口

在 `BaseWrapper` 中新增非抽象默认方法：

```python
def export_canonical_symbolic_program(self):
    raise NotImplementedError(...)
```

设计理由：

- 不会立刻打爆现有 10 个 wrapper；
- 可以分批迁移；
- benchmark runner 可以显式检查能力缺失，而不是隐式失败。

#### SymbolicRegressor 新接口

在 `srkit/regressor.py` 新增：

```python
def export_canonical_symbolic_program(self):
    ...
```

职责：

- 通过子进程向具体 wrapper 发起导出；
- 统一返回 JSON 可序列化对象；
- 供 benchmark runner 直接消费。

#### subprocess_runner 新 action

在 `srkit/subprocess_runner.py` 新增：

- `export_canonical_symbolic_program`

新增处理函数：

```python
def handle_export_canonical_symbolic_program(...)
```

职责：

- 反序列化回归器；
- 调用 wrapper 的 `export_canonical_symbolic_program()`；
- 返回 `{"success": True, "artifact": ...}`。

### 4.2 指标侧

当前指标入口在：

- [metrics.py](../scientific_intelligent_modelling/benchmarks/metrics.py)
- [profiles.py](../scientific_intelligent_modelling/benchmarks/profiles.py)
- [judges.py](../scientific_intelligent_modelling/benchmarks/judges.py)

建议新增以下模块。

#### 新增 `normalizers.py`

路径建议：

- `scientific_intelligent_modelling/benchmarks/normalizers.py`

职责：

- 把各工具原始输出归一化到 `CanonicalSymbolicProgram`
- 提供工具特化转换器

建议接口：

```python
def normalize_equation(
    *,
    tool_name: str,
    raw_equation: str,
    fitted_params: list[float] | None = None,
    total_equation_item: dict | None = None,
) -> dict:
    ...
```

以及：

- `normalize_gplearn_expression(...)`
- `normalize_dso_expression(...)`
- `normalize_python_function_equation(...)`
- `normalize_imcts_expression(...)`
- `validate_python_function_ast(...)`
- `fidelity_check_against_reference(...)`

#### 新增 `artifact_schema.py`

路径建议：

- `scientific_intelligent_modelling/benchmarks/artifact_schema.py`

职责：

- 定义 `CanonicalSymbolicProgram`
- 提供字段校验与默认值填充

#### 新增 `symbolic_runner.py`

路径建议：

- `scientific_intelligent_modelling/benchmarks/symbolic_runner.py`

职责：

- 接收 `CanonicalSymbolicProgram`
- 根据 benchmark profile 统一计算完整符号指标
- 按数据集是否有 `ground_truth` 决定可用指标

## 5. 统一数据流

建议最终数据流如下：

```text
SymbolicRegressor.fit/predict
  -> wrapper.get_optimal_equation / get_total_equations / get_fitted_params
  -> wrapper.export_canonical_symbolic_program
  -> benchmarks.normalizers / artifact_schema
  -> CanonicalSymbolicProgram
  -> benchmarks.symbolic_runner
  -> metrics.py / judges.py
  -> 统一指标结果 JSON
```

### 5.1 候选方程与最佳方程

建议统一支持两层输出：

- `best_artifact`
- `candidate_artifacts`

这可以直接复用你现有的：

- `get_optimal_equation()`
- `get_total_equations()`
- `get_fitted_params()`
- `get_total_equations_with_params()`

也就是说，不需要推翻已有接口，只需要在 wrapper 内新增“把它们组装成统一工件”的一层。

## 6. benchmark 指标怎么映射到统一工件

### 6.1 SRBench

需要的字段：

- `complexity_raw`
- `complexity_simplified`
- `symbolic_solution`
- `solution_rate`

映射方式：

- `complexity_raw` 来自统一 AST 树计数
- `complexity_simplified` 暂保留可选项；默认关闭，按配置开启
- `symbolic_solution` 基于统一 `normalized_expression` 与 GT 比较
- `solution_rate` 在任务集上聚合

### 6.2 SRSD

需要的字段：

- `ned`
- `solution_rate`

映射方式：

- `ned` 基于统一 AST 树比较
- `solution_rate` 沿用 SRBench 二值化逻辑

### 6.3 LLM-SRBench

需要的字段：

- `symbolic_accuracy`
- `id_test.acc_0_1`
- `ood_test.acc_0_1`
- `id_test.nmse`
- `ood_test.nmse`

映射方式：

- 数值部分继续用现有 `metrics.py`
- `symbolic_accuracy` 用统一 `CanonicalSymbolicProgram` 的 `normalized_expression` / `python_function_source` 作为 judge 输入
- LLM judge 保留在 [judges.py](../scientific_intelligent_modelling/benchmarks/judges.py)

## 7. profile 扩展建议

当前 profile 只定义了指标，不知道“前置依赖”。

建议扩展 `profiles.py` 和 `benchmark_metric_profiles.json`，给每个指标补：

- `requires_ground_truth`
- `requires_symbolic_artifact`
- `requires_llm_judge`
- `requires_candidate_artifacts`

示例：

```json
{
  "name": "symbolic_accuracy",
  "toolkit_field": "symbolic_accuracy",
  "requires_ground_truth": true,
  "requires_symbolic_artifact": true,
  "requires_llm_judge": true
}
```

这样 runner 可以显式做调度，而不是到运行时才抛异常。

## 8. 受控转换策略

### 8.1 总原则

- 规则化转换为主
- AST 抽取为主
- LLM 仅作为 fallback
- 每次转换后必须做忠实性验证

### 8.2 忠实性验证

每个算法从原始输出转成 `CanonicalSymbolicProgram` 后，必须跑一次数值一致性检查：

1. 用原始算法预测器对一批样本求值
2. 用规范 Python 函数再次求值
3. 比较两者输出

推荐验收线：

- `rmse <= 1e-10`，或
- `nmse <= 1e-12`

若不通过：

- 标记 `fidelity_passed = false`
- 不允许进入完整符号指标计算

### 8.3 LLM fallback 使用边界

只有以下情况才允许调用 LLM：

- 原始表达式中混入自然语言
- 原始表达式是非标准程序片段，规则解析失败
- 变量名和参数名无法自动对齐
- `iMCTS` 等输出存在上下文依赖，无法通过规则化稳定提取

LLM 转换后仍必须经过：

- AST 安全校验
- 忠实性验证

## 9. 10 个工具的迁移优先级

### 第一批：直接映射

- `pysr`
- `QLattice`

目标：

- 直接实现 `export_canonical_symbolic_program`
- 不引入 LLM

### 第二批：轻量规则化

- `pyoperon`
- `e2esr`
- `tpsr`

目标：

- 只做 token 统一、变量名统一、参数名统一

### 第三批：专用解析器

- `gplearn`
- `dso`

目标：

- `gplearn`：前缀表达式转中缀
- `dso`：pretty string 清洗后转表达式

### 第四批：程序式抽取

- `llmsr`
- `drsr`

目标：

- 从 `def equation(...)` 中提取标准函数
- 抽 `return_expression_source`
- 统一 `params[i]`

### 第五批：难点兜底

- `iMCTS`

目标：

- 优先深入 wrapper 或上游库暴露内部表达式
- 如果无法稳定直接导出，再用 LLM fallback

## 10. 建议的文件级改造清单

### 新增文件

- `scientific_intelligent_modelling/benchmarks/artifact_schema.py`
- `scientific_intelligent_modelling/benchmarks/normalizers.py`
- `scientific_intelligent_modelling/benchmarks/symbolic_runner.py`
- `tests/test_symbolic_artifact_schema.py`
- `tests/test_symbolic_normalizers.py`
- `tests/test_symbolic_runner.py`

### 修改文件

- `scientific_intelligent_modelling/algorithms/base_wrapper.py`
- `scientific_intelligent_modelling/srkit/regressor.py`
- `scientific_intelligent_modelling/srkit/subprocess_runner.py`
- `scientific_intelligent_modelling/benchmarks/metrics.py`
- `scientific_intelligent_modelling/benchmarks/profiles.py`
- `scientific_intelligent_modelling/config/benchmark_metric_profiles.json`
- `docs/benchmark_metrics.md`

### 第二阶段再修改的 wrapper

- `scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/operon_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/dso_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/llmsr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/drsr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py`

## 11. 分阶段实施顺序

### Phase 0：冻结协议

产出：

- `CanonicalSymbolicProgram` 字段定义
- AST 白名单
- fidelity 验收规则

### Phase 1：打通主链路

产出：

- `BaseWrapper.export_canonical_symbolic_program`
- `SymbolicRegressor.export_canonical_symbolic_program`
- `subprocess_runner` 新 action
- `artifact_schema.py`

验收：

- 单个算法能通过子进程导出标准工件

### Phase 2：打通最小四工具

先接：

- `pysr`
- `gplearn`
- `llmsr`
- `drsr`

原因：

- 覆盖四种最不同的输出风格
- 一旦这四个通了，架构就基本成立

验收：

- 四个工具都能导出 `CanonicalSymbolicProgram`
- fidelity check 通过
- 可计算完整符号指标

### Phase 3：接剩余六工具

- `QLattice`
- `pyoperon`
- `e2esr`
- `tpsr`
- `dso`
- `iMCTS`

### Phase 4：统一 benchmark runner

产出：

- 基于 profile 的自动调度
- 按数据集 GT 可用性自动切换完整符号指标
- 统一结果 JSON schema

## 12. 测试策略

### 12.1 单元测试

- 工件 schema 校验
- AST 白名单校验
- 各工具 normalizer 解析测试
- 忠实性验证测试
- LLM fallback 缓存测试

### 12.2 集成测试

每组至少挑一个方法：

- 直接映射组：`pysr`
- 规则化组：`gplearn`
- 程序式组：`llmsr`
- 难点组：`iMCTS`

### 12.3 Benchmark 回归测试

对带 GT 的 examples 数据集，统一检查：

- `symbolic_validity`
- `symbolic_solution`
- `ned`
- `symbolic_accuracy`

## 13. 风险点

### 风险 1：LLM 改写语义

应对：

- LLM 仅做 fallback
- fallback 后必须做 fidelity check

### 风险 2：程序式输出不可安全执行

应对：

- 严格 AST 白名单
- 禁止任意导入和副作用

### 风险 3：某些工具没有稳定暴露内部表达式

应对：

- 优先改 wrapper 暴露内部最佳表达式
- 不足时才引入 LLM fallback

### 风险 4：没有 ground truth 的数据集无法计算完整符号指标

应对：

- 明确写入 runner 规则
- 不再把这类任务标记为“完整符号评测”

## 14. 最小闭环建议

如果按最小可执行路径推进，建议先做以下闭环：

1. 新增 `artifact_schema.py`
2. 新增 `normalizers.py`
3. 给 `BaseWrapper` / `regressor.py` / `subprocess_runner.py` 加导出接口
4. 先实现四个工具：
   - `pysr`
   - `gplearn`
   - `llmsr`
   - `drsr`
5. 新增 `symbolic_runner.py`
6. 在一个有 GT 的 examples 数据集上跑通：
   - `symbolic_solution`
   - `ned`
   - `symbolic_accuracy`

做到这一步，系统就从“设计”进入“可验证架构”了。
