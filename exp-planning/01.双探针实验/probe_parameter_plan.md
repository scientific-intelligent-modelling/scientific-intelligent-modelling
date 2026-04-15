# 双探针实验参数方案

## 1. 目标

本方案用于全量扫描当前 `664` 个规范化数据集，使用两类归纳偏置差异明显的算法作为探针：

- `pysr`
- `llmsr`

目标不是把每个数据集都调到最好成绩，而是：

1. 用**统一配置**观察两种探针在全量数据集上的默认工作点表现。
2. 利用两种探针的性能差异、OOD 表现差异和失败模式差异，对数据集进行区分。
3. 进一步筛出区分度最大的 `50` 个核心 benchmark，再在这 `50` 个上做全量 `10` 算法评测。


## 2. 设计原则

本方案遵循以下原则：

1. **统一配置，不按 family 调参**
   - 目标是做 probe，不是追求某一类数据集的最优成绩。
2. **固定 wall-clock 预算**
   - 用相同时间预算比较探针，而不是让某一方因为更长搜索时间受益。
3. **中等强度，不追求极限**
   - 预算和搜索规模足以拉开差异，但不把成本烧在局部最优上。
4. **保留盲点**
   - probe 应该暴露算法偏好与不足，而不是靠扩展搜索语言把所有数据集都“吃掉”。
5. **优先稳定性**
   - 参数应适合大规模批量执行，减少失控、数值不稳定和不可复现情况。


## 3. 为什么这样设计 `PySR`

### 3.1 预算口径

`PySR` 使用 `timeout_in_seconds` 作为**主预算**，而不是依赖 `niterations` 先结束：

- `timeout_in_seconds=3600`
- `niterations=10_000_000`

这样做的目的：

1. 让实验预算统一为 **1 小时 wall-clock**
2. 避免不同数据集因为早碰到 `niterations` 上限而产生不一致的停止条件


### 3.2 搜索规模

选择：

- `population_size=64`
- `populations=8`
- `ncycles_per_iteration=500`

考虑：

1. 比非常小的 population 更稳定
2. 不会像过大的 population / populations 那样显著增加单任务成本
3. 适合用作统一默认工作点，而不是单任务最优调参


### 3.3 结构先验

当前统一算子集选为：

- 二元：`+ - * /`
- 一元：`square cube exp log sin cos`

选择原因：

1. 能覆盖大量科学公式常见结构：
   - 多项式
   - 有理式
   - 指数/对数
   - 三角
2. 不强行覆盖所有边角结构：
   - 不加 `abs`
   - 不加 `sqrt`
   - 不加 `tan`
   - 不加 `asin/acos`
   - 不加 piecewise

这样做的原因不是懒，而是 probe 需要保留语言盲点，才能在数据集筛选时产生区分度。


### 3.4 理论覆盖面的依据

对当前 `664` 个数据集的 `formula.py` 做静态扫描后，可得到：

- `435 / 664`：`likely covered`
- `168 / 664`：`conditionally covered via rewrite or domain assumption`
- `61 / 664`：`unlikely exact with current ops`

含义：

1. **likely covered**
   - 当前算子集下理论上可直接表示。
2. **conditionally covered**
   - 需要通过代数改写或额外定义域假设表示，例如：
     - `sqrt(x)` 通过 `exp(0.5 * log(x))`
     - `tan(x)` 通过 `sin(x) / cos(x)`
3. **unlikely exact**
   - 当前算子集下很难精确表达，例如：
     - `abs`
     - `np.where`
     - `arcsin`
     - `arccos`

这个分布本身说明：当前语言足够做 probe，但不适合拿来宣称“覆盖全部公式”。


### 3.5 复杂度与结构控制

`PySR` 不仅靠 `parsimony` 控制复杂度，还应显式使用：

- `constraints`
- `nested_constraints`
- `complexity_of_operators`
- `complexity_of_constants`
- `complexity_of_variables`

原因：

1. `constraints`
   - 限制某个算子子树的复杂度，避免 `exp(...)`、`log(...)` 内部无限膨胀。
2. `nested_constraints`
   - 限制某些算子彼此嵌套，例如不允许：
     - `exp(exp(x))`
     - `log(log(x))`
3. `complexity_of_operators`
   - 给更复杂的算子更高复杂度代价，例如：
     - `square: 2`
     - `cube: 3`
     - `exp: 3`
4. `complexity_of_constants=2`
   - 压制“靠堆常数拟合”的行为。
5. `complexity_of_variables=1`
   - 保持变量使用相对便宜。


### 3.6 可复现性

`PySR` 配置中保留：

- `precision=32`
- `deterministic=True`
- `procs=1`

原因：

1. 降低单任务资源占用
2. 适合大规模并发调度
3. 减少探针结果中的非必要随机波动


## 4. `PySR` 推荐 probe 配置

```python
pysr_probe = dict(
    timeout_in_seconds=3600,
    niterations=10_000_000,

    population_size=64,
    populations=8,
    ncycles_per_iteration=500,

    maxsize=30,
    maxdepth=10,
    parsimony=1e-3,

    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "cube", "exp", "log", "sin", "cos"],

    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 7,
        "log": 7,
        "sin": 9,
        "cos": 9,
    },
    nested_constraints={
        "exp": {"exp": 0, "log": 1},
        "log": {"exp": 0, "log": 0},
        "square": {"square": 1, "cube": 1, "exp": 0, "log": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0, "log": 0},
    },

    complexity_of_operators={
        "/": 2,
        "square": 2,
        "cube": 3,
        "sin": 2,
        "cos": 2,
        "exp": 3,
        "log": 3,
    },
    complexity_of_constants=2,
    complexity_of_variables=1,

    max_evals=None,
    early_stop_condition=None,

    precision=32,
    deterministic=True,

    model_selection="best",
    progress=True,
    verbosity=1,
    procs=1,
)
```


## 5. 为什么这样设计 `LLMSR`

### 5.1 预算口径

`LLMSR` 同样使用 **1 小时 wall-clock** 作为主预算：

- `timeout_in_seconds=3600`

`niterations` 设得足够大，让真正起作用的是时间预算而不是轮数上限：

- `niterations=100000`


### 5.2 搜索规模

保留：

- `samples_per_iteration=4`

原因：

1. 这是本地已有真实实验中验证过的工作点
2. 兼顾了样本多样性与 token 成本
3. 适合做 probe，不至于太保守


### 5.3 参数空间

使用：

- `max_params=12`

原因：

1. 对中低维公式足够灵活
2. 不至于让参数优化成本过高
3. 对小样本集更稳，减少过拟合风险

另外，当前共享 spec 生成器已经保证：

- `MAX_NPARAMS >= 特征数 + 1`

避免线性 seed 因参数槽位不足而越界。


### 5.4 背景策略

本方案不按 family 补背景知识，也不使用强领域提示。  
统一使用中性背景：

```text
This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.
```

目的：

1. 防止透题
2. 不让不同 family 因背景信息不均衡而被污染
3. 保留 `LLMSR` 作为弱语义先验探针的性质


### 5.5 LLM 选择

统一使用：

- `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct`

原因：

1. 已在当前项目中验证可用
2. 已用于本地真实实验
3. 成本可控
4. 对 probe 来说足够产生信息量


### 5.6 采样配置

沿用当前已验证工作点：

```json
{
  "model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
  "base_url": "https://api.deepinfra.com/v1/openai",
  "max_tokens": 1024,
  "temperature": 0.6,
  "top_p": 0.3
}
```

这些值不是拍脑袋给出的，而是来自本地已完成的真实实验配置。


## 6. `LLMSR` 推荐 probe 配置

```python
llmsr_probe = dict(
    timeout_in_seconds=3600,
    niterations=100000,
    samples_per_iteration=4,
    max_params=12,

    background="This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.",

    persist_all_samples=False,
)
```

配套 `llm.config`：

```json
{
  "model": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
  "base_url": "https://api.deepinfra.com/v1/openai",
  "max_tokens": 1024,
  "temperature": 0.6,
  "top_p": 0.3
}
```


## 7. 为什么这两套配置适合做双探针

### `PySR`

- 强结构约束
- 明确复杂度偏好
- 数值驱动搜索
- 对表达语言覆盖和复杂度控制高度敏感

### `LLMSR`

- 弱语义先验
- 程序骨架驱动
- prompt 统一但搜索行为与 `PySR` 明显不同
- 对统一背景下的公式归纳偏好更敏感

因此这两者作为双探针有明显互补性：

1. `PySR` 更像结构受控的经典搜索器
2. `LLMSR` 更像语义弱先验的程序生成搜索器
3. 两者差异足以对全量数据集产生有价值的区分信号


## 8. 不建议现在做的事

### 对 `PySR`

不建议在 probe 阶段先加入：

- `abs`
- `sqrt`
- `tan`
- `asin`
- `acos`
- piecewise 类 operator

原因：

1. 会抬高语言覆盖率
2. 会降低 probe 的区分性
3. 会让搜索空间和不稳定性显著增加


### 对 `LLMSR`

不建议在 probe 阶段：

- 按 family 改背景
- 使用更大模型
- 让背景包含领域知识或结构暗示

原因：

1. 这会让 probe 变成“谁更会吃背景”
2. 会污染后续 50 个核心 benchmark 的筛选逻辑


## 9. 当前方案的定位

本方案的定位不是：

- 让 `PySR` 和 `LLMSR` 在所有数据集上都最强

而是：

- 给两个归纳偏置差异明显的算法设定统一、稳定、可解释、成本可控的默认工作点
- 用于对全量数据集进行 probe
- 从中筛出最具区分度的核心 benchmark

