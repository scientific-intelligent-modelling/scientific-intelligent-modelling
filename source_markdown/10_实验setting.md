# 实验setting

- 原始文件：`实验setting.pdf`
- 文档类型：实验设置 / 成本估算 / 大规模实验规划
- 整理方式：按主题重排

## 核心内容

1. 这份文档是整个 benchmark 实验规模、运行成本和 prompt 设计的核心记录。
2. 你已经开始量化 DrSR 和 LLMSR 的 token、时间和价格成本。
3. 同时你也把背景提示、探针实验、超参数搜索、全量海量实验和 SSR-50 最佳结果实验都列成了可执行的实验矩阵。

## 整理正文

## 1. 初步成本记录

### DrSR
- 模型：`gpt-4o-mini`
- 时长：约 4 个半小时
- iteration：200
- 成本：约 0.54 元

记录示例：
- 第 1408 次
- 本次 tokens：prompt=1499, thinking=0, content=40, total=1539
- 累计 tokens：prompt=1652625, thinking=0, content=494158, total=2146783

### LLMSR
- 模型：`gpt-4o-mini`
- 时长：约 1 小时 15 分
- iteration：100
- 成本：约 0.2 元

记录示例：
- 第 400 次
- 本次 tokens：prompt=1094, thinking=0, content=771, total=1865
- 累计 tokens：prompt=281676, thinking=0, content=254960, total=536636
- 本次用时：27.35 秒
- 累计用时：4320.25 秒（约 1 小时 12 分）

## 2. 背景消融 / prompt 探测设计

文档里已经把这部分改造成 “best prompts 探测”。

### LLMSR 实验规模
- 数据集：`(15+15+15+16)` 个
- seed：3
- 背景数：7
- 总实验数：1281

### 数据集来源设定
- 【SRSD】使用 `1_domain`, `2_math`
- 【SRBench】使用 `1, 2, 3, 4, 4（匿名）`
- 【LLMBench】使用 `1, 2, 3, 4, 4（匿名）`

### 4 个背景模板

#### L0_TEMPLATE
```text
Mathematical function for {tag}
{var_block}
Task: Find the mathematical function skeleton that represents {meaning_sentence}
```

#### L1_TEMPLATE
```text
This is a problem from the natural sciences.
{var_block}
Structural Hint: The target formula captures a non-linear relationship. It may range from simple interactions to nested compositions of operators such as trigonometric functions, exponentials, or logarithms.
Task: Construct a mathematical expression to capture the patterns. The formula should exhibit scientific simplicity.
```

#### L2_TEMPLATE
```text
These data originate from real-world experiments in the natural sciences.
{var_block}
Task: Discover the underlying natural law governing the data. Requirements: The formula should exhibit scientific simplicity (Occam's Razor) and interpretability. It is typically composed of elementary mathematical functions.
```

#### L3_TEMPLATE
```text
This is an abstract symbolic regression problem.
{var_block}
Task: Find a mathematical function f that accurately fits the given data points (X, y)
```

### 变量块模板

#### anon_var_block
```text
Args:
 {vars_str}: Independent variables (anonymized)
 params: Array of numeric constants or parameters to be optimized
Return:
 y: Dependent variable (target)
```

#### named_var_block
```text
Args:
 {args_section},
 params: Array of numeric constants or parameters to be optimized
Return:
 y (modeling {target_name}) represents {target_desc}
```

#### get_srsd_named_var_block
```text
Args:
 {vars_str}: These are independent variables (candidate features).
 Note: This dataset contains irrelevant distractor variables. The active variables among them represent: {meanings_str}.
 params: Array of numeric constants or parameters to be optimized
Return:
 y (modeling {target_name}) represents {target_desc}
```

## 3. 结果产出设计

计划输出：
- 四子图
  - 左上：求解率 - prompts
  - 右上：五分类求解率 - prompts
  - 左下：求解效率
  - 右下：`R^2` - prompts
- 另外还有 5 条曲线：NMSE 随 iterations 变化

## 4. 探针实验规模
- 800 个数据集 × LLMSR 探针 × 3 seed = 2400 个实验
- 800 个数据集 × PySR 探针 × 3 seed = 2400 个实验

## 5. 平均超参数搜索实验
- 探针结果选出 25 个超参数数据集
- `25 × 8 个传统算法 × 3 seed × 30 gridsearch = 18000`
- `25 × 2 个 LLM 算法 × 3 seed × 6 prompts = 900`

## 6. 海量实验
冻结超参数后作用到 800 个数据集中：
- `800 × 8 个传统算法 × 3 seed = 19200`
- `800 × 2 个大模型方法 × 3 seed = 4800`

## 7. SSR-50 平均结果
- 各个算法在 SSR-50 上的平均水平
- 这里可以直接复用海量实验结果

## 8. SSR-50 最佳结果
- `50 × 8 个传统算法 × 3 seed × 30 超参 = 36000`
- `50 × 2 个大模型算法 × 2 基座 × 3 seed × 6 prompts = 3600`

## 归档备注

这份文档很重要，因为它把你的 benchmark 从“想法”推进到了“有明确成本、规模、变量控制和结果产物的可执行实验计划”。
