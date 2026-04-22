# 全工具数据维度约束审计（2026-04-23）

## 目的

审计当前工具集在 benchmark runner 下，是否把**当前数据集的真实输入维度**正确传递给各算法，并进一步判断：

1. 算法是否只是“拿到了 `X`”，还是**真正把维度约束接入了搜索 / 候选 / 导出链**
2. 哪些算法已经满足当前 benchmark 工程要求
3. 哪些算法仍存在“拿到了数据，但没有把维度变成约束”的集成缺口

---

## 审计口径

本次审计同时采用两种标准。

### 标准 A：工程可用标准

只要算法在 `fit(X, y)` 里能从 `X.shape[1]` 推断当前维度，并且最终不会把非法变量带进 benchmark 评测链，就视为**工程可用**。

### 标准 B：严格契约标准

如果要求：

> 工具集必须在 runner 层把当前数据集维度作为**显式契约信息**传给每个算法，
> 而不是只靠 wrapper 自己从 `X` 推断，

那么当前框架**还不完全达标**。

原因是：

- `runner -> SymbolicRegressor -> wrapper`
- 目前统一传的是：
  - `X`
  - `y`
- 没有一条全局统一、显式的：
  - `n_features`
  - `feature_names`
  - `target_name`
 运行时契约注入到所有 wrapper

也就是说：

- 现在多数算法是**隐式拿到维度**
- 不是**框架显式声明维度**

---

## 当前框架的真实状态

### 框架层

在 runner 里，所有算法统一是：

- [runner.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/benchmarks/runner.py)
- `run_benchmark_task(...) -> reg.fit(dataset.train.X, dataset.train.y)`

也就是说：

- 当前框架一定会把真实 `X` 传给 wrapper
- 所以 wrapper 只要愿意，就能从：
  - `X.shape[1]`
 直接知道当前数据集维度

### 问题不在“X 没传进去”

问题在于：

- 有些算法只是拿到了 `X`
- 但没有把当前维度真正接到：
  - 候选生成
  - 中间快照
  - 公式导出
  - 评测恢复

`TPSR` 就是这类问题的典型案例。

---

## 审计结论总表

| 算法 | 当前是否拿到真实维度 | 维度约束方式 | 是否已知存在集成缺口 | 结论 |
| --- | --- | --- | --- | --- |
| `gplearn` | 是 | 通过 `X` 进入 sklearn GP，本身无独立变量词表；`n_features_in_` 记录维度 | 否 | 通过 |
| `pysr` | 是 | 通过 `X` 进入 PySR，本身无独立变量词表；导出使用 `n_features_in_` | 否 | 通过 |
| `pyoperon` | 是 | `fit()` 显式记录 `n_features_`，搜索直接吃 `X` | 否 | 通过 |
| `llmsr` | 是 | 显式把 `X` 改写成 `x0..x{n-1}` CSV，并把 `n_features` 接到 prompt / 导出链 | 否 | 通过 |
| `drsr` | 是 | 显式记录 `_n_features`，接到 prompt / compile / export 链 | 否 | 通过 |
| `dso` | 是 | 训练时将 `X,y` 写成 CSV，DSO 从数据集推断变量；wrapper 记录 `_dso_n_features` | 否 | 通过 |
| `QLattice` | 是 | 显式把 `X` 转成列名为 `x0..x{n-1}` 的 DataFrame，预测也检查列数 | 否 | 通过 |
| `iMCTS` | 是 | 显式记录 `_n_features`，并把 `X` 转成 `(n_features, n_samples)` 供底层使用 | 否 | 通过 |
| `TPSR` | 是 | wrapper 记录 `_n_features`，但原始预训练解码词表固定 10 维；已加集成层投影/过滤修复 | **是，已修复** | 通过（需注明修复版） |
| `E2ESR` | 是 | wrapper 记录 `n_features_`；训练与进度写出链依赖内部 tree 评估 | 暂未复现，但属同类预训练词表路线 | **通过，建议继续专项观察** |

---

## 逐工具审计

### 1. `gplearn`

代码位置：

- [gplearn_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py)

关键点：

- `fit()` 中显式记录：
  - `self.model.n_features_in_`
- 搜索直接在 `X` 上进行
- 不存在独立变量词表
- 导出时使用：
  - `expected_n_features = n_features_in_`

判断：

- 这是典型“隐式维度传递但完全成立”的算法
- 当前没有集成缺口

### 2. `pysr`

代码位置：

- [pysr_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py)

关键点：

- 训练直接吃 `X`
- 底层模型会记录：
  - `n_features_in_`
- 工件导出使用：
  - `expected_n_features = model.n_features_in_`

判断：

- 维度约束由底层训练数据天然决定
- 当前没有独立词表问题

### 3. `pyoperon`

代码位置：

- [pyoperon_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/pyoperon_wrapper/wrapper.py)

关键点：

- `fit()` 里显式记录：
  - `self.n_features_ = X.shape[1]`
- 搜索直接基于 `X`
- 导出时使用：
  - `expected_n_features = self.n_features_`

判断：

- 显式记录维度，且无独立变量词表
- 当前没有集成缺口

### 4. `llmsr`

代码位置：

- [llmsr_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/llmsr_wrapper/wrapper.py)

关键点：

- `fit()` 里显式计算：
  - `n_features = X.shape[1]`
- 把输入数据转写为：
  - `x0, x1, ..., y`
- prompt / spec / 导出链都围绕这组列名工作

判断：

- runner 现在已经显式注入：
  - `n_features`
  - `feature_names`
  - `target_name`
- wrapper 也会把当前数据维度转换成统一命名空间
- 当前没有集成缺口

### 5. `drsr`

代码位置：

- [drsr_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/drsr_wrapper/wrapper.py)

关键点：

- `fit()` 显式记录：
  - `self._n_features`
- prompt 构造用：
  - `PromptContext(n_features=...)`
- 公式包装 / 编译 / 导出也都用同一套 `_n_features`

判断：

- 显式维度链条完整
- 当前没有集成缺口

### 6. `dso`

代码位置：

- [dso_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/dso_wrapper/wrapper.py)

关键点：

- 训练前把 `X,y` 写成临时 CSV
- DSO 底层从该 CSV 推断变量维度
- wrapper 训练后记录：
  - `_dso_n_features`
- 预测时若输入维度不足会显式报错

判断：

- 这里是“数据文件驱动的维度约束”
- 不是显式 `n_features` 参数，但工程上是成立的

### 7. `QLattice`

代码位置：

- [QLattice_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py)

关键点：

- 显式把 `X` 转成：
  - `DataFrame(columns=[x0, x1, ...])`
- 预测时还会再次校验：
  - 输入列数必须等于训练时列数

判断：

- 维度约束非常明确
- 当前没有集成缺口

### 8. `iMCTS`

代码位置：

- [iMCTS_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py)

关键点：

- `fit()` 显式记录：
  - `_n_features = X.shape[1]`
- 并把输入改造成底层要求的：
  - `(n_features, n_samples)`

判断：

- 当前数据维度已经真实进入底层优化器
- 当前没有集成缺口

### 9. `TPSR`

代码位置：

- [tpsr_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py)

关键点：

- wrapper 一直能拿到：
  - `X.shape[1]`
- 也记录了：
  - `_n_features`

但问题在于：

- `TPSR` 的预训练模型与环境词表绑定
- 环境默认词表是固定 10 维：
  - `x_0 ~ x_9`
- 原始中间候选可能生成：
  - `x_9`

这说明：

- 数据维度**被拿到了**
- 但没有真正进入**解码候选约束**

本次修复后：

- 不再粗暴修改环境词表大小
- 而是在集成层：
  - 记录当前 `_n_features`
  - 对超出维度的变量 token 投影为 `0`
  - 再写中间 best / 最终候选

判断：

- `TPSR` 是当前唯一**明确暴露出维度约束集成缺口**的算法
- 现在已通过集成修复关闭这个缺口

### 10. `E2ESR`

代码位置：

- [e2esr_wrapper/wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py)
- [e2esr/symbolicregression/model/sklearn_wrapper.py](/home/family/workplace/scientific-intelligent-modelling/scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/sklearn_wrapper.py)

关键点：

- wrapper 会记录：
  - `n_features_`
- 进度状态不是直接写原始字符串候选，而是：
  - 先对 tree 计算 metric
  - 再写 `.e2esr_current_best.json`

这和 `TPSR` 不一样：

- `TPSR` 是原始字符串候选先落盘，再被 runner 评测
- `E2ESR` 是候选先过内部 tree/metric 链，再写进度

判断：

- 当前没有复现出和 `TPSR` 同等级的越界变量问题
- 但由于它同样属于预训练固定词表路线，仍建议保留专项观察

---

## 总体结论

### 按工程可用标准（标准 A）

当前 10 个算法里：

- **9 个没有发现维度约束缺口**
- `TPSR` 发现过缺口，但已修复

也就是说：

> **当前工具集在“让算法拿到真实数据维度并进入 benchmark”这件事上，整体是可用的。**

### 按你要求的严格契约标准（标准 B）

如果按你的标准：

> **“每个算法在进入核心逻辑前，都应该被工具集显式告知当前数据维度”**

当前最新实现已经做到：

- `runner/build_runner_params()` 会统一显式注入：
  - `n_features = len(dataset.feature_names)`
  - `feature_names = dataset.feature_names`
  - `target_name = dataset.target_name`
- 严格白名单或可能透传到底层库的 wrapper 已显式吸收这些元参数，
  避免出现“未知参数”或污染第三方配置的情况。
- 所有 10 个 wrapper 现在都会在 `fit()` 入口校验：
  - `n_features == X.shape[1]`
  - `len(feature_names) == X.shape[1]`
  - `target_name` 为非空字符串

---

## 我建议的下一步

### 必做

1. 保持 `TPSR` 当前修复版不回退
2. 在文档中明确：
   - `TPSR` 的非法变量问题属于工具集集成缺口，已修复

### 推荐做
后续可以继续把这套显式契约再向前推进一层：

- 把 `feature_descriptions`
- `target_description`

也纳入统一可选契约，并对需要语义提示的算法（如 `llmsr / drsr`）统一消费。

这样不仅是“结构维度一致”，还能进一步做到：

> **“工具集会把当前任务的结构契约和语义契约一起显式交给算法。”**
