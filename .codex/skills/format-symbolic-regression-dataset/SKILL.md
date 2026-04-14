---
name: format-symbolic-regression-dataset
description: 当需要把符号回归数据源规范化为当前仓库统一数据集目录时使用。适用于 CSV/TSV/Excel/JSON/多文件导出或 benchmark 原始数据，整理成 train.csv、valid.csv、id_test.csv、metadata.yaml，并按需要生成可选的 ood_test.csv 与 formula.py，再用校验脚本检查输出是否符合当前 pipeline 的真实约定。
---

# Format Symbolic Regression Dataset

用于把符号回归数据源整理成当前仓库统一数据集目录。

## 何时使用

- 用户给了一份或多份异构数据，想接成当前项目可直接消费的数据集。
- 用户说“整理成和 examples 一样的格式”。
- 用户想把 benchmark 原始数据批量整理成统一 schema，例如 SRBench、SRSD、LLM-SRBench 一类数据。
- 需要从 CSV、Excel、多 sheet、JSON、导出目录、实验日志中抽出统一的训练/验证/测试数据。

## 目标输出

输出目录至少应包含：

- `train.csv`
- `valid.csv`
- `id_test.csv`
- `metadata.yaml`

可选包含：

- `ood_test.csv`
- `formula.py`

当前框架的加载契约可见：

- `scientific_intelligent_modelling/pipelines/iterative_experiment.py`
- `check/run_phys_osc_task.py`

## 快速流程

1. 先读目标格式：
   - `references/examples_contract.md`
2. 再按源数据类型选择整理方式：
   - `references/source_mapping_patterns.md`
3. 判断 split 来源：
   - 源数据已给好 train/valid/test：直接映射
   - 只有单表：你必须显式设计 train/valid/id_test/ood_test 的切分规则
4. 统一列名、目标列、特征列、数值类型、缺失值策略。
5. 若需要先起一个空壳目录，可执行：
   - `python3 tools/example_dataset_onboarder/scripts/scaffold_example_dataset.py --dataset-dir <dir> --dataset-name <name> --target <target> --features x0,x1`
6. 生成 `metadata.yaml`：
   - 可参考 `tools/example_dataset_onboarder/templates/metadata_template.yaml`
7. 若源数据是 benchmark 风格的解析式数据，可按需参考本 skill 自带脚本：
   - `scripts/analyze_feature_distributions.py`：分析 train split 中各特征分布
   - `scripts/generate_ood_samples.py`：基于区间规则生成 OOD 样本
8. 这些脚本中，只有下面两类应被视为当前主流程：
   - `analyze_feature_distributions.py`：适合已整理成 `train.csv` 的目录
   - `generate_ood_samples.py`：适合已有 `formula.py + feature_distributions.csv` 的目录
9. metadata 的生成与字段选择，不应依赖历史批处理脚本，默认应直接按照：
   - `references/examples_contract.md`
   - `references/metadata_rules.md`
   来手工或按当前任务写出 `metadata.yaml`
10. 旧的 Feynman 专用后处理脚本已降级为参考材料，放在：
   - `references/legacy-feynman-scripts/`
   这些脚本只在需要兼容旧 Feynman 产物时才看，不是主路径。
11. 因此默认策略应是：
   - 先用当前 skill 的通用流程完成 split、列名、基础 metadata
   - 若数据集已经满足 `<group>/<dataset>/train.csv`、存在 `formula.py`，则可以直接复用 `generate_ood_samples.py`
   - metadata、描述、ranges、notes 优先直接写入 `metadata.yaml`
12. 使用脚本前必须先读源码，确认列名、目录层级、目标列命名与当前数据一致。
13. 不要在语义不匹配的数据上直接运行。
14. 若已知真值公式，再写 `formula.py`。
15. 若存在 `ground_truth_formula.file`，必须做公式代入校验：
   - `python3 tools/example_dataset_onboarder/scripts/validate_example_dataset.py --dataset-dir <dir> --verify-formula`
16. 跑校验：
   - `python3 tools/example_dataset_onboarder/scripts/validate_example_dataset.py --dataset-dir <dir>`
17. 若是“无天然 OOD 且无 ground truth”的单表数据，可使用：
   - `python3 .codex/skills/format-symbolic-regression-dataset/scripts/extract_ood_by_iterative_shrink.py --input-csv <all.csv> --output-dir <dir>`
   - 该脚本会逐维按区间长度的 `0.01%` 进行内缩，并把区间外样本定义为 OOD
   - 当 OOD 占比达到约 `10%` 时停止
   - 若在达到目标前 OOD 占比超过 `20%`，则判定失败，不生成 `ood_test.csv`

## 缺文件决策规则

当源数据无法直接提供完整的 `train/valid/id_test/ood_test` 四件套时，默认按下面的规则处理。

### 情形一：只有一个总表

- 若没有时间顺序、实体分组或明确 OOD 语义：
  - 使用固定随机种子 `1314`
  - 若同时缺少 ground truth 且想构造 OOD：
    - 先尝试用 `scripts/extract_ood_by_iterative_shrink.py` 从总表中抽取 OOD
    - 抽取成功后，再把剩余 ID 样本切成：
      - `train`: 80%
      - `valid`: 10%
      - `id_test`: 10%
  - 若不构造 OOD 或 OOD 构造失败：
    - 直接切成：
      - `train`: 80%
      - `valid`: 10%
      - `id_test`: 10%
- 若数据量很小，必须做最小样本保护：
  - 优先保证 `train >= 1`
  - `valid >= 1`
  - `id_test >= 1`
  - `ood_test` 可缺失
- 若存在时间顺序、实体分组或明显范围外区域：
  - 禁止直接随机切分
  - 优先使用时间后段、未见实体或极端取值区间作为 `id_test` / `ood_test`

### 情形二：只有 `train` 和 `test`

- 约定：
  - 原 `test` 默认映射为 `id_test`
  - 原 `train` 再按固定随机种子 `1314` 切成：
    - `train`: 90%
    - `valid`: 10%
- 若没有可靠 OOD 定义：
  - 可以不生成 `ood_test.csv`
  - 在 `metadata.yaml` 中明确写明“OOD split unavailable”

### 情形三：只有 `train`

- 若没有额外语义：
  - 用固定随机种子 `1314`
  - 从原 `train` 中切出：
    - `train`: 80%
    - `valid`: 10%
    - `id_test`: 10%
  - `ood_test` 可缺失
- 若原数据量过小：
  - 保守切分，至少保证 `train/valid/id_test` 各有 1 行

### 情形四：缺少 `valid`

- 若已有 `train` 与 `id_test` 或 `test`：
  - 从 `train` 中按固定随机种子 `1314` 再切 `10%` 作为 `valid`
- 不允许直接把 `id_test` 复制一份当 `valid`

### 情形五：缺少 `id_test`

- 若已有 `train` 与 `valid`：
  - 从 `train` 中再切一部分出来作为 `id_test`
  - 默认使用固定随机种子 `1314`
- 不允许把 `valid` 直接复制成 `id_test`

### 情形六：缺少 `ood_test`

- 若没有可靠 OOD 定义：
  - 可以不生成 `ood_test.csv`
  - 在 `metadata.yaml` 的 `description` 或 `resources/notes` 中明确说明原因
- 若存在清晰 OOD 定义：
  - 优先按未见实体、更晚时间段、范围外采样、极端参数区间来构造
- 若没有天然 OOD 且没有 ground truth，但数据来自单表：
  - 优先使用 `scripts/extract_ood_by_iterative_shrink.py`
  - 算法规则：
    - 逐维迭代
    - 每轮将某一维区间按长度的 `0.01%` 向内收缩
    - OOD 定义为落在收缩后超矩形之外的样本
    - 目标 OOD 占比约 `10%`
    - 若在达到目标前占比超过 `20%`，则判定失败，不生成 `ood_test.csv`
- 若数据集满足以下条件：
  - 已经整理出 `train.csv`
  - 有可执行的 `formula.py`
  - 目录结构满足 `<group>/<dataset>` 约定
  - 那么可以走历史 OOD 生成链路：
    1. `scripts/analyze_feature_distributions.py`
    2. `scripts/generate_ood_samples.py`
    3. `scripts/generate_metadata_2.py`
- 注意：
  - 这条链路当前对 `OOD` 采样与 `NMSE` 回填已经可以服务新数据家族
  - 但若你还想自动补自然语言描述，仍需额外提供适配当前数据的语义来源文件

### 情形七：缺少目标列说明

- 若用户未显式指定 target：
  - 默认使用最后一列作为目标列
  - 必须在 metadata 中记录该假设
- 若数据语义不清且最后一列并不可靠：
  - 不应静默猜测，应先停下来澄清

### 情形八：缺少真值公式

- 默认不生成 `formula.py`
- 不得为了凑齐目录结构而伪造公式文件

### 情形九：列头不一致

- 必须先统一列头和列顺序，再导出四个 CSV
- 不允许让不同 split 使用不同列头

### 情形十：不能做的事

- 不能通过复制同一份样本同时充当 `valid` 和 `id_test`
- 不能在没有依据时伪造 OOD 数据
- 不能默默丢掉非数值列后不记录
- 不能在 split 规则不明确时假装已经完成“benchmark 级复现”

## 强约束

- 输出 CSV 的列头必须一致，且必须包含目标列。
- 训练与测试数据必须是数值型表格，不能把原始字符串类别直接塞进去。
- `metadata.yaml` 里的 `dataset.target.name` 必须和 CSV 目标列名一致。
- `ood_test.csv` 不是必须项；若缺失，必须在 metadata 里明确说明原因。
- `formula.py` 不是必须项；若缺失，不得伪造。
- 若做了列重命名、单位换算、过滤或聚合，必须在 metadata 的 `description` 或 `resources/notes` 中说明。
- 若声明了 `ground_truth_formula.file`，该公式必须能被实际导入，并在一个或多个 split 上代入得到与目标列一致或近似一致的结果。

## 这个 skill 默认会做什么

- 帮你识别当前仓库 examples 风格的目标结构
- 帮你把异构源数据映射成统一 split 目录
- 帮你补最小可用 metadata
- 帮你用校验器检查产物
- 在 benchmark 风格数据上，优先判断历史脚本是否真的适配，再决定复用还是重写
- 默认不把 metadata 批处理脚本当成主能力
- 若存在真值公式，帮你实际代入公式验证它是否与数据一致

## 这个 skill 不会替你臆造什么

- 不会猜一个并不存在的真值公式
- 不会在没有依据时伪造 OOD 测试集
- 不会默认把非数值列强行 one-hot 后当成符号回归输入
- 不会在用户数据语义不清时静默猜测目标列

## 参考资料

- 目标格式：`references/examples_contract.md`
- 源数据映射：`references/source_mapping_patterns.md`
- metadata 规则：`references/metadata_rules.md`
- 当前可运行脚本：`scripts/`
- 旧 Feynman 兼容脚本：`references/legacy-feynman-scripts/`
