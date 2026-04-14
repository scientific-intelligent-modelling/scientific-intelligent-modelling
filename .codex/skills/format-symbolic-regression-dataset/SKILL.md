---
name: format-symbolic-regression-dataset
description: 当需要把符号回归数据源规范化为当前仓库统一数据集目录时使用。适用于 CSV/TSV/Excel/JSON/多文件导出或 benchmark 原始数据，整理成 train.csv、valid.csv、id_test.csv、ood_test.csv、metadata.yaml，并用校验脚本检查输出是否符合当前 pipeline 的真实约定。
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
- `ood_test.csv`
- `metadata.yaml`

若已知真值公式，可额外包含：

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
   - `scripts/generate_metadata_2.py`：批量补 metadata 与公式一致性信息
   - `scripts/force_update_desp.py` / `scripts/debug_metadata.py`：用于排查或覆盖描述字段
8. 这些脚本目前是“历史后处理脚本”，不是通用格式化工具。
   - `analyze_feature_distributions.py` 相对通用：适合已整理成 `train.csv` 的目录
   - `generate_ood_samples.py` 明显依赖 `formula.py + feature_distributions.csv + feynman` 目录约定
   - `generate_metadata_2.py` 明显依赖 `feynman/srbench_feynman.csv` 与 `metadata_2.yaml`
   - `force_update_desp.py` / `debug_metadata.py` 是特定数据集排障脚本，不适合作为主流程
9. 因此默认策略应是：
   - 先用当前 skill 的通用流程完成 split、列名、基础 metadata
   - 只有当目录结构与旧 benchmark 假设一致时，才局部复用这些脚本
   - 对 SRBench 2025 这类新数据家族，不要直接运行这些脚本，除非先做针对性改造
10. 使用这些脚本前必须先读源码，确认列名、目录层级、目标列命名与当前数据一致。
11. 不要在语义不匹配的数据上直接运行。
12. 若已知真值公式，再写 `formula.py`。
13. 若存在 `ground_truth_formula.file`，必须做公式代入校验：
   - `python3 tools/example_dataset_onboarder/scripts/validate_example_dataset.py --dataset-dir <dir> --verify-formula`
14. 跑校验：
   - `python3 tools/example_dataset_onboarder/scripts/validate_example_dataset.py --dataset-dir <dir>`

## 强约束

- 输出 CSV 的列头必须一致，且必须包含目标列。
- 训练与测试数据必须是数值型表格，不能把原始字符串类别直接塞进去。
- `metadata.yaml` 里的 `dataset.target.name` 必须和 CSV 目标列名一致。
- 若没有可靠的 OOD 定义，不要伪造；可以生成带表头的空 `ood_test.csv`，并在 metadata 里写清楚。
- 若做了列重命名、单位换算、过滤或聚合，必须在 metadata 的 `description` 或 `resources/notes` 中说明。
- 若声明了 `ground_truth_formula.file`，该公式必须能被实际导入，并在一个或多个 split 上代入得到与目标列一致或近似一致的结果。

## 这个 skill 默认会做什么

- 帮你识别当前仓库 examples 风格的目标结构
- 帮你把异构源数据映射成统一 split 目录
- 帮你补最小可用 metadata
- 帮你用校验器检查产物
- 在 benchmark 风格数据上，优先判断历史脚本是否真的适配，再决定复用还是重写
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
- 历史后处理脚本：`scripts/`
