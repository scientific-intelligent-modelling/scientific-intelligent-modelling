---
name: heterogeneous-data-to-examples
description: 当需要把用户提供的异构数据源整理成当前仓库 examples 风格的数据集目录时使用。适用于 CSV/Excel/JSON/多文件导出转成 train.csv、valid.csv、id_test.csv、ood_test.csv、metadata.yaml，并用校验脚本检查输出是否符合当前 pipeline 的真实约定。
---

# Heterogeneous Data To Examples

用于把用户给的异构数据整理成当前仓库 `examples/` 风格的数据集目录。

## 何时使用

- 用户给了一份或多份异构数据，想接成当前项目可直接消费的数据集。
- 用户说“整理成和 examples 一样的格式”。
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
7. 若已知真值公式，再写 `formula.py`。
8. 跑校验：
   - `python3 tools/example_dataset_onboarder/scripts/validate_example_dataset.py --dataset-dir <dir>`

## 强约束

- 输出 CSV 的列头必须一致，且必须包含目标列。
- 训练与测试数据必须是数值型表格，不能把原始字符串类别直接塞进去。
- `metadata.yaml` 里的 `dataset.target.name` 必须和 CSV 目标列名一致。
- 若没有可靠的 OOD 定义，不要伪造；可以生成带表头的空 `ood_test.csv`，并在 metadata 里写清楚。
- 若做了列重命名、单位换算、过滤或聚合，必须在 metadata 的 `description` 或 `resources/notes` 中说明。

## 这个 skill 默认会做什么

- 帮你识别当前仓库 examples 风格的目标结构
- 帮你把异构源数据映射成统一 split 目录
- 帮你补最小可用 metadata
- 帮你用校验器检查产物

## 这个 skill 不会替你臆造什么

- 不会猜一个并不存在的真值公式
- 不会在没有依据时伪造 OOD 测试集
- 不会默认把非数值列强行 one-hot 后当成符号回归输入
- 不会在用户数据语义不清时静默猜测目标列

## 参考资料

- 目标格式：`references/examples_contract.md`
- 源数据映射：`references/source_mapping_patterns.md`
- metadata 规则：`references/metadata_rules.md`
