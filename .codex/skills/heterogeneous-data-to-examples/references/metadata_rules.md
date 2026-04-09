# Metadata 规则

## 必填

- `dataset.name`
- `dataset.description`
- `dataset.splits.*.file`
- `dataset.splits.*.samples`
- `dataset.features`
- `dataset.target.name`
- `dataset.target.type`

## 强烈建议填写

- `dataset.features[*].description`
- `dataset.features[*].train_range`
- `dataset.features[*].ood_range`
- `dataset.target.description`
- `dataset.target.train_range`
- `dataset.target.ood_range`
- `dataset.resources`
- `dataset.license`

## 描述原则

- 说明数据来源。
- 说明做了哪些筛选、清洗、重采样、聚合。
- 说明 split 依据，尤其是 OOD 的定义。
- 若 OOD 为空，明确写原因。

## 什么时候可以不写 formula.py

- 真值公式未知
- 数据来自真实实验而不是合成 benchmark
- 只能观测输入输出，无法给出解析式

## 常见错误

- metadata 的目标列名和 CSV 不一致
- features 顺序和 CSV 实际顺序不一致
- split 样本数写错
- 把原始文本字段直接放进建模 CSV
- OOD 和训练范围完全重叠，却声称是 OOD
