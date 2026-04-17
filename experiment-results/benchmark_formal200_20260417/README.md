# benchmark_formal200_20260417

这个目录汇总了两套实验结果：

1. 单种子探针结果（seed=1314，覆盖 664 数据集，方法为 PySR + LLM-SR）
2. 双种子正式结果（seeds=520/521，覆盖 200 候选数据集，方法为 PySR + LLM-SR）

## 文件说明

- `one_seed_probe_task_results.csv`：单种子探针任务级总表（1328 行）
- `one_seed_probe_dataset_compare.csv`：单种子探针按 dataset 对齐后的方法对照表（664 行）
- `one_seed_probe_summary.md`：单种子探针摘要
- `two_seed_formal_task_results.csv`：双种子正式任务级总表（800 行）
- `two_seed_formal_dataset_method_summary.csv`：双种子正式按 dataset×method 聚合后的摘要
- `two_seed_formal_dataset_compare.csv`：双种子正式按 dataset 对齐后的方法对照表（200 行）
- `two_seed_formal_summary.md`：双种子正式摘要

## 关键口径

- 所有正式结果都基于 `task_status.jsonl + experiment_dir/result.json` 汇总，不使用容易互相覆盖的外层 `result.json`。
- 单种子 probe 结果中额外补了 `is_formal200_candidate` 与候选池标签，便于和 200 候选集联动分析。
