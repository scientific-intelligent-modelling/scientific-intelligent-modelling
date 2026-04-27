# E1 Final Results Current 2026-04-26

本目录是当前 E1 修复后的最终落盘结果，只包含最终统计、digest 和剩余无效项说明，不包含远端补跑过程日志。

## 总览

- 有效输出：`1385 / 1400`
- 剩余无效：`15`
- wrong_dataset_collision：`0`
- 修复提交：`9a6b4fd [fix] 修复E1缺失结果与补跑回填`

## 文件说明

- `method_validity_summary.csv`：7 个算法的有效输出计数。
- `remaining_nonvalid_cases.csv`：剩余 15 条未形成完整指标的 run。
- `remaining_nonvalid_explanation.md`：剩余无效项的解释和代表例子。
- `digest/e1_result_table.csv`：当前完整 1400-run 结果总表。
- `digest/e1_dataset_algorithm_nmse_table.csv`：当前 `200×7` 数据集-算法 NMSE 明细表，包含 `family` 和 `srsd_variant` 标签。
- `digest/e1_result_digest.md`：当前 digest 文本汇总。
- `digest/e1_nonvalid_cases.csv`：digest 口径下的剩余无效项。
- `digest/e1_dataset_identity_audit.csv`：数据集身份审计结果。
- `e1_repair_summary_20260426.md`：本轮修复汇总。

## 口径

剩余 15 条不是数据错配或同步问题，而是算法输出在当前 3600s E1 预算下没有形成完整、有限、可比较的 valid/id/ood 指标。
