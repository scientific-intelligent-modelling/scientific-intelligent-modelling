# E1 Result Digest

## Scope

- Input archive: `exp-planning/02.E1选择验证/e1_final_results_20260424-041046_clean/`
- Unit: one row per `(dataset, method, seed)` run.
- `valid_output=true` means the run has a final expression, no invalid canonical artifact flag, and finite `id_nmse` plus `ood_nmse`.
- `timed_out` runs are not automatically discarded; `budget_exhausted_with_output` runs remain usable for rank-fidelity analysis.

## Totals

- Total rows: `1400`
- Valid output rows: `1138`
- Rows with finite ID/OOD NMSE: `1138`
- Official launcher status: `{'ok': 458, 'timed_out': 942}`
- Internal result.json status: `{'ok': 461, 'timed_out': 939}`
- Status mismatch rows: `7`
- Timeout type: `{'budget_exhausted_with_output': 712, 'no_valid_output': 4, 'not_timeout': 456, 'partial_output': 219, 'unvalidated_expression': 9}`

## Files

- `e1_result_table.csv`: full 7 algorithm x 200 candidate table.
- `e1_method_summary.csv`: method-level status and validity summary.
- `e1_method_family_summary.csv`: method x family status and validity summary.
- `e1_dataset_coverage.csv`: per-dataset valid method coverage.
- `e1_status_mismatch.csv`: rows whose launcher status differs from internal `result.json` status.
- `e1_nonvalid_cases.csv`: rows excluded by `valid_output=false`.

## Method Summary

| method | total | valid_output | valid_output_rate | status_ok | status_timed_out | status_mismatch | budget_exhausted_with_output | partial_output | no_valid_output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| drsr | 200 | 101 | 0.505 | 0 | 200 | 0 | 101 | 98 | 0 |
| dso | 200 | 188 | 0.940 | 28 | 172 | 0 | 161 | 7 | 0 |
| gplearn | 200 | 106 | 0.530 | 0 | 200 | 0 | 106 | 92 | 0 |
| llmsr | 200 | 200 | 1.000 | 193 | 7 | 5 | 7 | 0 | 0 |
| pyoperon | 200 | 171 | 0.855 | 200 | 0 | 0 | 0 | 0 | 0 |
| pysr | 200 | 177 | 0.885 | 0 | 200 | 0 | 177 | 21 | 0 |
| tpsr | 200 | 195 | 0.975 | 37 | 163 | 2 | 160 | 1 | 4 |
