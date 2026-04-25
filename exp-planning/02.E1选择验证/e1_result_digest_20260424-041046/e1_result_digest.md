# E1 Result Digest

## Scope

- Input archive: `exp-planning/02.E1选择验证/e1_final_results_20260424-041046_clean/`
- Unit: one row per `(dataset, method, seed)` run.
- `valid_output=true` means the run has a final expression, no invalid canonical artifact flag, and finite `id_nmse` plus `ood_nmse`.
- `timed_out` runs are not automatically discarded; `budget_exhausted_with_output` runs remain usable for rank-fidelity analysis.

## Totals

- Total rows: `1400`
- Valid output rows: `1385`
- Rows with finite ID/OOD NMSE: `1385`
- Official launcher status: `{'ok': 462, 'timed_out': 938}`
- Internal result.json status: `{'ok': 462, 'timed_out': 938}`
- Status mismatch rows: `0`
- Timeout type: `{'budget_exhausted_with_output': 928, 'no_valid_output': 2, 'not_timeout': 462, 'partial_output': 8}`
- Dataset identity status: `{'exact_match': 1396, 'temp_copy_equivalent': 4}`

## Files

- `e1_result_table.csv`: full 7 algorithm x 200 candidate table.
- `e1_method_summary.csv`: method-level status and validity summary.
- `e1_method_family_summary.csv`: method x family status and validity summary.
- `e1_dataset_coverage.csv`: per-dataset valid method coverage.
- `e1_status_mismatch.csv`: rows whose launcher status differs from internal `result.json` status.
- `e1_nonvalid_cases.csv`: rows excluded by `valid_output=false`.
- `e1_dataset_identity_audit.csv`: expected Candidate-200 dataset identity versus actual `result.json` dataset.
- `e1_dataset_identity_mismatch.csv`: rows excluded because a same-name dataset collision wrote the wrong result.
- `e1_dataset_identity_rerun_tasks.csv`: compact rerun checklist for wrong-dataset collision rows.

## Method Summary

| method | total | valid_output | valid_output_rate | status_ok | status_timed_out | status_mismatch | budget_exhausted_with_output | partial_output | no_valid_output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| drsr | 200 | 200 | 1.000 | 0 | 200 | 0 | 200 | 0 | 0 |
| dso | 200 | 198 | 0.990 | 27 | 173 | 0 | 171 | 2 | 0 |
| gplearn | 200 | 200 | 1.000 | 0 | 200 | 0 | 200 | 0 | 0 |
| llmsr | 200 | 200 | 1.000 | 199 | 1 | 0 | 1 | 0 | 0 |
| pyoperon | 200 | 195 | 0.975 | 200 | 0 | 0 | 0 | 0 | 0 |
| pysr | 200 | 197 | 0.985 | 0 | 200 | 0 | 197 | 3 | 0 |
| tpsr | 200 | 195 | 0.975 | 36 | 164 | 0 | 159 | 3 | 2 |
