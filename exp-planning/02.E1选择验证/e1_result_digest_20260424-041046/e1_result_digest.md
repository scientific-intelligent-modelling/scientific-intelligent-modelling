# E1 Result Digest

## Scope

- Input archive: `exp-planning/02.E1选择验证/e1_final_results_20260424-041046_clean/`
- Unit: one row per `(dataset, method, seed)` run.
- `valid_output=true` means the run has a final expression, no invalid canonical artifact flag, and finite `id_nmse` plus `ood_nmse`.
- `timed_out` runs are not automatically discarded; `budget_exhausted_with_output` runs remain usable for rank-fidelity analysis.

## Totals

- Total rows: `1400`
- Valid output rows: `1181`
- Rows with finite ID/OOD NMSE: `1230`
- Official launcher status: `{'ok': 458, 'timed_out': 942}`
- Internal result.json status: `{'ok': 461, 'timed_out': 939}`
- Status mismatch rows: `7`
- Timeout type: `{'budget_exhausted_with_output': 765, 'invalid_output': 39, 'no_valid_output': 4, 'not_timeout': 456, 'partial_output': 129, 'unvalidated_expression': 7}`
- Dataset identity status: `{'exact_match': 1337, 'temp_copy_equivalent': 4, 'wrong_dataset_collision': 59}`

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
| drsr | 200 | 92 | 0.460 | 0 | 200 | 0 | 92 | 98 | 0 |
| dso | 200 | 185 | 0.925 | 28 | 172 | 0 | 159 | 7 | 0 |
| gplearn | 200 | 191 | 0.955 | 0 | 200 | 0 | 191 | 2 | 0 |
| llmsr | 200 | 191 | 0.955 | 193 | 7 | 5 | 1 | 0 | 0 |
| pyoperon | 200 | 166 | 0.830 | 200 | 0 | 0 | 0 | 0 | 0 |
| pysr | 200 | 169 | 0.845 | 0 | 200 | 0 | 169 | 21 | 0 |
| tpsr | 200 | 187 | 0.935 | 37 | 163 | 2 | 153 | 1 | 4 |
