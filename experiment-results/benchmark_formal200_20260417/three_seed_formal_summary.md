# 三 seed 正式结果汇总（统一 1h 口径）

- 任务数：`1200`
- 数据集对照行数：`200`

## 方法 × seed × 状态

### `llmsr` / `seed=520`
- `ok`: `196`
- `timed_out`: `4`

### `llmsr` / `seed=521`
- `ok`: `192`
- `timed_out`: `8`

### `llmsr` / `seed=522`
- `ok`: `194`
- `timed_out`: `6`

### `pysr` / `seed=520`
- `timed_out`: `200`

### `pysr` / `seed=521`
- `timed_out`: `200`

### `pysr` / `seed=522`
- `timed_out`: `200`

## 指标覆盖

### `llmsr`
- `equation_only_or_partial`: `6`
- `full_id_ood`: `594`

### `pysr`
- `equation_only_or_partial`: `97`
- `full_id_ood`: `503`
