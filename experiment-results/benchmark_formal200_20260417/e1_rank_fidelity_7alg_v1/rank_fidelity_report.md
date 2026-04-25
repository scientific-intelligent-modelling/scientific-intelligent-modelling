# E1 7算法 Rank-Fidelity Pilot

## 口径

- 输入结果：E1 `7 algorithms × 200 candidate × 1 seed` 当前 digest。
- 评估集合：`Clean-Master-100`、由 clean master 重新切出的 `Core-50 / Reserve-50`。
- 单 run 分数：`0.5 * log10(id_nmse) + 0.5 * log10(ood_nmse)`，范围裁剪到 `[-12, 12]`。
- 缺失、非法或非完整指标按惩罚分 `12.0` 处理；分数越低越好。
- 切分本身不使用 E1 结果；E1 只用于事后验证 rank fidelity。

## Core/Reserve 对 Master-100 的保真度

| subset | Spearman | Kendall | pairwise win agreement | score MAE | score RMSE |
|---|---:|---:|---:|---:|---:|
| `core50` | 1.0000 | 1.0000 | 1.0000 | 0.2620 | 0.2954 |
| `reserve50` | 1.0000 | 1.0000 | 1.0000 | 0.2620 | 0.2954 |

## Core-50 相对随机 50 的位置

`at least as good` 包含并列；`strictly better` 不包含并列。排名相关指标中随机 50 经常也达到 1.0，因此这里重点看 aggregate score error。

| baseline | metric | Core at least as good | Core strictly better | baseline mean | p05 | p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `random50` | `spearman` | 1.0000 | 0.0810 | 0.9970 | 0.9643 | 1.0000 | 1.0000 |
| `random50` | `kendall` | 1.0000 | 0.0810 | 0.9921 | 0.9048 | 1.0000 | 1.0000 |
| `random50` | `pairwise_win_agreement` | 1.0000 | 0.0810 | 0.9960 | 0.9524 | 1.0000 | 1.0000 |
| `random50` | `aggregate_score_mae` | 0.8360 | 0.8360 | 0.4136 | 0.2024 | 0.4009 | 0.6850 |
| `random50` | `aggregate_score_rmse` | 0.8880 | 0.8880 | 0.5044 | 0.2500 | 0.4903 | 0.8198 |
| `family_stratified_random50` | `spearman` | 1.0000 | 0.0580 | 0.9979 | 0.9643 | 1.0000 | 1.0000 |
| `family_stratified_random50` | `kendall` | 1.0000 | 0.0580 | 0.9944 | 0.9048 | 1.0000 | 1.0000 |
| `family_stratified_random50` | `pairwise_win_agreement` | 1.0000 | 0.0580 | 0.9972 | 0.9524 | 1.0000 | 1.0000 |
| `family_stratified_random50` | `aggregate_score_mae` | 0.7530 | 0.7530 | 0.3553 | 0.1732 | 0.3410 | 0.5845 |
| `family_stratified_random50` | `aggregate_score_rmse` | 0.8230 | 0.8230 | 0.4354 | 0.2199 | 0.4169 | 0.7179 |

## 方法分数

| method | master score | core score | reserve score | master valid | core valid | reserve valid |
|---|---:|---:|---:|---:|---:|---:|
| `drsr` | 2.2542 | 1.9746 | 2.5339 | 100 | 50 | 50 |
| `dso` | -2.6636 | -2.2652 | -3.0620 | 98 | 49 | 49 |
| `gplearn` | -1.6696 | -1.2715 | -2.0676 | 100 | 50 | 50 |
| `llmsr` | 2.6577 | 2.3268 | 2.9885 | 100 | 50 | 50 |
| `pyoperon` | -0.7509 | -0.6506 | -0.8512 | 96 | 49 | 47 |
| `pysr` | -4.9426 | -4.6335 | -5.2517 | 99 | 49 | 50 |
| `tpsr` | 0.8301 | 0.8481 | 0.8121 | 99 | 50 | 49 |

## Clean-Master-100 中的非完整 run

- `pyoperon` / `g0012` / `feynman-i.26.2` / `not_timeout`
- `pyoperon` / `g0078` / `III.21.20_2_0` / `not_timeout`
- `dso` / `g0130` / `Keijzer-10` / `partial_output`
- `pysr` / `g0130` / `Keijzer-10` / `partial_output`
- `dso` / `g0141` / `Keijzer-15` / `partial_output`
- `tpsr` / `g0175` / `feynman-bonus.2` / `partial_output`
- `pyoperon` / `g0184` / `III.15.14_1_0` / `not_timeout`
- `pyoperon` / `g0194` / `II.21.32_2_0` / `not_timeout`
