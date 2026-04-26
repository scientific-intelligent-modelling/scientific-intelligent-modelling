# uDSR 远端 5 分钟 smoke 结果

## 设置

- 远端机器：`iaaccn22`
- 远端 staging：`/home/zhangziwen/workplace/udsr_full_repro_repo`
- 结果目录：`/home/zhangziwen/experiment-results/udsr_remote_smoke_20260427`
- 环境：`sim_dso`
- 工具：`udsr`
- 算法口径：`udsr_trunk_dso_poly_gp_meld`
- 组件开关：`aif=false, dsr=true, lspt=false, gp_meld=true, linear_poly=true`
- seed：`1314`
- 单任务预算：`300s`
- workers：`2`

## 数据集

- `g0007_CRK22`
- `g0016_first_principles_newton`

## 结果

| dataset | status | seconds | equation | artifact | valid nmse | id nmse | ood nmse | progress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CRK22` | `timed_out` | `301.627` | yes | yes | `1.2729461186850967e-07` | `1.1559484998934026e-07` | `8.751285626179886e-07` | 5 |
| `first_principles_newton` | `timed_out` | `301.540` | yes | yes | initially null | initially null | initially null | 5 |

`first_principles_newton` 的最终 current-best 在第 5 分钟数值发散，导致原始 `result.json` 指标为空；第 4 分钟快照存在可有限评估候选：

- valid nmse：`0.017704448257210906`
- id nmse：`0.005097199788343279`
- ood nmse：`0.030123314937159886`

已修复 timeout recovery：当最后 current-best 不可有限评估时，回退到最近一个可有限评估的分钟级快照。
