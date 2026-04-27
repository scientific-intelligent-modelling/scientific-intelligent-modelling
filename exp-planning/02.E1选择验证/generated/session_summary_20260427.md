# 2026-04-27 当前会话总结

本文档汇总本轮长会话中已经完成的实验规划、代码修复、远端运行、算法接入和当前待办。目标是让后续继续工作时可以直接从这里恢复上下文。

## 1. 当前总状态

- 当前 benchmark 主线已经从“继续盲跑”转为“消化 E1 结果、清洗 Master-100、切 Core/Reserve、做 rank-fidelity 验证”。
- E1 已经有核心原材料：`7 algorithms × 200 candidate × 1 seed`。
- 之前 `unknown family` 被确认不是数据问题，而是 dossier 统计脚本口径问题：应从 `dataset_dir` 解析真实 family。
- 当前所有已纳入快照能力的算法，runner 默认中间结果快照间隔已经统一为 `60s`。
- 对 timeout 的语义已经调整：如果预算耗尽后能恢复出可评估 best-so-far，最终 `status` 写 `ok`，而不是 `timed_out`。
- 当前工作区仍有一个未跟踪本地 smoke 目录：
  - `experiment-results/udsr_local_5min_smoke_20260427/`
  - 该目录没有被提交。

## 2. Benchmark 方案与实验路线

最初 review 的结论是：

- `664 -> 200 -> 100` 这条筛选链路合理，可以进入论文验证阶段。
- 但不应立刻冻结最终 Core-50。
- 当前 100 更准确地叫 `Master-100 draft`，需要先清洗、去重、切分和 rank-fidelity 验证。

确定的论文主线：

1. `Raw pool -> Valid pool -> Candidate-200 -> Master-100 -> Core-50 / Reserve-50`
2. `Candidate-200` 是 probe candidate selection。
3. `Master-100` 是 formal master reservoir construction。
4. `Core-50` 的核心论证不是“人工挑了 50 个”，而是“这 50 个高保真保留 Master-100 的算法排序和失败模式”。

关键风险已经记录：

- `srsd + strict` 占比高，可能被 reviewer 质疑挑 pathological cases。
- 存在跨来源语义重复，尤其是 Feynman 同公式不同命名。
- 初筛区分度主要来自 PySR / LLM-SR，需要用更多算法验证泛化排序。
- high gap 里有一部分是灾难性数值爆炸，需要作为 robustness/failure mode 显式处理，而不是隐藏。
- `quality_score=0` 的样本不应过多进入 Core-50。

建议的后续阶段：

1. 整理 E1 结果总表。
2. 清洗 `Master-100`，做 semantic dedup。
3. 只用非结果属性切 `Core-50 / Reserve-50`。
4. 用现有 7 算法 E1 结果做 rank-fidelity pilot。
5. 根据 7 算法结果决定是否补跑剩余算法。
6. 冻结 Core-50 后再跑正式 leaderboard。

## 3. E1 运行与结果整理

E1 设计：

- 数据集：`Candidate-200`
- seed：`1314`
- 算法：最初 7 算法
  - `gplearn`
  - `llmsr`
  - `pyoperon`
  - `drsr`
  - `pysr`
  - `dso`
  - `tpsr`

E1 wave 分配：

- `W1`: `gplearn`, `llmsr`
- `W2`: `pyoperon`, `drsr`
- `W3`: `pysr`
- `W4`: `dso`
- `W5`: `tpsr`

已生成过的核心资产包括：

- `check/generate_e1_formal_assets.py`
- `exp-planning/02.E1选择验证/generated/slices/`
- `exp-planning/02.E1选择验证/generated/params/`
- `exp-planning/02.E1选择验证/generated/remote_jobs/`
- `exp-planning/02.E1选择验证/hyperparams_snapshot_20260423/`

E1 结果修复和消化中已经明确：

- DRSR 曾有 best sample 参数没有进入 canonical artifact 的问题，已经补丁避免后续丢 `parameter_values`。
- gplearn 大量无效输出中有可恢复项，已做过批量修复。
- 部分剩余错误需要区分：
  - 工具集落盘/恢复问题，可以修。
  - 算法在 `3600s` 下仍没有可有限评估表达式或指标，属于当前预算和数据分布下无可用输出，不能强行伪装。
- `budget_exhausted_with_output` 被定义为预算耗尽但有可用 equation/artifact/metrics 的结果。

E1 结果后续应该优先做：

- 汇总 `dataset_id, method, seed, status, valid_output, timeout_type, id_nmse, ood_nmse, runtime, expression, complexity`
- 用 `timeout_type` 区分：
  - `not_timeout`
  - `budget_exhausted_with_output`
  - `partial_output`
  - `unvalidated_expression`
  - `no_valid_output`

## 4. 远端机器与环境

CPU 远端机器：

- `iaaccn22` 到 `iaaccn29`

GPU 远端机器：

- `iaaccn48` 到 `iaaccn55`
- 每台 6 到 8 张 3090，当前如无必要暂不使用。

远端重要目录：

- 远端仓库根目录：
  - `/home/zhangziwen/workplace/scientific-intelligent-modelling`
- 远端真实数据目录：
  - `/home/zhangziwen/sim-datasets-data`
- uDSR / RAG-SR staging 目录：
  - `/home/zhangziwen/workplace/udsr_full_repro_repo`

远端执行规则已经沉淀进 `AGENTS.md`：

- 本地到 `iaaccn23~29` 不稳定时，先同步到 `iaaccn22`，再从 `iaaccn22` 内网分发。
- 从 `iaaccn22` 访问其它机器时优先用 `10.10.100.23~29`。
- 远程复杂 Python/Shell 逻辑优先发脚本到 `/tmp/*.py`，不要内联长命令。
- 所有远程探测默认加 `timeout` 和 `ssh -o ConnectTimeout=...`。
- 临时 smoke CSV 里 `dataset_dir` 优先使用远端绝对路径。
- 远端真实数据目录不是仓库内的 `repo/sim-datasets-data`。

## 5. 显式数据契约

此前发现 TPSR 会产出超出当前数据维度的变量，例如 `x_9`。结论是：

- 这不是核心算法完全坏了。
- 更准确地说，是工具集集成层没有把数据维度契约接好。
- 对固定预训练词表算法，不能直接缩小预训练词表；应保留原始词表，在 wrapper 写出链或候选链过滤非法变量。

已经确定的规则：

- runner 必须给每个算法显式注入：
  - `n_features`
  - `feature_names`
  - `target_name`
- wrapper 在 `fit()` 入口必须校验：
  - `n_features == X.shape[1]`
  - `len(feature_names) == X.shape[1]`
  - `target_name` 非空
- wrapper 需要吸收这些元参数，不能透传给第三方库。

TPSR 已按正确思路修：

- 在 TPSR 解码/候选写出链过滤非法变量。
- 避免把固定预训练词表强行按当前数据集维度缩小。

## 6. 超参数对齐

本轮逐个对齐过的算法包括：

- `gplearn`
- `pyoperon`
- `llmsr`
- `drsr`
- `dso`
- `tpsr`
- `pysr`
- `e2esr`
- `qlattice`
- `imcts`
- `udsr`
- `ragsr`

关键调整：

- `gplearn` 不再只用四则运算，函数集扩展为：
  - `add, sub, mul, div, sqrt, log, sin, cos`
- `pyoperon` 改成更接近 benchmark 文献的显式口径：
  - `population_size=500`
  - `pool_size=500`
  - `max_length=50`
  - `tournament_size=5`
  - `allowed_symbols=add,mul,aq,exp,log,sin,tanh,constant,variable`
  - `max_evaluations=500000`
- `llmsr / drsr` 固定共用 LLM 配置：
  - `exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config`
  - 真实配置不进入 Git。
  - 仓库只保留 `benchmark_llm.config.example`。
  - `top_k` 改为 `30`。
- `drsr` 显式补上：
  - `max_params = 10`
  - `persist_all_samples = false`
- `dso` 改为显式 benchmark 口径：
  - `training.n_samples=2000000`
  - `batch_size=1000`
  - `epsilon=0.05`
  - `policy_optimizer.learning_rate=0.0005`
  - `entropy_weight=0.03`
  - `entropy_gamma=0.7`
- `tpsr` 对齐官方主配置后的 CPU 稳定口径。
- `pysr` 保持正式 benchmark 口径，区别于早期 probe 参数。

当前参数文件位置：

- `exp-planning/02.E1选择验证/generated/params/`
- `exp-planning/02.E1选择验证/hyperparams_snapshot_20260423/`

当前快照目录已经从 11 算法更新到 12 算法，并新增：

- `ragsr.json`

## 7. LLM 配置安全

已经明确：

- 真实 `benchmark_llm.config` 不能提交到 Git。
- 只提交安全示例：
  - `benchmark_llm.config.example`
- `.gitignore` 需要覆盖真实配置文件路径。
- `llmsr.json` 和 `drsr.json` 都指向固定配置路径。

之前已检查现有 `llm.config` 是否包含敏感字段，并落成安全方案。

## 8. 数据仓库与 LFS

处理过的相关问题：

- `sim-datasets-py` 指针问题已处理并提交。
- `sim-datasets-data` 出现过大量更改，原因与 LFS pointer / 真实数据落盘混杂有关。
- 明确规则：
  - 若 CSV 首行是 `version https://git-lfs.github.com/spec/v1`，说明还是 LFS pointer，不是真实 CSV。
  - 先补真实数据，再判断列名或 metadata 问题。
- 做过 LFS renormalize，将相关数据转为标准 LFS pointer。
- 数据修复真实规则：
  - 批量修了 `343` 个数据集的 `ood_test.csv` 表头，将最后一列 `target` 改成 metadata 里的真实 target name。
  - 通过 ModelScope + git-lfs 补齐最后 `3` 个 `srbench1.0/feynman` 数据集真实 CSV。

metadata 检查中发现：

- 多数 metadata 字段可自动补。
- 少量字段需要人工把关：
  - `dataset.citation`
  - `first_principles_kepler` 的 valid/id/ood nmse 等。

这些非关键字段曾被暂时搁置。

## 9. uDSR 接入

uDSR 当前接入口径：

- wrapper 名称保持：
  - `udsr_wrapper`
- tool name 保持：
  - `udsr`
- 当前不是 full uDSR。
- 当前是：
  - `DSO controller`
  - `LINEAR/poly token`
  - `GP-meld`
- 当前缺少：
  - AIF 递归化简
  - LSPT 预训练 encoder-controller

相关关键文件：

- `scientific_intelligent_modelling/algorithms/udsr_wrapper/wrapper.py`
- `tests/test_udsr_artifacts.py`
- `exp-planning/02.E1选择验证/generated/params/udsr.json`
- `exp-planning/02.E1选择验证/hyperparams_snapshot_20260423/udsr.json`

远端 5 分钟 smoke：

- 机器：`iaaccn22`
- 结果目录：
  - `/home/zhangziwen/experiment-results/udsr_remote_smoke_20260427`
- 数据集：
  - `CRK22`
  - `first_principles_newton`
- 两个任务都有：
  - `.udsr_current_best.json`
  - 外层 `progress/minute_0001.json` 到 `minute_0005.json`
  - 实验目录内 `progress/minute_0001.json` 到 `minute_0005.json`

结论：

- uDSR 已经有每分钟 best-so-far 落盘。
- 当前正式参数文件显式写了 `progress_snapshot_interval_seconds=60`。

## 10. RAG-SR 接入与修复

RAG-SR 已接入：

- `scientific_intelligent_modelling/algorithms/ragsr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/config/toolbox_config.json`
- `scientific_intelligent_modelling/config/envs_config.json`
- `check/check_ragsr.py`
- `tests/test_ragsr_wrapper.py`

调研结论：

- PyPI 版 `evolutionary-forest==0.2.5` 与官方 GitHub code 表现存在差异。
- 建过隔离环境：
  - `sim_ragsr_official_git`
- 对比中 `CRK22` 上 GitHub code 能出有效结果，而 PyPI 版曾 timeout 无指标。

已对齐官方默认参数：

- `categorical_encoding` 改为 `"Target"`。
- `fit()` 中如果未传 `categorical_features`，自动补 `[False] * n_features`。
- 这等价于官方示例里的 `np.zeros(X.shape[1])`。

RAG-SR 表达式回放曾修过：

- `Max/Min` 等表达式 normalizer/回放问题。

RAG-SR timeout 无 equation 的根因：

- wrapper 原来只有 `self.model.fit(...)` 正常返回后才提取 equation。
- 外层 `SymbolicRegressor` 到预算会硬 kill 子进程。
- 子进程被 kill 后，内存里的 model/population/hof 都丢失。
- 原来 RAG-SR 不在快照能力白名单里，runner 不会启动周期快照。
- 原来 runner 也没有 `_extract_ragsr_periodic_candidate(...)`。

已实现的修复：

- RAG-SR wrapper 在 EvolutionaryForest callback 中写：
  - `.ragsr_current_best.json`
- runner 增加：
  - `ragsr` 快照白名单
  - `_extract_ragsr_periodic_candidate(...)`
  - timeout 后从 `.ragsr_current_best.json` 或 `progress/minute_*.json` 恢复
- RAG-SR 默认也有 soft `time_limit`：
  - 如果传 `timeout_in_seconds` 且未显式传 `time_limit`
  - 则 `time_limit = timeout_in_seconds - 5`

远端 CRK22 5 分钟 smoke：

- 输出根：
  - `/tmp/ragsr_snapshot_smoke_20260427_155749`
- 最终 status 当时仍是 `timed_out`，但已经有：
  - equation
  - canonical artifact
  - valid/id/ood 指标
- 指标：
  - `valid_nmse=0.010888962524466972`
  - `id_nmse=0.017804896453143804`
  - `ood_nmse=368.2797217911007`
- 快照：
  - `minute_0001.json` 到 `minute_0005.json`

之后 timeout 语义已经修正，新结果会把这类可恢复 timeout 写为 `status=ok`。

## 11. 每分钟快照统一

用户明确要求：

- 最新两个算法以及所有算法都改成 1 分钟快照。

已完成：

- runner 默认所有支持快照的算法未显式传参时，返回 `60s`。
- 修复大小写匹配：
  - `QLattice / qlattice`
  - `iMCTS / imcts`
- 参数快照目录中所有算法显式写：
  - `progress_snapshot_interval_seconds = 60`
- 新增 RAG-SR 参数快照：
  - `exp-planning/02.E1选择验证/generated/params/ragsr.json`
  - `exp-planning/02.E1选择验证/hyperparams_snapshot_20260423/ragsr.json`

当前支持快照的算法：

- `llmsr`
- `drsr`
- `pysr`
- `dso`
- `udsr`
- `pyoperon`
- `gplearn`
- `e2esr`
- `iMCTS`
- `tpsr`
- `QLattice`
- `ragsr`

验证过：

- 24 个参数 JSON 文件的 `progress_snapshot_interval_seconds` 都是 `60`。
- 相关测试通过：
  - `tests/test_benchmark_progress_snapshots.py`
  - `tests/test_ragsr_wrapper.py`
  - `tests/test_udsr_artifacts.py`

## 12. Timeout 状态语义修复

用户希望：

- timeout 后如果已经恢复出结果，不要显示为超时失败。

修复前：

- `run_benchmark_task()` 捕获 `TimeoutError` 后直接固定：
  - `status = "timed_out"`
  - `error = TimeoutError(...)`
- 即使后续恢复出 equation/artifact/valid/id/ood，也仍然显示 `timed_out`。

修复后：

- 如果 timeout 后成功恢复出：
  - equation
  - canonical artifact
  - valid metrics
  - id_test metrics
  - ood_test metrics
- 则最终 `result.json` 写：
  - `status = "ok"`
  - `error = null`
  - `budget_exhausted = true`
  - `timeout_type = "budget_exhausted_with_output"`
  - `recovered_from_timeout = true`
  - `termination_reason = "budget_exhausted_with_output"`
  - `raw_timeout_error = 原始 TimeoutError`

如果 timeout 后没有可用输出，则仍然写：

- `status = "timed_out"`
- `error = TimeoutError(...)`
- `budget_exhausted = true`
- `timeout_type = "no_valid_output"`
- `recovered_from_timeout = false`

这不是“把所有 timeout 都伪装成 ok”。只有恢复出完整可评估输出才会变成 `ok`。

相关代码：

- `scientific_intelligent_modelling/benchmarks/runner.py`
- `check/launch_e1_benchmark.py`
- `check/build_e1_result_digest.py`
- `tests/test_benchmark_runner.py`

验证：

- `PYTHONPATH=. pytest -q tests/test_benchmark_runner.py tests/test_benchmark_progress_snapshots.py tests/test_ragsr_wrapper.py`
- 结果：
  - `34 passed`

注意：

- 该修复只影响之后新跑的结果。
- 旧的 `result.json` 不会自动改状态。
- 如果需要旧结果也采用新显示口径，需要做一次回填。

## 13. 本轮重要提交

最近相关提交：

- `6680700 [fix] 调整预算耗尽恢复结果状态`
  - 可恢复 timeout 写为 `status=ok`
  - 增加 `budget_exhausted / timeout_type / recovered_from_timeout / raw_timeout_error`
  - 更新 E1 launcher 和 digest
- `d5b92ca [update] 统一算法分钟级快照间隔`
  - 所有快照能力算法默认 `60s`
  - 补 RAG-SR 参数快照
  - 文档更新到 12 算法
- `f2db559 [fix] 为 RAG-SR 增加分钟级快照恢复`
  - RAG-SR callback 写 `.ragsr_current_best.json`
  - runner 支持 RAG-SR 快照恢复
- `e6df0f9 [fix] 对齐 RAG-SR 官方默认参数`
  - `categorical_encoding="Target"`
  - 自动补数值特征 mask
- `93ac77c [fix] 修复 RAG-SR MaxMin 表达式回放`
- `36fb469 [fix] 修复 RAG-SR 环境与序列化验收`
- `4ba65d3 [update] 接入 RAG-SR wrapper 初版`
- `ef1886c [update] 固定 uDSR trunk 组件口径`
- `ae8c49c [fix] 修复 uDSR 超时恢复并记录 full 复现计划`
- `30c3bc7 [fix] 对齐 uDSR trunk 论文口径`
- `ef55fd9 [update] 接入 uDSR 符号回归工具`

## 14. 当前建议的下一步

最小下一步闭环：

1. 把最新代码同步到远端。
2. 用 `RAG-SR` 和 `uDSR` 各做一次短 smoke，确认新 timeout 语义：
   - 可恢复预算耗尽结果应显示 `status=ok`
   - 同时有 `budget_exhausted=true`
3. 如需旧结果统一显示口径，写一个只改归档 JSON 的回填脚本：
   - 读取旧 `result.json`
   - 若已有 equation/artifact/valid/id/ood
   - 将 `status` 从 `timed_out` 改成 `ok`
   - 增加 `budget_exhausted=true`
   - 增加 `timeout_type=budget_exhausted_with_output`
   - 保留原始 timeout 到 `raw_timeout_error`
4. 继续 E1 消化：
   - 生成 7 算法总表
   - 清洗 Master-100
   - 切 Core-50 / Reserve-50
   - 做 rank-fidelity pilot

## 15. 当前仓库状态

最后一次检查时：

- Git 已提交本轮代码改动。
- 工作区唯一未跟踪项：
  - `experiment-results/udsr_local_5min_smoke_20260427/`
- 该目录是本地 smoke 产物，之前一直没有提交。

