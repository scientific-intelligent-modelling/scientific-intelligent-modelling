# 远端同步与显式数据契约 Smoke 报告（2026-04-23）

## 1. 目标

本报告记录两件事：

1. 将“显式数据契约”相关代码同步到 `iaaccn22~29`。
2. 在远端做代表性 smoke，确认新契约不会破坏 benchmark 运行链路。

这里的“显式数据契约”指：

- `n_features`
- `feature_names`
- `target_name`

由 `runner.build_runner_params()` 统一注入，再由各 wrapper 在 `fit()` 入口校验。

---

## 2. 关键前提

### 2.1 远端真实代码根

这次排查确认，远端实际被 `conda` 环境 import 的项目根是：

```text
/home/zhangziwen/projects/scientific-intelligent-modelling
```

不是之前常用但本轮未实际生效的：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling
```

因此本次同步和 smoke 一律以 `projects/` 这条路径为准。

### 2.2 同步文件范围

本次只定点同步“显式数据契约”相关核心文件：

- `scientific_intelligent_modelling/benchmarks/runner.py`
- `scientific_intelligent_modelling/algorithms/base_wrapper.py`
- `scientific_intelligent_modelling/algorithms/pysr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/gplearn_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/pyoperon_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/dso_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/llmsr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/drsr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/QLattice_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py`
- `scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py`

---

## 3. 同步结果

### 3.1 同步范围

已同步到：

- `iaaccn22`
- `iaaccn23`
- `iaaccn24`
- `iaaccn25`
- `iaaccn26`
- `iaaccn27`
- `iaaccn28`
- `iaaccn29`

### 3.2 验收方式

对上述 12 个核心文件逐台做 `sha256` 校验。

### 3.3 验收结论

结果为：

- `VERIFY_OK iaaccn22`
- `VERIFY_OK iaaccn23`
- `VERIFY_OK iaaccn24`
- `VERIFY_OK iaaccn25`
- `VERIFY_OK iaaccn26`
- `VERIFY_OK iaaccn27`
- `VERIFY_OK iaaccn28`
- `VERIFY_OK iaaccn29`

结论：

> `iaaccn22~29` 8 台 CPU 机上，这次显式数据契约相关代码已经完全对齐。

---

## 4. 代表性 Smoke 设计

为了验证“新契约注入 + wrapper 入口校验”没有破坏远端运行链路，本次选了 3 个代表性算法：

- `pysr`
- `drsr`
- `tpsr`

对应机器：

- `pysr @ iaaccn23`
- `drsr @ iaaccn26`
- `tpsr @ iaaccn25`

### 4.1 数据集

- `pysr`：
  - `/home/zhangziwen/sim-datasets-data/benchmark-splits/core50/nguyen/Nguyen-11`
- `drsr`：
  - `/home/zhangziwen/sim-datasets-data/benchmark-splits/core50/nguyen/Nguyen-11`
- `tpsr`：
  - `/home/zhangziwen/sim-datasets-data/benchmark-splits/core50/llm-srbench/lsrtransform/III.13.18_0_0`

### 4.2 预算

- `timeout_in_seconds = 300`
- `progress_snapshot_interval_seconds = 60`

`tpsr` 额外带：

- `cpu = true`

---

## 5. Smoke 结果

## 5.1 `pysr @ iaaccn23`

输出根目录：

```text
/home/zhangziwen/projects/scientific-intelligent-modelling/experiments/contract_smoke_pysr_20260423
```

最终结果：

- `status = ok`
- `seconds = 76.183`
- `equation = True`
- `canonical_artifact = True`
- `valid = True`
- `id_test = True`
- `ood_test = True`

结论：

> 显式数据契约注入后，`pysr` 远端 benchmark 链路正常，结果完整。

## 5.2 `drsr @ iaaccn26`

输出根目录：

```text
/home/zhangziwen/projects/scientific-intelligent-modelling/experiments/contract_smoke_drsr_20260423
```

最终结果：

- `status = ok`
- `seconds = 35.812`
- `equation = True`
- `canonical_artifact = True`
- `valid = True`
- `id_test = True`
- `ood_test = True`

说明：

- 这条任务在 `60s` 前自然结束，因此本轮不用于验证 `minute_0001.json`
- 但最终结果链路完全正常

结论：

> 显式数据契约注入后，`drsr` 远端 benchmark 链路正常，结果完整。

## 5.3 `tpsr @ iaaccn25`

第一次 smoke 暴露了一个额外问题：

- `sim_tpsr` 仍然是 Python `3.9`
- `BaseWrapper` 中的 `int | None` 注解会直接触发 `TypeError`

已补充兼容修复：

- 提交：`c027761 [fix] 兼容 Python3.9 的基类类型注解`

随后在**全新目录**重跑，避免旧结果污染判断。

第二次 smoke 输出根目录：

```text
/home/zhangziwen/projects/scientific-intelligent-modelling/experiments/contract_smoke_tpsr_20260423_v2
```

### 分钟级快照

- `minute_0001.json = True`
- `minute_status = ok`
- `minute_has_equation = True`
- `minute_has_artifact = True`
- `minute_has_valid = True`
- `minute_has_id = True`
- `minute_has_ood = True`

### 最终结果

- `status = timed_out`
- `seconds = 311.071`
- `equation = True`
- `canonical_artifact = True`
- `valid = True`
- `id_test = True`
- `ood_test = True`

结论：

> 显式数据契约注入后，`tpsr` 的分钟级快照链和最终结果链都正常。

---

## 6. 总结

本轮“远端同步 + 代表性 smoke”可以给出如下结论：

1. `iaaccn22~29` 已全部同步到这次显式数据契约改动对应代码。
2. `pysr / drsr / tpsr` 三个代表性算法在远端真实 benchmark 跑法下均已通过。
3. 新契约没有破坏：
   - 算法启动
   - 周期快照
   - 最终结果落盘
   - `valid / id_test / ood_test` 指标恢复
4. `tpsr` 额外暴露的 Python `3.9` 类型注解兼容问题已经修复，不再构成阻塞。

最终判断：

> 这次“显式数据契约注入 + wrapper 入口校验”已经具备远端继续放大的工程条件。
