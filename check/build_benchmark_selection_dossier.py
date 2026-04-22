#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/family/workplace/scientific-intelligent-modelling")
RESULT_ROOT = REPO_ROOT / "experiment-results" / "benchmark_formal200_20260417"
DEV_CORE_ROOT = RESULT_ROOT / "dev_core_split_v1"
CANDIDATE_JSON = Path("/tmp/candidate_seeds_200_v3.json")
OUTPUT_ROOT = REPO_ROOT / "experiment-results" / "benchmark_selection_dossier_20260422"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _f(v: Any) -> float | None:
    if v in ("", None, "None"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _i(v: Any) -> int | None:
    if v in ("", None, "None"):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _median(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [_f(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _fmt_float(v: float | None, digits: int = 6) -> str:
    if v is None:
        return "NA"
    return f"{v:.{digits}g}"


def _fmt_counter(counter: Counter[str]) -> str:
    return ", ".join(f"`{k}={v}`" for k, v in sorted(counter.items()))


def _probe_stage_payload() -> dict[str, Any]:
    task_rows = _load_csv(RESULT_ROOT / "one_seed_probe_task_results.csv")
    compare_rows = _load_csv(RESULT_ROOT / "one_seed_probe_dataset_compare.csv")
    candidate_obj = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    candidate_rows = candidate_obj["pool_A"] + candidate_obj["pool_B"] + candidate_obj["pool_C"]

    task_status = Counter((r["method"], r["task_status"]) for r in task_rows)
    comparable_rows = []
    family_wins: dict[str, Counter[str]] = defaultdict(Counter)
    win_counter = Counter()
    for row in compare_rows:
        keys = ["pysr_id_nmse", "pysr_ood_nmse", "llmsr_id_nmse", "llmsr_ood_nmse"]
        if any(row.get(k) in ("", None, "None") for k in keys):
            continue
        comparable_rows.append(row)
        py = (float(row["pysr_id_nmse"]) + float(row["pysr_ood_nmse"])) / 2.0
        ll = (float(row["llmsr_id_nmse"]) + float(row["llmsr_ood_nmse"])) / 2.0
        side = "pysr" if py < ll else "llmsr"
        win_counter[side] += 1
        family = row.get("candidate_family") or "unknown"
        family_wins[family][side] += 1

    candidate_family = Counter(r["family"] for r in candidate_rows)
    candidate_subgroup = Counter(r["subgroup"] for r in candidate_rows)
    candidate_mode = Counter(r["selection_mode"] for r in candidate_rows)
    candidate_adv = Counter(r["advantage_side"] for r in candidate_rows)
    candidate_pool = Counter(r["pool"] for r in candidate_rows)

    flattened_rows = []
    for row in candidate_rows:
        out = dict(row)
        out["pysr_id_r2"] = row.get("pysr", {}).get("id_test", {}).get("r2")
        out["pysr_id_nmse"] = row.get("pysr", {}).get("id_test", {}).get("nmse")
        out["pysr_ood_r2"] = row.get("pysr", {}).get("ood_test", {}).get("r2")
        out["pysr_ood_nmse"] = row.get("pysr", {}).get("ood_test", {}).get("nmse")
        out["llmsr_id_r2"] = row.get("llmsr", {}).get("id_test", {}).get("r2")
        out["llmsr_id_nmse"] = row.get("llmsr", {}).get("id_test", {}).get("nmse")
        out["llmsr_ood_r2"] = row.get("llmsr", {}).get("ood_test", {}).get("r2")
        out["llmsr_ood_nmse"] = row.get("llmsr", {}).get("ood_test", {}).get("nmse")
        out.pop("pysr", None)
        out.pop("llmsr", None)
        flattened_rows.append(out)

    family_win_rows = []
    for family, counter in sorted(family_wins.items()):
        family_win_rows.append(
            {
                "family": family,
                "pysr_wins": counter.get("pysr", 0),
                "llmsr_wins": counter.get("llmsr", 0),
                "total_comparable": counter.get("pysr", 0) + counter.get("llmsr", 0),
            }
        )

    return {
        "task_rows": len(task_rows),
        "dataset_rows": len(compare_rows),
        "task_status": task_status,
        "comparable_rows": len(comparable_rows),
        "win_counter": win_counter,
        "candidate_rows": candidate_rows,
        "candidate_flat_rows": flattened_rows,
        "candidate_family": candidate_family,
        "candidate_subgroup": candidate_subgroup,
        "candidate_mode": candidate_mode,
        "candidate_adv": candidate_adv,
        "candidate_pool": candidate_pool,
        "family_win_rows": family_win_rows,
    }


def _formal_stage_payload() -> dict[str, Any]:
    task_rows = _load_csv(RESULT_ROOT / "three_seed_formal_task_results.csv")
    method_rows = _load_csv(RESULT_ROOT / "three_seed_formal_dataset_method_summary.csv")
    compare_rows = _load_csv(RESULT_ROOT / "three_seed_formal_dataset_compare.csv")
    master_rows = _load_csv(DEV_CORE_ROOT / "master100_candidates.csv")
    dev_rows = _load_csv(DEV_CORE_ROOT / "benchmark_dev50.csv")
    core_rows = _load_csv(DEV_CORE_ROOT / "benchmark_core50.csv")

    task_status = Counter((r["method"], r["seed"], r["task_status"]) for r in task_rows)
    method_breakdown = {}
    for method in ["pysr", "llmsr"]:
        sub = [r for r in method_rows if r["method"] == method]
        method_breakdown[method] = {
            "datasets": len(sub),
            "full_id_ood": sum(
                1
                for r in sub
                if r["id_nmse_mean"] not in ("", "None", None)
                and r["ood_nmse_mean"] not in ("", "None", None)
            ),
            "id_nmse_median": _median(sub, "id_nmse_mean"),
            "ood_nmse_median": _median(sub, "ood_nmse_mean"),
            "id_r2_median": _median(sub, "id_r2_mean"),
            "ood_r2_median": _median(sub, "ood_r2_mean"),
        }

    comparable_rows = []
    win_counter = Counter()
    family_wins: dict[str, Counter[str]] = defaultdict(Counter)
    for row in compare_rows:
        keys = ["pysr_id_nmse_mean", "pysr_ood_nmse_mean", "llmsr_id_nmse_mean", "llmsr_ood_nmse_mean"]
        if any(row.get(k) in ("", None, "None") for k in keys):
            continue
        comparable_rows.append(row)
        py = (float(row["pysr_id_nmse_mean"]) + float(row["pysr_ood_nmse_mean"])) / 2.0
        ll = (float(row["llmsr_id_nmse_mean"]) + float(row["llmsr_ood_nmse_mean"])) / 2.0
        side = "pysr" if py < ll else "llmsr"
        win_counter[side] += 1
        family_wins[row["family"]][side] += 1

    family_win_rows = []
    for family, counter in sorted(family_wins.items()):
        family_win_rows.append(
            {
                "family": family,
                "pysr_wins": counter.get("pysr", 0),
                "llmsr_wins": counter.get("llmsr", 0),
                "total_comparable": counter.get("pysr", 0) + counter.get("llmsr", 0),
            }
        )

    return {
        "task_rows": len(task_rows),
        "task_status": task_status,
        "compare_rows": len(compare_rows),
        "comparable_rows": len(comparable_rows),
        "win_counter": win_counter,
        "family_win_rows": family_win_rows,
        "method_breakdown": method_breakdown,
        "master_rows": master_rows,
        "dev_rows": dev_rows,
        "core_rows": core_rows,
    }


def _write_overview_md(out_dir: Path, probe: dict[str, Any], formal: dict[str, Any]) -> None:
    text = f"""# Benchmark 选题归纳总览

这个 dossier 专门回答三个问题：

1. `664` 个全量候选是如何缩成 `200` 个的；
2. `200` 个候选又如何基于正式实验缩成 `Master-100`；
3. 这些缩减过程中到底用到了哪些评价指标、结构约束和数据依据。

## 关键阶段

### 阶段 A：`664 -> 200`

- 数据来源：单种子双探针结果
- 任务规模：`{probe['task_rows']}` 个任务
- 数据集规模：`{probe['dataset_rows']}` 个数据集
- 目标：从全量中挑出更有区分度、覆盖更广、信息量更高的候选池

### 阶段 B：`200 -> 100`

- 数据来源：三 seed 正式结果
- 任务规模：`{formal['task_rows']}` 个任务
- 说明：
  - 最初正式阶段先跑了 `800` 个任务（`200 datasets × 2 seeds × 2 methods`）
  - 后续补齐 `seed=522` 后，形成真正用于 `200 -> 100` 的 `1200` 任务正式结果
- 目标：从候选池里筛出更稳定、更可信、对方法差异更有信息量的 `Master-100`

### 阶段 C：`100 -> 50 + 50`

- 数据来源：`Master-100` 的非结果属性
- 目标：把 `Master-100` 切成同分布的 `Dev-50 / Core-50`
- 注意：这一步不再使用正式结果做切分决策，以减少测试集污染

## 本文件夹内容

- `README.md`：总说明与文件导航
- `01_stage_664_to_200.md`：`664 -> 200` 的指标、约束与数据支撑
- `02_stage_200_to_100.md`：`200 -> 100` 的指标、约束与正式实验支撑
- `03_metrics_catalog.md`：整个筛选流程里用到的关键评价指标定义
- `tables/`：关键中间表，便于追溯
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _write_stage1_md(out_dir: Path, probe: dict[str, Any]) -> None:
    top_subgroups = ", ".join(
        f"`{k}={v}`" for k, v in probe["candidate_subgroup"].most_common(10)
    )
    text = f"""# 01. `664 -> 200`：单种子双探针如何压缩成候选池

## 数据来源

- 任务级总表：`one_seed_probe_task_results.csv`
- 数据集级对照：`one_seed_probe_dataset_compare.csv`
- 候选池最终结果：`/tmp/candidate_seeds_200_v3.json`

## 这一阶段实际跑了什么

- 总任务数：`{probe['task_rows']}`
- 总数据集数：`{probe['dataset_rows']}`
- `llmsr` 状态：
  - `ok = {probe['task_status'][('llmsr', 'ok')]}`
  - `timed_out = {probe['task_status'][('llmsr', 'timed_out')]}`
- `pysr` 状态：
  - `timed_out = {probe['task_status'][('pysr', 'timed_out')]}`

## 这一阶段真正用到的评价指标

### 1. `R²`

- 分别在 `id_test` 和 `ood_test` 上统计；
- 用来判断一个方法在某个数据集上是不是“至少还能做”。

### 2. `NMSE`

- 分别在 `id_test` 和 `ood_test` 上统计；
- 这是候选筛选里更核心的误差指标，因为它是归一化误差，跨数据集可比性更强。

### 3. `overall_gap_score`

- 定义：基于 `id/ood` 两个 split 上 `log10(NMSE)` 差异的平均绝对值；
- 作用：衡量 `pysr` 和 `llmsr` 在这个数据集上的**区分度**。

### 4. `signed_advantage / advantage_side`

- 用 `log10(NMSE)` 差异的方向判断哪一边占优；
- 作用：保证候选池不至于全部偏向同一方法。

### 5. `one-sided evaluability`

- 如果只有一边有完整 `id+ood` 指标，另一边没有；
- 这类样本不会被当作噪声直接丢掉，而是单独进入 `pool B`。

## 数据上到底看到了什么

- 能直接比较两种方法 `id+ood NMSE` 的数据集数：`{probe['comparable_rows']}`
- 在这批可比较数据集上：
  - `pysr` 胜：`{probe['win_counter']['pysr']}`
  - `llmsr` 胜：`{probe['win_counter']['llmsr']}`

family 级胜负统计见：

- `tables/stage1_probe_family_wins.csv`

## 为什么不是直接取 top-k

如果只按 `overall_gap_score` 排序，候选池会明显失衡，所以这一阶段还叠加了结构约束：

- `subgroup <= 25`
- `basename <= 2`
- `srsd <= 70`
- 方法优势配比控制到 `pysr : llmsr = 120 : 80`

## 最终得到的 200 候选池

- 总数：`200`
- pool 构成：{_fmt_counter(probe['candidate_pool'])}
- family 分布：{_fmt_counter(probe['candidate_family'])}
- `selection_mode` 分布：{_fmt_counter(probe['candidate_mode'])}
- 候选优势配比：{_fmt_counter(probe['candidate_adv'])}

top subgroup 分布（前 10）：

- {top_subgroups}

## 这一步的结论

`664 -> 200` 不是简单按分数裁剪，而是：

1. 先用单种子双探针识别方法差异；
2. 再用 `gap / advantage / one-sided evaluability` 保住高信息量样本；
3. 最后用 `family / subgroup / basename` 约束，把候选池控制成一个仍有多样性的 `200`。
"""
    (out_dir / "01_stage_664_to_200.md").write_text(text, encoding="utf-8")


def _write_stage2_md(out_dir: Path, formal: dict[str, Any]) -> None:
    master_family = Counter(r["family"] for r in formal["master_rows"])
    master_mode = Counter(r["selection_mode"] for r in formal["master_rows"])
    master_candidate_adv = Counter(r["candidate_advantage_side"] for r in formal["master_rows"])
    master_final_adv = Counter(r["final_advantage_side"] for r in formal["master_rows"])

    text = f"""# 02. `200 -> 100`：正式三 seed 结果如何支撑 `Master-100`

## 数据来源

- 任务级总表：`three_seed_formal_task_results.csv`
- `dataset × method` 聚合表：`three_seed_formal_dataset_method_summary.csv`
- `dataset` 对照表：`three_seed_formal_dataset_compare.csv`
- `Master-100`：`dev_core_split_v1/master100_candidates.csv`

## 这一阶段实际跑了什么

- 最初先完成 `800` 个正式任务：
  - `200 datasets × 2 seeds × 2 methods`
- 后续补齐 `seed=522` 后，真正用于 `200 -> 100` 的正式结果规模是：
  - `200 datasets × 3 seeds × 2 methods = {formal['task_rows']} tasks`

## 任务级状态

### `llmsr`

- `seed=520`: `ok = {formal['task_status'][('llmsr', '520', 'ok')]}`, `timed_out = {formal['task_status'][('llmsr', '520', 'timed_out')]}`
- `seed=521`: `ok = {formal['task_status'][('llmsr', '521', 'ok')]}`, `timed_out = {formal['task_status'][('llmsr', '521', 'timed_out')]}`
- `seed=522`: `ok = {formal['task_status'][('llmsr', '522', 'ok')]}`, `timed_out = {formal['task_status'][('llmsr', '522', 'timed_out')]}`

### `pysr`

- `seed=520`: `timed_out = {formal['task_status'][('pysr', '520', 'timed_out')]}`
- `seed=521`: `timed_out = {formal['task_status'][('pysr', '521', 'timed_out')]}`
- `seed=522`: `timed_out = {formal['task_status'][('pysr', '522', 'timed_out')]}`

这里 `pysr` 的 `timed_out` 不能直接理解成失败，因为它后续仍然能恢复出大量可用结果。

## 这一阶段真正用到的评价指标

### 1. `three_seed_gap`

- 用三 seed 聚合后的 `id/ood NMSE` 差异来衡量方法区分度；
- 它比单种子 probe 的 `gap` 更可信，因为已经过了 seed 扰动。

### 2. `quality_score`

- 看至少一边在 `OOD` 上是否“还能做”；
- 这里主要用：
  - `best_ood_r2`
  - `best_ood_nmse`

### 3. `stability_score`

- 看两个方法各自在三 seed 上有多少完整 `id+ood` 结果；
- 它反映这个数据集是不是“稳定可评估”。

### 4. `priority_score`

- 这是 `Master-100` 排序的核心综合分；
- 具体由三部分加权构成：
  - `55%`：区分度（`three_seed_gap`）
  - `25%`：任务质量（`quality_score`）
  - `20%`：跨 seed 稳定性（`stability_score`）

## 数据上到底看到了什么

### 方法级中位数表现

#### `pysr`

- `full_id_ood = {formal['method_breakdown']['pysr']['full_id_ood']} / 200`
- `id_nmse_mean` 中位数：`{_fmt_float(formal['method_breakdown']['pysr']['id_nmse_median'])}`
- `ood_nmse_mean` 中位数：`{_fmt_float(formal['method_breakdown']['pysr']['ood_nmse_median'])}`
- `id_r2_mean` 中位数：`{_fmt_float(formal['method_breakdown']['pysr']['id_r2_median'])}`
- `ood_r2_mean` 中位数：`{_fmt_float(formal['method_breakdown']['pysr']['ood_r2_median'])}`

#### `llmsr`

- `full_id_ood = {formal['method_breakdown']['llmsr']['full_id_ood']} / 200`
- `id_nmse_mean` 中位数：`{_fmt_float(formal['method_breakdown']['llmsr']['id_nmse_median'])}`
- `ood_nmse_mean` 中位数：`{_fmt_float(formal['method_breakdown']['llmsr']['ood_nmse_median'])}`
- `id_r2_mean` 中位数：`{_fmt_float(formal['method_breakdown']['llmsr']['id_r2_median'])}`
- `ood_r2_mean` 中位数：`{_fmt_float(formal['method_breakdown']['llmsr']['ood_r2_median'])}`

### 可直接比较的胜负

- 具有完整 `pysr + llmsr` 三 seed 平均 `id+ood` 指标的数据集数：`{formal['comparable_rows']}`
- 其中：
  - `pysr` 胜：`{formal['win_counter']['pysr']}`
  - `llmsr` 胜：`{formal['win_counter']['llmsr']}`

family 级胜负统计见：

- `tables/stage2_formal_family_wins.csv`

## 为什么最终是这 100 个

这一阶段不是直接按一个全局分数 top-100，而是：

1. 用 `priority_score` 先排序；
2. 再施加 `family` 配额；
3. 再施加 `basename <= 1`；
4. 再用 `subgroup` 软上限防止某个细分族刷满；
5. 额外保留一定比例的 `llmsr` 优势样本，避免 `Master-100` 完全滑向 `pysr` 主场。

## 最终得到的 `Master-100`

- family 分布：{_fmt_counter(master_family)}
- `selection_mode` 分布：{_fmt_counter(master_mode)}
- 候选阶段优势标签：{_fmt_counter(master_candidate_adv)}
- 正式三 seed 胜负标签：{_fmt_counter(master_final_adv)}

## 这一步的边界

正式结果在这里的作用是：

- **决定哪些题进入 `Master-100`**

正式结果不再用于：

- 决定某个题进 `Dev-50` 还是 `Core-50`

因为后者会直接污染最终测试集。
"""
    (out_dir / "02_stage_200_to_100.md").write_text(text, encoding="utf-8")


def _write_metrics_catalog_md(out_dir: Path) -> None:
    text = """# 03. 评价指标与决策依据目录

下面这张表只列**真正参与筛选决策**的指标，不列所有底层实验日志字段。

| 指标 | 使用阶段 | 含义 | 在决策中的作用 |
|---|---|---|---|
| `id_test R²` | `664 -> 200` | 模型在 ID split 上的拟合质量 | 判断某个方法在该数据集上是否“至少还能做” |
| `ood_test R²` | `664 -> 200`, `200 -> 100` | 模型在 OOD split 上的泛化质量 | 判断是否属于真正有泛化信息量的样本 |
| `id_test NMSE` | `664 -> 200`, `200 -> 100` | ID split 上的归一化误差 | 跨数据集可比的误差主指标之一 |
| `ood_test NMSE` | `664 -> 200`, `200 -> 100` | OOD split 上的归一化误差 | 选题时最重要的泛化误差指标 |
| `overall_gap_score` | `664 -> 200` | 单种子 probe 中，`pysr/llmsr` 在 `id+ood log10(NMSE)` 上的差异强度 | 用来识别高区分度候选 |
| `advantage_side` | `664 -> 200` | 单种子 probe 中哪一方法占优 | 用来平衡候选池，避免单边 benchmark |
| `one-sided evaluability` | `664 -> 200` | 只有一边有完整 `id+ood` 指标 | 保留方法可评估性差异，不把这类题直接当噪声删掉 |
| `three_seed_gap` | `200 -> 100` | 三 seed 聚合后两方法在 `id+ood NMSE` 上的差异强度 | 比单种子 probe 更可信的区分度指标 |
| `quality_score` | `200 -> 100` | 至少一边在 OOD 上是否仍然“能做” | 防止 `Master-100` 被纯噪声或双边崩溃样本主导 |
| `stability_score` | `200 -> 100` | 三个 seed 中有多少条完整可评估结果 | 保证 `Master-100` 不只是高 gap，还要稳定 |
| `priority_score` | `200 -> 100` | 综合 `gap + quality + stability` 的总分 | `Master-100` 的主排序依据 |
| `family / subgroup` | 两阶段都有 | 数据集来源与细分题型 | 保证题型覆盖，不让大族刷满 |
| `basename` | 两阶段都有 | 题目的基础身份 | 控制同 basename 变体泄漏 |

## 额外说明

### 为什么主要用 `NMSE` 而不是 `MSE`

- `MSE` 会受目标量纲和数值范围影响；
- `NMSE` 经过归一化后更适合做跨数据集比较；
- 因此在候选筛选和 `Master-100` 选择中，`NMSE` 才是主误差指标。

### 为什么 `200 -> 100` 要看三 seed

- 单种子 probe 更像前置筛查；
- 真正决定哪些题值得进入最终 benchmark 宇宙，必须看三 seed 下是否仍然稳定。

### 为什么 `Dev/Core` 切分时不用这些结果指标

- 因为这一步的目标不是继续“选题”，而是“切测试集”；
- 如果切分还继续看正式结果，就会把测试集做成结果驱动的定制集合。
"""
    (out_dir / "03_metrics_catalog.md").write_text(text, encoding="utf-8")


def build() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table_dir = OUTPUT_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    probe = _probe_stage_payload()
    formal = _formal_stage_payload()

    _write_overview_md(OUTPUT_ROOT, probe, formal)
    _write_stage1_md(OUTPUT_ROOT, probe)
    _write_stage2_md(OUTPUT_ROOT, formal)
    _write_metrics_catalog_md(OUTPUT_ROOT)

    # 关键表
    _dump_csv(table_dir / "stage1_candidate200_flat.csv", probe["candidate_flat_rows"])
    _dump_csv(table_dir / "stage1_probe_family_wins.csv", probe["family_win_rows"])
    _dump_csv(table_dir / "stage2_master100_selection_basis.csv", formal["master_rows"])
    _dump_csv(table_dir / "stage2_formal_family_wins.csv", formal["family_win_rows"])

    pipeline_rows = [
        {
            "stage": "664_to_200_probe",
            "input_tasks": probe["task_rows"],
            "input_datasets": probe["dataset_rows"],
            "output_size": 200,
            "notes": "单种子双探针 + 候选池约束",
        },
        {
            "stage": "200_to_100_formal",
            "input_tasks": formal["task_rows"],
            "input_datasets": 200,
            "output_size": 100,
            "notes": "三 seed 正式结果 + family/basename/subgroup 约束",
        },
    ]
    _dump_csv(table_dir / "pipeline_stage_overview.csv", pipeline_rows)


if __name__ == "__main__":
    build()
