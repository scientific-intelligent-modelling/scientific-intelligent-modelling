#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MASTER = Path("experiment-results/benchmark_formal200_20260417/clean_master100_v1/clean_master100.csv")
DEFAULT_OUTPUT = Path("experiment-results/benchmark_formal200_20260417/clean_core_reserve_split_v1")
RESULT_DERIVED_COLUMNS_EXCLUDED_FROM_SPLIT = [
    "priority_score",
    "three_seed_gap",
    "quality_score",
    "stability_score",
    "final_advantage_side",
    "candidate_advantage_side",
    "selection_mode",
]
STATIC_FEATURES = [
    "feature_count",
    "train_samples",
    "valid_samples",
    "id_test_samples",
    "ood_test_samples",
    "formula_line_count",
    "formula_char_count",
    "formula_operator_count",
]


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(v: Any) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        out = float(v)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _contains_dummy(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(k, "")) for k in ("subgroup", "dataset_name", "dataset_dir"))
    return "dummy" in text.lower()


def _static_value(row: dict[str, Any], key: str) -> float:
    raw = _float(row.get(key))
    if raw is None:
        return 0.0
    if key.endswith("_samples") or key in {"formula_char_count"}:
        return math.log10(max(raw, 1.0))
    return raw


def _numeric_scales(rows: list[dict[str, Any]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for key in STATIC_FEATURES:
        vals = [_static_value(r, key) for r in rows]
        if len(vals) <= 1:
            scales[key] = 1.0
            continue
        stdev = statistics.pstdev(vals)
        scales[key] = stdev if stdev > 1e-12 else 1.0
    return scales


def _counter_distance(left: Counter, right: Counter, weight: float = 1.0) -> float:
    keys = set(left) | set(right)
    return weight * sum(abs(left[k] - right[k]) for k in keys)


def _split_loss(core: list[dict[str, Any]], reserve: list[dict[str, Any]], scales: dict[str, float]) -> float:
    loss = 0.0
    loss += _counter_distance(Counter(r["family"] for r in core), Counter(r["family"] for r in reserve), 100.0)
    loss += _counter_distance(Counter(r["subgroup"] for r in core), Counter(r["subgroup"] for r in reserve), 3.0)
    loss += _counter_distance(Counter(_contains_dummy(r) for r in core), Counter(_contains_dummy(r) for r in reserve), 2.0)

    # 下列字段只用于审计，不参与 loss；这里显式不看 PySR/LLMSR/E1 成绩。
    for key in STATIC_FEATURES:
        core_vals = [_static_value(r, key) for r in core]
        reserve_vals = [_static_value(r, key) for r in reserve]
        loss += abs(statistics.mean(core_vals) - statistics.mean(reserve_vals)) / scales[key]
        loss += 0.35 * abs(statistics.median(core_vals) - statistics.median(reserve_vals)) / scales[key]
    return loss


def _sample_split_by_family(rows: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core: list[dict[str, Any]] = []
    reserve: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family, items in sorted(by_family.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) % 2 != 0:
            raise ValueError(f"family `{family}` 数量为奇数，无法精确半分：{len(items)}")
        half = len(items) // 2
        core.extend(items[:half])
        reserve.extend(items[half:])
    return core, reserve


def _best_split(rows: list[dict[str, Any]], iterations: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    rng = random.Random(seed)
    scales = _numeric_scales(rows)
    best_core: list[dict[str, Any]] | None = None
    best_reserve: list[dict[str, Any]] | None = None
    best_loss = float("inf")
    for _ in range(iterations):
        core, reserve = _sample_split_by_family(rows, rng)
        loss = _split_loss(core, reserve, scales)
        if loss < best_loss:
            best_core = core
            best_reserve = reserve
            best_loss = loss
    if best_core is None or best_reserve is None:
        raise RuntimeError("未能生成 Core/Reserve split")
    return best_core, best_reserve, best_loss


def _summary_counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(key, "")) for r in rows).items()))


def _numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    vals = [_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(vals), "mean": statistics.mean(vals), "median": statistics.median(vals)}


def _write_audit(output_dir: Path, master: list[dict[str, Any]], core: list[dict[str, Any]], reserve: list[dict[str, Any]], loss: float, seed: int, iterations: int) -> None:
    audit = {
        "master_size": len(master),
        "core_size": len(core),
        "reserve_size": len(reserve),
        "seed": seed,
        "iterations": iterations,
        "split_loss": loss,
        "result_derived_columns_excluded_from_split": RESULT_DERIVED_COLUMNS_EXCLUDED_FROM_SPLIT,
        "core_family": _summary_counter(core, "family"),
        "reserve_family": _summary_counter(reserve, "family"),
        "core_subgroup": _summary_counter(core, "subgroup"),
        "reserve_subgroup": _summary_counter(reserve, "subgroup"),
        "audit_only_core_selection_mode": _summary_counter(core, "selection_mode"),
        "audit_only_reserve_selection_mode": _summary_counter(reserve, "selection_mode"),
        "audit_only_core_candidate_advantage_side": _summary_counter(core, "candidate_advantage_side"),
        "audit_only_reserve_candidate_advantage_side": _summary_counter(reserve, "candidate_advantage_side"),
        "numeric_static_summary": {
            key: {
                "core": _numeric_summary(core, key),
                "reserve": _numeric_summary(reserve, key),
            }
            for key in STATIC_FEATURES
        },
    }
    (output_dir / "clean_core_reserve_split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Clean-Master-100 / Core-50 / Reserve-50 切分审计",
        "",
        "## 口径",
        "",
        "- 输入：`Clean-Master-100`。",
        "- 输出：`Core-50` 与 `Reserve-50`。",
        "- 切分时只使用静态/非结果属性：`family / subgroup / dummy 标记 / feature_count / sample size / formula complexity`。",
        "- `selection_mode`、`candidate_advantage_side`、`priority_score`、`quality_score` 等结果派生字段只在本报告中审计，不参与切分优化。",
        "",
        "## 总览",
        "",
        f"- Master 数量：`{len(master)}`",
        f"- Core 数量：`{len(core)}`",
        f"- Reserve 数量：`{len(reserve)}`",
        f"- 随机搜索次数：`{iterations}`",
        f"- 随机种子：`{seed}`",
        f"- split loss：`{loss:.6f}`",
        "",
        "## family 分布",
        "",
        "| family | Core-50 | Reserve-50 |",
        "|---|---:|---:|",
    ]
    core_family = Counter(r["family"] for r in core)
    reserve_family = Counter(r["family"] for r in reserve)
    for family in sorted(set(core_family) | set(reserve_family)):
        lines.append(f"| `{family}` | {core_family[family]} | {reserve_family[family]} |")

    lines.extend(["", "## 审计字段分布（未参与切分优化）", "", "### selection_mode", "", "| selection_mode | Core-50 | Reserve-50 |", "|---|---:|---:|"])
    core_mode = Counter(r.get("selection_mode", "") for r in core)
    reserve_mode = Counter(r.get("selection_mode", "") for r in reserve)
    for key in sorted(set(core_mode) | set(reserve_mode)):
        lines.append(f"| `{key}` | {core_mode[key]} | {reserve_mode[key]} |")
    lines.extend(["", "### candidate_advantage_side", "", "| side | Core-50 | Reserve-50 |", "|---|---:|---:|"])
    core_adv = Counter(r.get("candidate_advantage_side", "") for r in core)
    reserve_adv = Counter(r.get("candidate_advantage_side", "") for r in reserve)
    for key in sorted(set(core_adv) | set(reserve_adv)):
        lines.append(f"| `{key}` | {core_adv[key]} | {reserve_adv[key]} |")

    lines.extend(["", "## 静态数值字段", "", "| 字段 | Core mean | Core median | Reserve mean | Reserve median |", "|---|---:|---:|---:|---:|"])
    for key in STATIC_FEATURES:
        cs = _numeric_summary(core, key)
        rs = _numeric_summary(reserve, key)
        lines.append(f"| `{key}` | {cs['mean']} | {cs['median']} | {rs['mean']} | {rs['median']} |")
    (output_dir / "clean_core_reserve_split_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Clean-Master-100 生成不看结果的 Core-50 / Reserve-50 切分。")
    parser.add_argument("--master-csv", default=str(DEFAULT_MASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260426)
    args = parser.parse_args()

    master = _load_rows(Path(args.master_csv))
    if len(master) != 100:
        raise ValueError(f"预期 Clean-Master-100 有 100 条，实际 {len(master)} 条")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core, reserve, loss = _best_split(master, args.iterations, args.seed)
    core = sorted(core, key=lambda r: (r["family"], r["subgroup"], r["dataset_name"], r["dataset_dir"]))
    reserve = sorted(reserve, key=lambda r: (r["family"], r["subgroup"], r["dataset_name"], r["dataset_dir"]))
    master_sorted = sorted(master, key=lambda r: (r["family"], r["subgroup"], r["dataset_name"], r["dataset_dir"]))

    _write_rows(output_dir / "clean_master100.csv", master_sorted)
    _write_rows(output_dir / "benchmark_core50.csv", core)
    _write_rows(output_dir / "benchmark_reserve50.csv", reserve)
    _write_rows(output_dir / "benchmark_dev50.csv", reserve)
    _write_audit(output_dir, master_sorted, core, reserve, loss, args.seed, args.iterations)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "master_size": len(master),
                "core_size": len(core),
                "reserve_size": len(reserve),
                "split_loss": loss,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
