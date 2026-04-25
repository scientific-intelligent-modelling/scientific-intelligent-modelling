#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_E1 = Path("exp-planning/02.E1选择验证/e1_final_results_current_20260426/digest/e1_result_table.csv")
DEFAULT_SPLIT_DIR = Path("experiment-results/benchmark_formal200_20260417/clean_core_reserve_split_v1")
DEFAULT_OUTPUT = Path("experiment-results/benchmark_formal200_20260417/e1_rank_fidelity_7alg_v1")
METHODS = ["drsr", "dso", "gplearn", "llmsr", "pyoperon", "pysr", "tpsr"]
INVALID_SCORE = 12.0
LOG_NMSE_FLOOR = 1e-12
LOG_NMSE_CLIP = (-12.0, 12.0)


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


def _bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def _normalize_dataset_dir(path: Any) -> str:
    text = "" if path is None else str(path).replace("\\", "/").strip()
    marker = "sim-datasets-data/"
    if marker in text:
        return marker + text.split(marker, 1)[1].strip("/")
    return text.rstrip("/")


def _clipped_log_nmse(value: Any) -> float | None:
    num = _float(value)
    if num is None or num < 0:
        return None
    out = math.log10(max(num, LOG_NMSE_FLOOR))
    return min(LOG_NMSE_CLIP[1], max(LOG_NMSE_CLIP[0], out))


def _row_score(row: dict[str, str]) -> tuple[float, str]:
    if not _bool(row.get("valid_output")):
        return INVALID_SCORE, "invalid_output"
    id_score = _clipped_log_nmse(row.get("id_nmse"))
    ood_score = _clipped_log_nmse(row.get("ood_nmse"))
    if id_score is None or ood_score is None:
        return INVALID_SCORE, "missing_metric"
    return 0.5 * id_score + 0.5 * ood_score, "valid"


def _dataset_dirs(rows: list[dict[str, str]]) -> set[str]:
    return {_normalize_dataset_dir(r["dataset_dir"]) for r in rows}


def _result_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        method = row.get("method", "")
        dataset_dir = _normalize_dataset_dir(row.get("dataset_dir"))
        if method:
            index[(method, dataset_dir)] = row
    return index


def _rankdata(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(master_scores: dict[str, float], subset_scores: dict[str, float]) -> float:
    keys = sorted(set(master_scores) & set(subset_scores))
    mr = _rankdata({k: master_scores[k] for k in keys})
    sr = _rankdata({k: subset_scores[k] for k in keys})
    return _pearson([mr[k] for k in keys], [sr[k] for k in keys])


def _kendall(master_scores: dict[str, float], subset_scores: dict[str, float]) -> float:
    keys = sorted(set(master_scores) & set(subset_scores))
    concordant = 0
    discordant = 0
    for a, b in combinations(keys, 2):
        m = (master_scores[a] > master_scores[b]) - (master_scores[a] < master_scores[b])
        s = (subset_scores[a] > subset_scores[b]) - (subset_scores[a] < subset_scores[b])
        if m == 0 or s == 0:
            continue
        if m == s:
            concordant += 1
        else:
            discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else float("nan")


def _pairwise_agreement(master_scores: dict[str, float], subset_scores: dict[str, float]) -> float:
    keys = sorted(set(master_scores) & set(subset_scores))
    agree = 0
    total = 0
    for a, b in combinations(keys, 2):
        m = (master_scores[a] > master_scores[b]) - (master_scores[a] < master_scores[b])
        s = (subset_scores[a] > subset_scores[b]) - (subset_scores[a] < subset_scores[b])
        if m == 0:
            continue
        total += 1
        if m == s:
            agree += 1
    return agree / total if total else float("nan")


def _method_scores(
    result_index: dict[tuple[str, str], dict[str, str]],
    dataset_dirs: set[str],
    methods: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores: dict[str, float] = {}
    detail_rows: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        reasons = Counter()
        for dataset_dir in dataset_dirs:
            row = result_index.get((method, dataset_dir))
            if row is None:
                values.append(INVALID_SCORE)
                reasons["missing_run"] += 1
                continue
            score, reason = _row_score(row)
            values.append(score)
            reasons[reason] += 1
        score_mean = statistics.mean(values) if values else INVALID_SCORE
        scores[method] = score_mean
        detail_rows.append(
            {
                "method": method,
                "score": score_mean,
                "dataset_count": len(dataset_dirs),
                "valid_count": reasons["valid"],
                "invalid_or_missing_count": len(dataset_dirs) - reasons["valid"],
                "missing_run_count": reasons["missing_run"],
                "invalid_output_count": reasons["invalid_output"],
                "missing_metric_count": reasons["missing_metric"],
            }
        )
    return scores, detail_rows


def _subset_metrics(
    label: str,
    result_index: dict[tuple[str, str], dict[str, str]],
    master_dirs: set[str],
    subset_dirs: set[str],
    methods: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    master_scores, master_detail = _method_scores(result_index, master_dirs, methods)
    subset_scores, subset_detail = _method_scores(result_index, subset_dirs, methods)
    keys = sorted(master_scores)
    aggregate_abs_error = statistics.mean(abs(master_scores[k] - subset_scores[k]) for k in keys)
    aggregate_rmse = math.sqrt(statistics.mean((master_scores[k] - subset_scores[k]) ** 2 for k in keys))
    row = {
        "subset": label,
        "subset_size": len(subset_dirs),
        "master_size": len(master_dirs),
        "spearman": _spearman(master_scores, subset_scores),
        "kendall": _kendall(master_scores, subset_scores),
        "pairwise_win_agreement": _pairwise_agreement(master_scores, subset_scores),
        "aggregate_score_mae": aggregate_abs_error,
        "aggregate_score_rmse": aggregate_rmse,
    }
    score_rows: list[dict[str, Any]] = []
    detail_by_method = {r["method"]: r for r in subset_detail}
    master_detail_by_method = {r["method"]: r for r in master_detail}
    for method in methods:
        score_rows.append(
            {
                "subset": label,
                "method": method,
                "master_score": master_scores[method],
                "subset_score": subset_scores[method],
                "score_delta": subset_scores[method] - master_scores[method],
                "master_valid_count": master_detail_by_method[method]["valid_count"],
                "subset_valid_count": detail_by_method[method]["valid_count"],
                "subset_invalid_or_missing_count": detail_by_method[method]["invalid_or_missing_count"],
            }
        )
    return row, score_rows


def _sample_random(rows: list[dict[str, str]], rng: random.Random, n: int) -> set[str]:
    return {_normalize_dataset_dir(r["dataset_dir"]) for r in rng.sample(rows, n)}


def _sample_family_stratified(
    rows: list[dict[str, str]],
    core_family_counts: Counter,
    rng: random.Random,
) -> set[str]:
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    selected: set[str] = set()
    for family, count in sorted(core_family_counts.items()):
        selected.update(_normalize_dataset_dir(r["dataset_dir"]) for r in rng.sample(by_family[family], count))
    return selected


def _baseline_summary(core_metric: dict[str, Any], baseline_rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    out = {"baseline": label, "replicates": len(baseline_rows)}
    for metric, higher_is_better in [
        ("spearman", True),
        ("kendall", True),
        ("pairwise_win_agreement", True),
        ("aggregate_score_mae", False),
        ("aggregate_score_rmse", False),
    ]:
        vals = [float(r[metric]) for r in baseline_rows if r.get(metric) not in (None, "")]
        core_val = float(core_metric[metric])
        if not vals:
            out[f"{metric}_mean"] = None
            out[f"{metric}_p05"] = None
            out[f"{metric}_p50"] = None
            out[f"{metric}_p95"] = None
            out[f"{metric}_core_at_least_as_good_fraction"] = None
            out[f"{metric}_core_strictly_better_fraction"] = None
            continue
        vals_sorted = sorted(vals)
        def q(p: float) -> float:
            idx = min(len(vals_sorted) - 1, max(0, round(p * (len(vals_sorted) - 1))))
            return vals_sorted[idx]
        out[f"{metric}_mean"] = statistics.mean(vals)
        out[f"{metric}_p05"] = q(0.05)
        out[f"{metric}_p50"] = q(0.50)
        out[f"{metric}_p95"] = q(0.95)
        if higher_is_better:
            out[f"{metric}_core_at_least_as_good_fraction"] = sum(core_val >= v for v in vals) / len(vals)
            out[f"{metric}_core_strictly_better_fraction"] = sum(core_val > v for v in vals) / len(vals)
        else:
            out[f"{metric}_core_at_least_as_good_fraction"] = sum(core_val <= v for v in vals) / len(vals)
            out[f"{metric}_core_strictly_better_fraction"] = sum(core_val < v for v in vals) / len(vals)
    return out


def _write_report(
    output_dir: Path,
    subset_rows: list[dict[str, Any]],
    baseline_summary_rows: list[dict[str, Any]],
    method_score_rows: list[dict[str, Any]],
    nonvalid_master_rows: list[dict[str, Any]],
) -> None:
    core = next(r for r in subset_rows if r["subset"] == "core50")
    reserve = next(r for r in subset_rows if r["subset"] == "reserve50")
    lines = [
        "# E1 7算法 Rank-Fidelity Pilot",
        "",
        "## 口径",
        "",
        "- 输入结果：E1 `7 algorithms × 200 candidate × 1 seed` 当前 digest。",
        "- 评估集合：`Clean-Master-100`、由 clean master 重新切出的 `Core-50 / Reserve-50`。",
        "- 单 run 分数：`0.5 * log10(id_nmse) + 0.5 * log10(ood_nmse)`，范围裁剪到 `[-12, 12]`。",
        f"- 缺失、非法或非完整指标按惩罚分 `{INVALID_SCORE}` 处理；分数越低越好。",
        "- 切分本身不使用 E1 结果；E1 只用于事后验证 rank fidelity。",
        "",
        "## Core/Reserve 对 Master-100 的保真度",
        "",
        "| subset | Spearman | Kendall | pairwise win agreement | score MAE | score RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in [core, reserve]:
        lines.append(
            f"| `{row['subset']}` | {row['spearman']:.4f} | {row['kendall']:.4f} | "
            f"{row['pairwise_win_agreement']:.4f} | {row['aggregate_score_mae']:.4f} | {row['aggregate_score_rmse']:.4f} |"
        )
    lines.extend([
        "",
        "## Core-50 相对随机 50 的位置",
        "",
        "`at least as good` 包含并列；`strictly better` 不包含并列。排名相关指标中随机 50 经常也达到 1.0，因此这里重点看 aggregate score error。",
        "",
        "| baseline | metric | Core at least as good | Core strictly better | baseline mean | p05 | p50 | p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in baseline_summary_rows:
        for metric in ["spearman", "kendall", "pairwise_win_agreement", "aggregate_score_mae", "aggregate_score_rmse"]:
            lines.append(
                f"| `{row['baseline']}` | `{metric}` | {row[f'{metric}_core_at_least_as_good_fraction']:.4f} | "
                f"{row[f'{metric}_core_strictly_better_fraction']:.4f} | "
                f"{row[f'{metric}_mean']:.4f} | {row[f'{metric}_p05']:.4f} | {row[f'{metric}_p50']:.4f} | {row[f'{metric}_p95']:.4f} |"
            )
    lines.extend(["", "## 方法分数", "", "| method | master score | core score | reserve score | master valid | core valid | reserve valid |", "|---|---:|---:|---:|---:|---:|---:|"])
    by = {(r["subset"], r["method"]): r for r in method_score_rows}
    for method in METHODS:
        m = by[("master100", method)]
        c = by[("core50", method)]
        r = by[("reserve50", method)]
        lines.append(
            f"| `{method}` | {m['subset_score']:.4f} | {c['subset_score']:.4f} | {r['subset_score']:.4f} | "
            f"{m['subset_valid_count']} | {c['subset_valid_count']} | {r['subset_valid_count']} |"
        )
    lines.extend(["", "## Clean-Master-100 中的非完整 run", ""])
    if nonvalid_master_rows:
        for row in nonvalid_master_rows:
            lines.append(f"- `{row['method']}` / `{row['dataset_id']}` / `{row['dataset_name']}` / `{row['timeout_type']}`")
    else:
        lines.append("- 无。")
    (output_dir / "rank_fidelity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="用 E1 7算法结果验证 Core-50 对 Clean-Master-100 的 rank fidelity。")
    parser.add_argument("--e1-result-table", default=str(DEFAULT_E1))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--baseline-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260426)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    master_rows = _load_rows(split_dir / "clean_master100.csv")
    core_rows = _load_rows(split_dir / "benchmark_core50.csv")
    reserve_rows = _load_rows(split_dir / "benchmark_reserve50.csv")
    e1_rows = _load_rows(Path(args.e1_result_table))
    result_index = _result_index(e1_rows)
    methods = [m for m in METHODS if any(row.get("method") == m for row in e1_rows)]

    master_dirs = _dataset_dirs(master_rows)
    core_dirs = _dataset_dirs(core_rows)
    reserve_dirs = _dataset_dirs(reserve_rows)
    if len(master_dirs) != 100 or len(core_dirs) != 50 or len(reserve_dirs) != 50:
        raise ValueError("Master/Core/Reserve 集合大小不符合预期")
    if core_dirs & reserve_dirs:
        raise ValueError("Core 与 Reserve 存在重叠数据集")
    if (core_dirs | reserve_dirs) != master_dirs:
        raise ValueError("Core ∪ Reserve 不等于 Clean-Master-100")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_rows: list[dict[str, Any]] = []
    method_score_rows: list[dict[str, Any]] = []
    for label, dirs in [("master100", master_dirs), ("core50", core_dirs), ("reserve50", reserve_dirs)]:
        row, scores = _subset_metrics(label, result_index, master_dirs, dirs, methods)
        if label == "master100":
            row.update({"spearman": 1.0, "kendall": 1.0, "pairwise_win_agreement": 1.0, "aggregate_score_mae": 0.0, "aggregate_score_rmse": 0.0})
        subset_rows.append(row)
        method_score_rows.extend(scores)

    core_metric = next(r for r in subset_rows if r["subset"] == "core50")
    rng = random.Random(args.seed)
    random_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    core_family_counts = Counter(r["family"] for r in core_rows)
    for i in range(args.baseline_repeats):
        dirs = _sample_random(master_rows, rng, 50)
        row, _ = _subset_metrics(f"random50_{i:04d}", result_index, master_dirs, dirs, methods)
        random_rows.append(row)
        dirs = _sample_family_stratified(master_rows, core_family_counts, rng)
        row, _ = _subset_metrics(f"family_stratified_random50_{i:04d}", result_index, master_dirs, dirs, methods)
        stratified_rows.append(row)

    baseline_summary_rows = [
        _baseline_summary(core_metric, random_rows, "random50"),
        _baseline_summary(core_metric, stratified_rows, "family_stratified_random50"),
    ]

    master_nonvalid: list[dict[str, Any]] = []
    for row in e1_rows:
        if _normalize_dataset_dir(row.get("dataset_dir")) in master_dirs and row.get("method") in methods and not _bool(row.get("valid_output")):
            master_nonvalid.append(
                {
                    "method": row.get("method"),
                    "dataset_id": row.get("dataset_id"),
                    "dataset_name": row.get("dataset_name"),
                    "timeout_type": row.get("timeout_type"),
                    "id_nmse": row.get("id_nmse"),
                    "ood_nmse": row.get("ood_nmse"),
                    "expression": row.get("expression"),
                }
            )

    _write_rows(output_dir / "subset_rank_fidelity_metrics.csv", subset_rows)
    _write_rows(output_dir / "method_scores.csv", method_score_rows)
    _write_rows(output_dir / "random50_baseline_metrics.csv", random_rows)
    _write_rows(output_dir / "family_stratified_random50_baseline_metrics.csv", stratified_rows)
    _write_rows(output_dir / "baseline_summary.csv", baseline_summary_rows)
    _write_rows(output_dir / "master100_nonvalid_runs.csv", master_nonvalid)
    metadata = {
        "e1_result_table": str(Path(args.e1_result_table)),
        "split_dir": str(split_dir),
        "methods": methods,
        "baseline_repeats": args.baseline_repeats,
        "seed": args.seed,
        "invalid_score": INVALID_SCORE,
        "score_definition": "0.5*clipped_log10(id_nmse)+0.5*clipped_log10(ood_nmse); lower is better",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output_dir, subset_rows, baseline_summary_rows, method_score_rows, master_nonvalid)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "core_spearman": core_metric["spearman"],
                "core_pairwise_win_agreement": core_metric["pairwise_win_agreement"],
                "core_aggregate_score_mae": core_metric["aggregate_score_mae"],
                "master_nonvalid_runs": len(master_nonvalid),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
