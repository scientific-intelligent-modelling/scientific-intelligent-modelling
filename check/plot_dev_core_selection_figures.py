#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _count(rows: list[dict[str, str]], key: str) -> Counter:
    return Counter(r[key] for r in rows)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_family_distribution(
    all200: list[dict[str, str]],
    master: list[dict[str, str]],
    dev: list[dict[str, str]],
    core: list[dict[str, str]],
    output: Path,
) -> None:
    families = sorted(set(_count(all200, "family")) | set(_count(master, "family")) | set(_count(dev, "family")) | set(_count(core, "family")))
    sources = {
        "Candidate-200": _count(all200, "family"),
        "Master-100": _count(master, "family"),
        "Dev-50": _count(dev, "family"),
        "Core-50": _count(core, "family"),
    }
    df = pd.DataFrame(
        [
            {"family": fam, "split": split, "count": counts.get(fam, 0)}
            for fam in families
            for split, counts in sources.items()
        ]
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="family", y="count", hue="split")
    plt.title("Family distribution from Candidate-200 to Master/Dev/Core")
    plt.xlabel("family")
    plt.ylabel("count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_mode_adv_distribution(master: list[dict[str, str]], dev: list[dict[str, str]], core: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, title in [
        (axes[0], "selection_mode", "Selection mode distribution"),
        (axes[1], "candidate_advantage_side", "Candidate-side advantage distribution"),
    ]:
        values = sorted(set(r[key] for r in master + dev + core))
        rows = []
        for label, sample in [("Master-100", master), ("Dev-50", dev), ("Core-50", core)]:
            counter = Counter(r[key] for r in sample)
            for value in values:
                rows.append({"group": label, "category": value, "count": counter.get(value, 0)})
        df = pd.DataFrame(rows)
        sns.barplot(data=df, x="category", y="count", hue="group", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_method_metric_distributions(summary_rows: list[dict[str, str]], output: Path) -> None:
    records = []
    for row in summary_rows:
        method = row["method"]
        for metric in ("id_nmse_mean", "ood_nmse_mean", "id_r2_mean", "ood_r2_mean"):
            v = row.get(metric)
            if v in ("", None, "None"):
                continue
            try:
                value = float(v)
            except Exception:
                continue
            records.append({"method": method, "metric": metric, "value": value})

    df = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metric_order = ["id_nmse_mean", "ood_nmse_mean", "id_r2_mean", "ood_r2_mean"]
    title_map = {
        "id_nmse_mean": "ID NMSE",
        "ood_nmse_mean": "OOD NMSE",
        "id_r2_mean": "ID R²",
        "ood_r2_mean": "OOD R²",
    }
    for ax, metric in zip(axes.flatten(), metric_order, strict=True):
        sub = df[df["metric"] == metric].copy()
        if "nmse" in metric:
            sub["value"] = sub["value"].clip(lower=1e-12)
            sns.boxplot(data=sub, x="method", y="value", ax=ax)
            ax.set_yscale("log")
        else:
            sns.boxplot(data=sub, x="method", y="value", ax=ax)
        ax.set_title(title_map[metric])
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_family_win_distribution(compare_rows: list[dict[str, str]], output: Path) -> None:
    family_wins = defaultdict(lambda: Counter())
    for row in compare_rows:
        needed = ["pysr_id_nmse_mean", "pysr_ood_nmse_mean", "llmsr_id_nmse_mean", "llmsr_ood_nmse_mean"]
        if any(row.get(k) in ("", None, "None") for k in needed):
            continue
        py = (float(row["pysr_id_nmse_mean"]) + float(row["pysr_ood_nmse_mean"])) / 2.0
        ll = (float(row["llmsr_id_nmse_mean"]) + float(row["llmsr_ood_nmse_mean"])) / 2.0
        winner = "PySR" if py < ll else "LLM-SR"
        family_wins[row["family"]][winner] += 1

    families = sorted(family_wins)
    pysr_vals = [family_wins[f]["PySR"] for f in families]
    ll_vals = [family_wins[f]["LLM-SR"] for f in families]

    plt.figure(figsize=(11, 5.5))
    x = range(len(families))
    plt.bar(x, pysr_vals, label="PySR wins")
    plt.bar(x, ll_vals, bottom=pysr_vals, label="LLM-SR wins")
    plt.xticks(list(x), families, rotation=25, ha="right")
    plt.ylabel("datasets")
    plt.title("Method wins by family on comparable three-seed datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Dev/Core 选择理由的指标图。")
    parser.add_argument(
        "--result-dir",
        default="experiment-results/benchmark_formal200_20260417",
        help="实验结果目录",
    )
    parser.add_argument(
        "--split-dir",
        default="experiment-results/benchmark_formal200_20260417/dev_core_split_v1",
        help="Dev/Core 切分目录",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/benchmark/figures/dev_core_selection_v1",
        help="图输出目录",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    result_dir = Path(args.result_dir).resolve()
    split_dir = Path(args.split_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(output_dir)

    all200 = _load_csv(result_dir / "three_seed_formal_dataset_compare.csv")
    summary = _load_csv(result_dir / "three_seed_formal_dataset_method_summary.csv")
    master = _load_csv(split_dir / "master100_candidates.csv")
    dev = _load_csv(split_dir / "benchmark_dev50.csv")
    core = _load_csv(split_dir / "benchmark_core50.csv")

    _plot_family_distribution(all200, master, dev, core, output_dir / "family_distribution_200_master_dev_core.png")
    _plot_mode_adv_distribution(master, dev, core, output_dir / "mode_and_advantage_distribution.png")
    _plot_method_metric_distributions(summary, output_dir / "method_metric_distributions.png")
    _plot_family_win_distribution(all200, output_dir / "family_win_distribution.png")

    print("已生成图目录：")
    print(output_dir)


if __name__ == "__main__":
    main()
