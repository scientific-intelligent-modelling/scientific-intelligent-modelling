#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


MASTER100_FAMILY_TARGET = {
    "srsd": 34,
    "llm-srbench": 30,
    "srbench1.0": 16,
    "nguyen": 6,
    "keijzer": 6,
    "korns": 4,
    "srbench2025": 4,
}

MASTER100_SELECTION_MODE_TARGET = {
    "strict": 52,
    "mid-gap": 26,
    "relaxed": 10,
    "one-sided": 12,
}

MASTER100_ADVANTAGE_TARGET = {
    "pysr": 70,
    "llmsr": 30,
}

MASTER100_SUBGROUP_SOFT_CAP = 16
CORE_ONE_SIDED_CAP = 5


def _float(v: Any) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _int(v: Any) -> int | None:
    if v in (None, "", "None"):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _load_compare_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_metadata(dataset_dir: str, repo_root: Path) -> dict[str, Any]:
    meta_path = repo_root / dataset_dir / "metadata.yaml"
    result: dict[str, Any] = {
        "metadata_path": str(meta_path),
        "feature_count": None,
        "train_samples": None,
        "valid_samples": None,
        "id_test_samples": None,
        "ood_test_samples": None,
        "formula_line_count": None,
        "formula_char_count": None,
        "formula_operator_count": None,
    }
    if meta_path.exists():
        try:
            payload = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            dataset = payload.get("dataset") or {}
            features = dataset.get("features") or []
            splits = dataset.get("splits") or {}
            result["feature_count"] = len(features) if isinstance(features, list) else None
            for key, out_key in [
                ("train", "train_samples"),
                ("valid", "valid_samples"),
                ("id_test", "id_test_samples"),
                ("ood_test", "ood_test_samples"),
            ]:
                section = splits.get(key) or {}
                result[out_key] = section.get("samples")
        except Exception:
            pass

    formula_path = repo_root / dataset_dir / "formula.py"
    if formula_path.exists():
        text = formula_path.read_text(encoding="utf-8")
        result["formula_line_count"] = len(text.splitlines())
        result["formula_char_count"] = len(text)
        result["formula_operator_count"] = len(
            re.findall(r"sin|cos|exp|log|sqrt|tan|\\*\\*|\\+|\\-|\\*|/", text)
        )
    return result


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    py_id = _float(row.get("pysr_id_nmse_mean"))
    py_ood = _float(row.get("pysr_ood_nmse_mean"))
    ll_id = _float(row.get("llmsr_id_nmse_mean"))
    ll_ood = _float(row.get("llmsr_ood_nmse_mean"))
    py_id_r2 = _float(row.get("pysr_id_r2_mean"))
    py_ood_r2 = _float(row.get("pysr_ood_r2_mean"))
    ll_id_r2 = _float(row.get("llmsr_id_r2_mean"))
    ll_ood_r2 = _float(row.get("llmsr_ood_r2_mean"))
    py_cov = min(_int(row.get("pysr_id_r2_count")) or 0, _int(row.get("pysr_ood_r2_count")) or 0) / 3.0
    ll_cov = min(_int(row.get("llmsr_id_r2_count")) or 0, _int(row.get("llmsr_ood_r2_count")) or 0) / 3.0

    gap_parts: list[float] = []
    if None not in (py_id, ll_id) and py_id and ll_id and py_id > 0 and ll_id > 0:
        gap_parts.append(abs(math.log10(ll_id) - math.log10(py_id)))
    if None not in (py_ood, ll_ood) and py_ood and ll_ood and py_ood > 0 and ll_ood > 0:
        gap_parts.append(abs(math.log10(ll_ood) - math.log10(py_ood)))
    three_seed_gap = sum(gap_parts) / len(gap_parts) if gap_parts else 0.0
    gap_norm = min(three_seed_gap, 40.0) / 40.0

    best_ood_r2 = max([x for x in [py_ood_r2, ll_ood_r2] if x is not None], default=-1e9)
    best_ood_nmse = min([x for x in [py_ood, ll_ood] if x is not None], default=float("inf"))
    quality = 0.0
    if best_ood_r2 != -1e9:
        quality = max(quality, max(0.0, min(1.0, (best_ood_r2 + 1.0) / 1.5)))
    if math.isfinite(best_ood_nmse):
        quality = max(quality, max(0.0, min(1.0, (-math.log10(max(best_ood_nmse, 1e-12))) / 4.0)))
        if best_ood_nmse < 1:
            quality = max(quality, 0.55)
        if best_ood_nmse < 0.1:
            quality = max(quality, 0.75)
        if best_ood_nmse < 0.01:
            quality = max(quality, 0.9)

    stability = (py_cov + ll_cov) / 2.0
    priority_score = 100.0 * (0.55 * gap_norm + 0.25 * quality + 0.20 * stability)

    final_adv = None
    if None not in (py_id, py_ood, ll_id, ll_ood):
        py_mean = (py_id + py_ood) / 2.0
        ll_mean = (ll_id + ll_ood) / 2.0
        final_adv = "pysr" if py_mean < ll_mean else "llmsr"
    else:
        final_adv = row.get("candidate_advantage_side") or "unknown"

    row["priority_score"] = round(priority_score, 6)
    row["three_seed_gap"] = three_seed_gap
    row["quality_score"] = round(quality, 6)
    row["stability_score"] = round(stability, 6)
    row["final_advantage_side"] = final_adv
    return row


def _select_master100(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda x: x["priority_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    family_count = Counter()
    mode_count = Counter()
    adv_count = Counter()
    basename_used: set[str] = set()
    subgroup_count = Counter()
    family_availability = Counter()
    for row in sorted_rows:
        family_availability[row["family"]] += 1

    def family_cap(family: str) -> int:
        return min(MASTER100_FAMILY_TARGET[family], family_availability[family])

    def row_bonus(row: dict[str, Any]) -> float:
        family = row["family"]
        mode = row["selection_mode"]
        adv = row["candidate_advantage_side"]
        subgroup = row["subgroup"]
        bonus = 0.0
        bonus += 120.0 * max(0, family_cap(family) - family_count[family])
        bonus += 45.0 * max(0, MASTER100_SELECTION_MODE_TARGET[mode] - mode_count[mode])
        bonus += 30.0 * max(0, MASTER100_ADVANTAGE_TARGET[adv] - adv_count[adv])
        bonus -= 2.5 * subgroup_count[subgroup]
        return bonus

    while len(selected) < 100:
        best_row = None
        best_score = None
        for row in sorted_rows:
            family = row["family"]
            basename = row["basename"]
            subgroup = row["subgroup"]
            if basename in basename_used:
                continue
            if family_count[family] >= family_cap(family):
                continue
            if subgroup_count[subgroup] >= MASTER100_SUBGROUP_SOFT_CAP:
                continue
            score = row["priority_score"] + row_bonus(row)
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            break
        selected.append(best_row)
        family_count[best_row["family"]] += 1
        mode_count[best_row["selection_mode"]] += 1
        adv_count[best_row["candidate_advantage_side"]] += 1
        subgroup_count[best_row["subgroup"]] += 1
        basename_used.add(best_row["basename"])

    if len(selected) != 100:
        raise RuntimeError(f"无法选满 Master-100，当前只选出 {len(selected)} 个")

    audit = {
        "family_target": dict(MASTER100_FAMILY_TARGET),
        "family_realized": dict(family_count),
        "family_cap_used": {k: family_cap(k) for k in MASTER100_FAMILY_TARGET},
        "mode_target": dict(MASTER100_SELECTION_MODE_TARGET),
        "mode_realized": dict(mode_count),
        "adv_target": dict(MASTER100_ADVANTAGE_TARGET),
        "adv_realized": dict(adv_count),
        "max_subgroup_count": max(subgroup_count.values()) if subgroup_count else 0,
    }
    return selected, audit


def _static_vector(row: dict[str, Any]) -> dict[str, float]:
    return {
        "feature_count": float(_int(row.get("feature_count")) or 0),
        "train_log": math.log10(max((_int(row.get("train_samples")) or 1), 1)),
        "valid_log": math.log10(max((_int(row.get("valid_samples")) or 1), 1)),
        "id_log": math.log10(max((_int(row.get("id_test_samples")) or 1), 1)),
        "ood_log": math.log10(max((_int(row.get("ood_test_samples")) or 1), 1)),
        "formula_lines": float(_int(row.get("formula_line_count")) or 0),
        "formula_chars": math.log10(max((_int(row.get("formula_char_count")) or 1), 1)),
        "formula_ops": float(_int(row.get("formula_operator_count")) or 0),
    }


def _compute_split_loss(dev: list[dict[str, Any]], core: list[dict[str, Any]]) -> float:
    def count(rows: list[dict[str, Any]], key: str) -> Counter:
        return Counter(r[key] for r in rows)

    dev_family = count(dev, "family")
    core_family = count(core, "family")
    dev_subgroup = count(dev, "subgroup")
    core_subgroup = count(core, "subgroup")
    dev_mode = count(dev, "selection_mode")
    core_mode = count(core, "selection_mode")
    dev_adv = count(dev, "candidate_advantage_side")
    core_adv = count(core, "candidate_advantage_side")

    total = 0.0
    for dev_c, core_c in ((dev_family, core_family), (dev_mode, core_mode), (dev_adv, core_adv)):
        keys = set(dev_c) | set(core_c)
        total += sum(abs(dev_c[k] - core_c[k]) for k in keys)

    keys = set(dev_subgroup) | set(core_subgroup)
    total += 0.3 * sum(abs(dev_subgroup[k] - core_subgroup[k]) for k in keys)

    if dev and core:
        for feature_key in _static_vector(dev[0]).keys():
            dev_vals = [_static_vector(r)[feature_key] for r in dev]
            core_vals = [_static_vector(r)[feature_key] for r in core]
            total += 0.8 * abs(sum(dev_vals) / len(dev) - sum(core_vals) / len(core))

    total += 4.0 * abs(len(dev) - len(core))
    total += 1000.0 * max(0, sum(1 for r in core if r["selection_mode"] == "one-sided") - CORE_ONE_SIDED_CAP)
    return total


def _rebalance_core_one_sided(dev: list[dict[str, Any]], core: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    while sum(1 for r in core if r["selection_mode"] == "one-sided") > CORE_ONE_SIDED_CAP:
        best_pair = None
        best_loss = None
        for core_row in core:
            if core_row["selection_mode"] != "one-sided":
                continue
            for dev_row in dev:
                if dev_row["selection_mode"] == "one-sided":
                    continue
                new_dev = [r for r in dev if r is not dev_row] + [core_row]
                new_core = [r for r in core if r is not core_row] + [dev_row]
                loss = _compute_split_loss(new_dev, new_core)
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_pair = (dev_row, core_row)
        if best_pair is None:
            break
        dev_row, core_row = best_pair
        dev = [r for r in dev if r is not dev_row] + [core_row]
        core = [r for r in core if r is not core_row] + [dev_row]
    return dev, core


def _split_master100(master: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    strata = defaultdict(list)
    for row in master:
        key = (row["family"], row["selection_mode"], row["candidate_advantage_side"])
        strata[key].append(row)

    dev: list[dict[str, Any]] = []
    core: list[dict[str, Any]] = []
    dev_counts = {"family": Counter(), "subgroup": Counter(), "mode": Counter(), "adv": Counter()}
    core_counts = {"family": Counter(), "subgroup": Counter(), "mode": Counter(), "adv": Counter()}
    dev_sums = Counter()
    core_sums = Counter()

    def add_to(bucket: list[dict[str, Any]], counters: dict[str, Counter], sums: Counter, row: dict[str, Any]) -> None:
        bucket.append(row)
        counters["family"][row["family"]] += 1
        counters["subgroup"][row["subgroup"]] += 1
        counters["mode"][row["selection_mode"]] += 1
        counters["adv"][row["candidate_advantage_side"]] += 1
        for k, v in _static_vector(row).items():
            sums[k] += v

    def loss() -> float:
        total = 0.0
        for name in ("family", "mode", "adv"):
            keys = set(dev_counts[name]) | set(core_counts[name])
            total += sum(abs(dev_counts[name][k] - core_counts[name][k]) for k in keys)
        for name in ("subgroup",):
            keys = set(dev_counts[name]) | set(core_counts[name])
            total += 0.3 * sum(abs(dev_counts[name][k] - core_counts[name][k]) for k in keys)
        if dev and core:
            for key in dev_sums:
                total += 0.8 * abs(dev_sums[key] / len(dev) - core_sums[key] / len(core))
        total += 4.0 * abs(len(dev) - len(core))
        total += 1000.0 * max(0, core_counts["mode"]["one-sided"] - CORE_ONE_SIDED_CAP)
        return total

    leftovers: list[dict[str, Any]] = []
    for key, items in strata.items():
        items = sorted(
            items,
            key=lambda x: (
                x["subgroup"],
                -(_int(x.get("feature_count")) or 0),
                -(_int(x.get("train_samples")) or 0),
                -(_int(x.get("formula_char_count")) or 0),
                x["dataset_dir"],
            ),
        )
        for i in range(0, len(items) - 1, 2):
            a = items[i]
            b = items[i + 1]
            # option 1
            add_to(dev, dev_counts, dev_sums, a)
            add_to(core, core_counts, core_sums, b)
            loss1 = loss()
            dev.pop()
            core.pop()
            for counter_name, row in [("family", a["family"]), ("subgroup", a["subgroup"]), ("mode", a["selection_mode"]), ("adv", a["candidate_advantage_side"])]:
                dev_counts[counter_name][row] -= 1
            for counter_name, row in [("family", b["family"]), ("subgroup", b["subgroup"]), ("mode", b["selection_mode"]), ("adv", b["candidate_advantage_side"])]:
                core_counts[counter_name][row] -= 1
            for k, v in _static_vector(a).items():
                dev_sums[k] -= v
            for k, v in _static_vector(b).items():
                core_sums[k] -= v
            # option 2
            add_to(dev, dev_counts, dev_sums, b)
            add_to(core, core_counts, core_sums, a)
            loss2 = loss()
            dev.pop()
            core.pop()
            for counter_name, row in [("family", b["family"]), ("subgroup", b["subgroup"]), ("mode", b["selection_mode"]), ("adv", b["candidate_advantage_side"])]:
                dev_counts[counter_name][row] -= 1
            for counter_name, row in [("family", a["family"]), ("subgroup", a["subgroup"]), ("mode", a["selection_mode"]), ("adv", a["candidate_advantage_side"])]:
                core_counts[counter_name][row] -= 1
            for k, v in _static_vector(b).items():
                dev_sums[k] -= v
            for k, v in _static_vector(a).items():
                core_sums[k] -= v
            # commit best
            if loss1 <= loss2:
                add_to(dev, dev_counts, dev_sums, a)
                add_to(core, core_counts, core_sums, b)
            else:
                add_to(dev, dev_counts, dev_sums, b)
                add_to(core, core_counts, core_sums, a)
        if len(items) % 2 == 1:
            leftovers.append(items[-1])

    for row in leftovers:
        if len(dev) < len(core):
            add_to(dev, dev_counts, dev_sums, row)
        elif len(core) < len(dev):
            add_to(core, core_counts, core_sums, row)
        else:
            # choose side with lower marginal loss
            add_to(dev, dev_counts, dev_sums, row)
            l1 = loss()
            dev.pop()
            for counter_name, val in [("family", row["family"]), ("subgroup", row["subgroup"]), ("mode", row["selection_mode"]), ("adv", row["candidate_advantage_side"])]:
                dev_counts[counter_name][val] -= 1
            for k, v in _static_vector(row).items():
                dev_sums[k] -= v
            add_to(core, core_counts, core_sums, row)
            l2 = loss()
            core.pop()
            for counter_name, val in [("family", row["family"]), ("subgroup", row["subgroup"]), ("mode", row["selection_mode"]), ("adv", row["candidate_advantage_side"])]:
                core_counts[counter_name][val] -= 1
            for k, v in _static_vector(row).items():
                core_sums[k] -= v
            if l1 <= l2:
                add_to(dev, dev_counts, dev_sums, row)
            else:
                add_to(core, core_counts, core_sums, row)

    if len(dev) != 50 or len(core) != 50:
        raise RuntimeError(f"切分失败，Dev={len(dev)}，Core={len(core)}")

    dev, core = _rebalance_core_one_sided(dev, core)
    if len(dev) != 50 or len(core) != 50:
        raise RuntimeError(f"one-sided 再平衡后切分失败，Dev={len(dev)}，Core={len(core)}")

    audit = {
        "dev_family": dict(Counter(r["family"] for r in dev)),
        "core_family": dict(Counter(r["family"] for r in core)),
        "dev_subgroup_top": dict(Counter(r["subgroup"] for r in dev).most_common()),
        "core_subgroup_top": dict(Counter(r["subgroup"] for r in core).most_common()),
        "dev_mode": dict(Counter(r["selection_mode"] for r in dev)),
        "core_mode": dict(Counter(r["selection_mode"] for r in core)),
        "dev_adv": dict(Counter(r["candidate_advantage_side"] for r in dev)),
        "core_adv": dict(Counter(r["candidate_advantage_side"] for r in core)),
        "core_one_sided_cap": CORE_ONE_SIDED_CAP,
        "core_one_sided_realized": sum(1 for r in core if r["selection_mode"] == "one-sided"),
        "split_loss": _compute_split_loss(dev, core),
    }
    return dev, core, audit


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = [_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "median": None}
    return {"count": len(vals), "mean": sum(vals) / len(vals), "median": statistics.median(vals)}


def _write_audit_md(
    path: Path,
    master_audit: dict[str, Any],
    split_audit: dict[str, Any],
    master: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    core: list[dict[str, Any]],
) -> None:
    lines = [
        "# Master-100 / Dev-50 / Core-50 切分审计报告",
        "",
        "## 方案说明",
        "",
        "- `Master-100`：允许使用三 seed 正式结果进行筛选。",
        "- `Dev-50 / Core-50`：只使用非结果信息切分，包括 `family / subgroup / selection_mode / candidate_advantage_side / basename / 特征维度 / 样本量 / 静态公式复杂度`。",
        "- `basename <= 1` 在整个 `Master-100` 上成立，避免 dev/test 结构泄漏。",
        "",
        f"- Master-100 数量：`{len(master)}`",
        f"- Dev-50 数量：`{len(dev)}`",
        f"- Core-50 数量：`{len(core)}`",
        "",
        "## Master-100 配额实现",
        "",
        "### family（目标 / 实现 / 实际上限）",
    ]
    for fam in MASTER100_FAMILY_TARGET:
        lines.append(
            f"- `{fam}`: target=`{master_audit['family_target'].get(fam)}`, "
            f"realized=`{master_audit['family_realized'].get(fam, 0)}`, "
            f"cap_used=`{master_audit['family_cap_used'].get(fam, 0)}`"
        )
    lines.extend([
        "",
        "### selection_mode（目标 / 实现）",
    ])
    for key in MASTER100_SELECTION_MODE_TARGET:
        lines.append(
            f"- `{key}`: target=`{master_audit['mode_target'].get(key)}`, "
            f"realized=`{master_audit['mode_realized'].get(key, 0)}`"
        )
    lines.extend([
        "",
        "### candidate_advantage_side（目标 / 实现）",
    ])
    for key in MASTER100_ADVANTAGE_TARGET:
        lines.append(
            f"- `{key}`: target=`{master_audit['adv_target'].get(key)}`, "
            f"realized=`{master_audit['adv_realized'].get(key, 0)}`"
        )
    lines.extend([
        "",
        f"- Master-100 最大 subgroup 占用：`{master_audit['max_subgroup_count']}`",
        "",
        "## 离散分布",
        "",
        "### family",
    ])
    for fam in sorted(split_audit["dev_family"]):
        lines.append(f"- `{fam}`: dev=`{split_audit['dev_family'].get(fam, 0)}`, core=`{split_audit['core_family'].get(fam, 0)}`")
    lines.extend(["", "### selection_mode"])
    for key in sorted(split_audit["dev_mode"]):
        lines.append(f"- `{key}`: dev=`{split_audit['dev_mode'].get(key, 0)}`, core=`{split_audit['core_mode'].get(key, 0)}`")
    lines.extend(["", "### candidate_advantage_side"])
    for key in sorted(split_audit["dev_adv"]):
        lines.append(f"- `{key}`: dev=`{split_audit['dev_adv'].get(key, 0)}`, core=`{split_audit['core_adv'].get(key, 0)}`")
    lines.extend([
        "",
        f"- `Core-50 one-sided` 上限：`{split_audit['core_one_sided_cap']}`，实际：`{split_audit['core_one_sided_realized']}`",
    ])
    lines.extend(["", "## 连续静态特征摘要"])
    for key in [
        "feature_count",
        "train_samples",
        "valid_samples",
        "id_test_samples",
        "ood_test_samples",
        "formula_line_count",
        "formula_char_count",
        "formula_operator_count",
    ]:
        ds = _numeric_summary(dev, key)
        cs = _numeric_summary(core, key)
        lines.append(
            f"- `{key}`: dev(mean=`{ds['mean']}`, median=`{ds['median']}`, n=`{ds['count']}`), "
            f"core(mean=`{cs['mean']}`, median=`{cs['median']}`, n=`{cs['count']}`)"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Master-100 / Dev-50 / Core-50")
    parser.add_argument("--compare-csv", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--repo-root", default="/home/family/workplace/scientific-intelligent-modelling")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = _load_compare_rows(Path(args.compare_csv))
    candidate_payload = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    candidate_map: dict[str, dict[str, Any]] = {}
    for pool_name in ("pool_A", "pool_B", "pool_C"):
        for item in candidate_payload[pool_name]:
            candidate_map[item["dataset_dir"]] = item

    scored: list[dict[str, Any]] = []
    repo_root = Path(args.repo_root)
    for row in rows:
        row = dict(row)
        row.update(_load_metadata(row["dataset_dir"], repo_root))
        if row["dataset_dir"] in candidate_map:
            row["candidate_pool"] = candidate_map[row["dataset_dir"]]["pool"]
        scored.append(_score_row(row))

    master, master_audit = _select_master100(scored)
    dev, core, split_audit = _split_master100(master)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_csv = output_dir / "master100_candidates.csv"
    dev_csv = output_dir / "benchmark_dev50.csv"
    core_csv = output_dir / "benchmark_core50.csv"
    audit_json = output_dir / "benchmark_dev_core_split_audit.json"
    audit_md = output_dir / "benchmark_dev_core_split_audit.md"

    _write_csv(master_csv, sorted(master, key=lambda x: x["priority_score"], reverse=True))
    _write_csv(dev_csv, sorted(dev, key=lambda x: x["priority_score"], reverse=True))
    _write_csv(core_csv, sorted(core, key=lambda x: x["priority_score"], reverse=True))
    audit_json.write_text(
        json.dumps(
            {
                "master_selection": master_audit,
                "split_audit": split_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_audit_md(audit_md, master_audit, split_audit, master, dev, core)

    print(
        json.dumps(
            {
                "master_csv": str(master_csv),
                "dev_csv": str(dev_csv),
                "core_csv": str(core_csv),
                "audit_json": str(audit_json),
                "audit_md": str(audit_md),
                "master_size": len(master),
                "dev_size": len(dev),
                "core_size": len(core),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
