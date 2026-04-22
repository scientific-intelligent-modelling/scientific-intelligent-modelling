#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import sympy as sp


REPO_ROOT = Path("/home/family/workplace/scientific-intelligent-modelling")
DEFAULT_MASTER = REPO_ROOT / "experiment-results/benchmark_formal200_20260417/dev_core_split_v1/master100_candidates.csv"
DEFAULT_CANDIDATE = REPO_ROOT / "experiment-results/benchmark_formal200_20260417/three_seed_formal_dataset_compare.csv"
DEFAULT_TASK = REPO_ROOT / "experiment-results/benchmark_formal200_20260417/three_seed_formal_task_results.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiment-results/benchmark_formal200_20260417/clean_master100_v1"

SELECTION_MODE_RANK = {
    "strict": 4,
    "mid-gap": 3,
    "relaxed": 2,
    "one-sided": 1,
}

SYM_LOCALS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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


def _contains_dummy(text: str) -> bool:
    return "dummy" in (text or "").lower()


class _FormulaArgNormalizer(ast.NodeTransformer):
    def __init__(self, arg_map: dict[str, str]):
        self.arg_map = arg_map

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.arg_map:
            return ast.copy_location(ast.Name(id=self.arg_map[node.id], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy", "math"}:
            return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        return node


def normalize_feynman_name(name: str) -> str | None:
    if not name:
        return None
    lowered = name.lower()
    if not lowered.startswith("feynman"):
        return None
    normalized = lowered.replace("-", ".").replace("_", ".")
    normalized = re.sub(r"\.+", ".", normalized).strip(".")
    return normalized


def canonicalize_formula_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "formula_parse_ok": False,
            "formula_error": "missing_formula_py",
            "formula_hash": None,
            "formula_normalized_expression": None,
        }

    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if func is None:
            raise ValueError("formula.py 中未找到函数定义")
        ret = next((n.value for n in func.body if isinstance(n, ast.Return)), None)
        if ret is None:
            raise ValueError("formula.py 中未找到 return 语句")
        arg_map = {arg.arg: f"x{i}" for i, arg in enumerate(func.args.args)}
        normalized_ret = _FormulaArgNormalizer(arg_map).visit(ast.fix_missing_locations(ret))
        expr_src = ast.unparse(normalized_ret)
        expr = sp.sympify(expr_src, locals=SYM_LOCALS)
        expr = sp.simplify(expr)
        normalized_expr = sp.sstr(expr)
        return {
            "formula_parse_ok": True,
            "formula_error": None,
            "formula_hash": hashlib.sha1(sp.srepr(expr).encode("utf-8")).hexdigest(),
            "formula_normalized_expression": normalized_expr,
        }
    except Exception as exc:
        return {
            "formula_parse_ok": False,
            "formula_error": repr(exc),
            "formula_hash": None,
            "formula_normalized_expression": None,
        }


def semantic_keys_for_row(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    formula_hash = row.get("formula_hash")
    if formula_hash:
        keys.add(f"formula:{formula_hash}")
    feynman_key = row.get("feynman_key")
    if feynman_key:
        keys.add(f"feynman:{feynman_key}")
    return keys


def build_semantic_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    key_to_indices: dict[str, set[int]] = defaultdict(set)
    row_keys: list[set[str]] = []
    for idx, row in enumerate(rows):
        keys = semantic_keys_for_row(row)
        row_keys.append(keys)
        for key in keys:
            key_to_indices[key].add(idx)

    groups: list[list[dict[str, Any]]] = []
    visited: set[int] = set()
    for start in range(len(rows)):
        if start in visited:
            continue
        frontier = [start]
        component: set[int] = set()
        while frontier:
            idx = frontier.pop()
            if idx in component:
                continue
            component.add(idx)
            for key in row_keys[idx]:
                frontier.extend(key_to_indices[key] - component)
        visited |= component
        if len(component) > 1:
            groups.append([rows[i] for i in sorted(component)])
    return groups


def row_survival_score(row: dict[str, Any]) -> tuple[float, float, float, int, int]:
    return (
        _f(row.get("priority_score")) or float("-inf"),
        _f(row.get("quality_score")) or float("-inf"),
        _f(row.get("stability_score")) or float("-inf"),
        SELECTION_MODE_RANK.get(row.get("selection_mode", ""), 0),
        0 if _contains_dummy(row.get("subgroup", "")) else 1,
    )


def classify_status_semantics(row: dict[str, str]) -> str:
    full_metrics = all(
        row.get(k) not in ("", None, "None")
        for k in ("valid_r2", "id_r2", "ood_r2", "valid_nmse", "id_nmse", "ood_nmse")
    )
    has_artifact = str(row.get("canonical_artifact_present", "")).lower() == "true"
    has_any_metrics = any(
        row.get(k) not in ("", None, "None")
        for k in ("valid_r2", "id_r2", "ood_r2", "valid_nmse", "id_nmse", "ood_nmse")
    )
    task_status = row.get("task_status")
    result_status = row.get("result_status")

    if full_metrics:
        if task_status == "timed_out":
            return "budget_exhausted_with_output"
        return "ok_full"
    if has_artifact or has_any_metrics:
        if task_status == "timed_out":
            return "budget_exhausted_with_output"
        return "partial_output"
    if result_status == "error" or task_status == "error":
        return "no_valid_output"
    if task_status == "timed_out":
        return "no_valid_output"
    return "partial_output"


def replacement_score(source: dict[str, Any], cand: dict[str, Any]) -> float:
    score = 0.0
    if cand["family"] == source["family"]:
        score += 500.0
    if cand["subgroup"] == source["subgroup"]:
        score += 250.0
    if cand["selection_mode"] == source["selection_mode"]:
        score += 160.0
    if cand["candidate_advantage_side"] == source["candidate_advantage_side"]:
        score += 120.0
    if _contains_dummy(cand["subgroup"]) == _contains_dummy(source["subgroup"]):
        score += 40.0

    score += (_f(cand.get("priority_score")) or 0.0)

    for key, weight in [
        ("feature_count", 20.0),
        ("train_samples", 8.0),
        ("id_test_samples", 6.0),
        ("ood_test_samples", 6.0),
        ("formula_operator_count", 10.0),
    ]:
        a = _f(source.get(key))
        b = _f(cand.get(key))
        if a is None or b is None:
            continue
        if key.endswith("_samples"):
            score -= weight * abs(math.log10(max(a, 1)) - math.log10(max(b, 1)))
        else:
            score -= weight * abs(a - b)
    return score


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        dataset_dir = out["dataset_dir"]
        formula_info = canonicalize_formula_file(REPO_ROOT / dataset_dir / "formula.py")
        out.update(formula_info)
        out["feynman_key"] = normalize_feynman_name(out.get("dataset_name", ""))
        enriched.append(out)
    return enriched


def select_replacement(
    source_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    selected_dataset_dirs: set[str],
    selected_semantic_keys: set[str],
) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_score = None
    for cand in candidate_rows:
        dataset_dir = cand["dataset_dir"]
        if dataset_dir in selected_dataset_dirs:
            continue
        cand_keys = semantic_keys_for_row(cand)
        if cand_keys & selected_semantic_keys:
            continue
        score = replacement_score(source_row, cand)
        if best_score is None or score > best_score:
            best = cand
            best_score = score
    return best, best_score or float("-inf")


def build_clean_master100(
    master_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_by_dataset = {row["dataset_dir"]: row for row in candidate_rows}
    master_dataset_dirs = {row["dataset_dir"] for row in master_rows}
    groups = build_semantic_groups(candidate_rows)

    duplicate_group_rows: list[dict[str, Any]] = []
    groups_on_master: list[list[dict[str, Any]]] = []
    for gid, group in enumerate(groups, start=1):
        members_in_master = [row for row in group if row["dataset_dir"] in master_dataset_dirs]
        if len(members_in_master) > 1:
            groups_on_master.append(members_in_master)
        for row in group:
            duplicate_group_rows.append(
                {
                    "group_id": gid,
                    "group_size": len(group),
                    "dataset_dir": row["dataset_dir"],
                    "dataset_name": row["dataset_name"],
                    "family": row["family"],
                    "subgroup": row["subgroup"],
                    "in_master100": row["dataset_dir"] in master_dataset_dirs,
                    "formula_hash": row.get("formula_hash"),
                    "feynman_key": row.get("feynman_key"),
                }
            )

    survivors: dict[str, dict[str, Any]] = {}
    removed_dataset_dirs: set[str] = set()
    for group in groups_on_master:
        survivor = max(group, key=row_survival_score)
        survivors[survivor["dataset_dir"]] = survivor
        for row in group:
            if row["dataset_dir"] != survivor["dataset_dir"]:
                removed_dataset_dirs.add(row["dataset_dir"])

    # 固定保留的行先整体建模，替补必须避开它们的语义键，不能只避开“已经遍历到的前缀”。
    kept_rows = [row for row in master_rows if row["dataset_dir"] not in removed_dataset_dirs]
    kept_dataset_dirs = {row["dataset_dir"] for row in kept_rows}
    kept_semantic_keys: set[str] = set()
    for row in kept_rows:
        kept_semantic_keys |= semantic_keys_for_row(row)

    selected_rows: list[dict[str, Any]] = []
    replacement_dataset_dirs: set[str] = set()
    replacement_semantic_keys: set[str] = set()
    replacement_log: list[dict[str, Any]] = []

    replacement_sources_by_dir = {
        row["dataset_dir"]: row for row in master_rows if row["dataset_dir"] in removed_dataset_dirs
    }

    for row in master_rows:
        dataset_dir = row["dataset_dir"]
        if dataset_dir in removed_dataset_dirs:
            source_row = replacement_sources_by_dir[dataset_dir]
            replacement, score = select_replacement(
                source_row,
                candidate_rows,
                kept_dataset_dirs | replacement_dataset_dirs | removed_dataset_dirs,
                kept_semantic_keys | replacement_semantic_keys,
            )
            if replacement is None:
                raise RuntimeError(f"无法为被删除样本补位：{dataset_dir}")
            replacement_log.append(
                {
                    "removed_dataset_dir": dataset_dir,
                    "removed_dataset_name": row["dataset_name"],
                    "removed_family": row["family"],
                    "removed_subgroup": row["subgroup"],
                    "removed_selection_mode": row["selection_mode"],
                    "replacement_dataset_dir": replacement["dataset_dir"],
                    "replacement_dataset_name": replacement["dataset_name"],
                    "replacement_family": replacement["family"],
                    "replacement_subgroup": replacement["subgroup"],
                    "replacement_selection_mode": replacement["selection_mode"],
                    "replacement_score": round(score, 6),
                    "replacement_reason": "constraint_nearest_neighbor",
                }
            )
            selected_rows.append(replacement)
            replacement_dataset_dirs.add(replacement["dataset_dir"])
            replacement_semantic_keys |= semantic_keys_for_row(replacement)
        else:
            selected_rows.append(row)

    if len(selected_rows) != 100:
        raise RuntimeError(f"Clean-Master-100 行数异常：{len(selected_rows)}")
    if len({row["dataset_dir"] for row in selected_rows}) != 100:
        raise RuntimeError("Clean-Master-100 出现重复 dataset_dir")

    return selected_rows, duplicate_group_rows, replacement_log


def build_audit_md(
    path: Path,
    *,
    master_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    duplicate_group_rows: list[dict[str, Any]],
    replacement_log: list[dict[str, Any]],
) -> None:
    def counter(rows: Iterable[dict[str, Any]], key: str) -> Counter[str]:
        return Counter(row.get(key, "") for row in rows)

    before_family = counter(master_rows, "family")
    after_family = counter(clean_rows, "family")
    before_mode = counter(master_rows, "selection_mode")
    after_mode = counter(clean_rows, "selection_mode")
    before_adv = counter(master_rows, "candidate_advantage_side")
    after_adv = counter(clean_rows, "candidate_advantage_side")

    duplicate_master_groups = defaultdict(list)
    master_set = {row["dataset_dir"] for row in master_rows}
    for row in duplicate_group_rows:
        if row["in_master100"]:
            duplicate_master_groups[row["group_id"]].append(row)
    duplicate_master_groups = {k: v for k, v in duplicate_master_groups.items() if len(v) > 1}

    text = f"""# Clean-Master-100 审计报告

## 总结

- 输入 `Master-100`：`{len(master_rows)}` 个
- 输出 `Clean-Master-100`：`{len(clean_rows)}` 个
- 在 `Candidate-200` 全局语义审计中发现的重复组数：`{len({r['group_id'] for r in duplicate_group_rows})}`
- 当前 `Master-100` 内实际命中的重复组数：`{len(duplicate_master_groups)}`
- 本次删除并补位条数：`{len(replacement_log)}`

## 删除规则

本次只删除两类：

1. semantic duplicate
2. 结构性坏样本（本轮若存在）

不处理：

- 质量分数偏低
- `strict` 太多
- `srsd` 太多
- 某一算法表现偏强/偏弱

## 清洗前后分布

### family

- before: {dict(before_family)}
- after: {dict(after_family)}

### selection_mode

- before: {dict(before_mode)}
- after: {dict(after_mode)}

### candidate_advantage_side

- before: {dict(before_adv)}
- after: {dict(after_adv)}

## 被识别出的 `Master-100` 内重复组

"""
    for gid, members in sorted(duplicate_master_groups.items()):
        text += f"\n### group {gid}\n\n"
        for row in members:
            text += f"- `{row['dataset_name']}` | `{row['dataset_dir']}`\n"

    text += "\n## 补位记录\n\n"
    for row in replacement_log:
        text += (
            f"- 删除 `{row['removed_dataset_name']}`，补入 `{row['replacement_dataset_name']}`；"
            f"family `{row['removed_family']} -> {row['replacement_family']}`，"
            f"subgroup `{row['removed_subgroup']} -> {row['replacement_subgroup']}`，"
            f"score=`{row['replacement_score']}`\n"
        )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: 清洗当前 Master-100，做最小扰动补位。")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--task-csv", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    master_rows = enrich_rows(_load_csv(args.master_csv))
    candidate_rows = enrich_rows(_load_csv(args.candidate_csv))
    clean_rows, duplicate_group_rows, replacement_log = build_clean_master100(master_rows, candidate_rows)

    _dump_csv(output_dir / "clean_master100.csv", clean_rows)
    _dump_csv(output_dir / "duplicate_groups.csv", duplicate_group_rows)
    _dump_csv(output_dir / "replacement_log.csv", replacement_log)

    task_rows = _load_csv(args.task_csv)
    clean_dataset_dirs = {row["dataset_dir"] for row in clean_rows}
    status_rows = []
    for row in task_rows:
        if row["dataset_dir"] not in clean_dataset_dirs:
            continue
        out = dict(row)
        out["status_semantics"] = classify_status_semantics(row)
        status_rows.append(out)
    _dump_csv(output_dir / "status_semantics_map.csv", status_rows)

    build_audit_md(
        output_dir / "clean_master100_audit.md",
        master_rows=master_rows,
        clean_rows=clean_rows,
        duplicate_group_rows=duplicate_group_rows,
        replacement_log=replacement_log,
    )

    summary = {
        "master_input_count": len(master_rows),
        "clean_output_count": len(clean_rows),
        "duplicate_group_count": len({row["group_id"] for row in duplicate_group_rows}),
        "replacement_count": len(replacement_log),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
