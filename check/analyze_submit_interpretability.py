from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scientific_intelligent_modelling.benchmarks.metrics import (
    normalized_tree_edit_distance,
    srbench_model_size,
    srbench_symbolic_solution,
)
from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_build_canonical_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 submit 结果批量计算可解释性指标")
    parser.add_argument(
        "--submit-dir",
        required=True,
        help="submit 结果根目录，例如 experiments/submit/three_tools_3seeds_2h_blt35_20260413_013856",
    )
    parser.add_argument(
        "--examples-dir",
        default="examples",
        help="examples 数据集目录，用于读取 ground truth formula 与 metadata",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层不是对象: {path}")
    return data


def _strip_math_prefixes(expr: str) -> str:
    return expr.replace("np.", "").replace("numpy.", "").replace("math.", "")


def _extract_formula_expression(formula_path: Path) -> tuple[str | None, list[str]]:
    source = formula_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(formula_path))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        arg_names = [arg.arg for arg in node.args.args]
        for stmt in node.body:
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                expr = ast.unparse(stmt.value).strip()
                expr = _strip_math_prefixes(expr)
                for idx, name in enumerate(arg_names):
                    expr = re.sub(rf"\b{re.escape(name)}\b", f"x{idx}", expr)
                return expr, arg_names
    return None, []


def _load_ground_truth(dataset_name: str, examples_dir: Path) -> tuple[str | None, list[str], str | None]:
    dataset_dir = examples_dir / dataset_name
    if not dataset_dir.is_dir():
        return None, [], f"examples 数据集不存在: {dataset_dir}"

    metadata = _load_yaml(dataset_dir / "metadata.yaml").get("dataset", {})
    features = metadata.get("features") or []
    feature_names = [str(item.get("name")) for item in features if isinstance(item, dict) and item.get("name")]
    formula_info = metadata.get("ground_truth_formula") or {}
    formula_file = formula_info.get("file")
    if not isinstance(formula_file, str) or not formula_file.strip():
        return None, feature_names, None

    formula_path = dataset_dir / formula_file
    if not formula_path.exists():
        return None, feature_names, f"ground truth formula 文件不存在: {formula_path}"

    expr, arg_names = _extract_formula_expression(formula_path)
    if expr is None:
        return None, feature_names, f"无法从 {formula_path} 抽取 return 表达式"
    if feature_names and arg_names and len(feature_names) != len(arg_names):
        return expr, feature_names, (
            f"formula 参数个数 {len(arg_names)} 与 metadata.features 个数 {len(feature_names)} 不一致"
        )
    return expr, feature_names, None


def _iter_candidate_param_files(experiments_dir: Path):
    patterns = [
        "**/samples/top*.json",
        "**/best_history/*.json",
        "**/samples/samples_*.json",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        for path in experiments_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            yield path


def _load_best_params_for_equation(result_path: Path, equation_text: str) -> tuple[list[float] | None, str | None]:
    experiments_dir = result_path.parent / "experiments"
    if not experiments_dir.is_dir():
        return None, f"实验目录不存在: {experiments_dir}"

    exact_matches: list[tuple[float, list[float], Path]] = []
    fallback_matches: list[tuple[float, list[float], Path]] = []

    for path in _iter_candidate_param_files(experiments_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        function_text = payload.get("function") or payload.get("equation")
        params = payload.get("params")
        if not isinstance(function_text, str) or not isinstance(params, list):
            continue
        try:
            params = [float(x) for x in params]
        except Exception:
            continue

        score = payload.get("score")
        try:
            score_value = float(score) if score is not None else float("-inf")
        except Exception:
            score_value = float("-inf")

        candidate = (score_value, params, path)
        if function_text.strip() == equation_text.strip():
            exact_matches.append(candidate)
        elif path.name.startswith("top01_"):
            fallback_matches.append(candidate)

    if exact_matches:
        exact_matches.sort(key=lambda item: item[0], reverse=True)
        best = exact_matches[0]
        return best[1], f"exact_match:{best[2].name}"

    if fallback_matches:
        fallback_matches.sort(key=lambda item: item[0], reverse=True)
        best = fallback_matches[0]
        return best[1], f"top01_fallback:{best[2].name}"

    return None, "未找到匹配的 params"


def _resolve_optional_param_guards(equation_text: str, param_values: list[float] | None) -> str:
    if not equation_text or param_values is None:
        return equation_text

    param_count = len(param_values)

    def repl(match: re.Match[str]) -> str:
        param_idx = int(match.group(1))
        threshold = int(match.group(2))
        fallback = match.group(3).strip()
        return f"params[{param_idx}]" if param_count > threshold else fallback

    pattern = re.compile(
        r"\(\s*params\[(\d+)\]\s*if\s*len\(params\)\s*>\s*(\d+)\s*else\s*([^)]+?)\s*\)"
    )
    return pattern.sub(repl, equation_text)


def _safe_bool_mean(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.astype(float).mean())


def _safe_float_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _safe_float_std(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.std(ddof=0))


def main() -> None:
    args = parse_args()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    examples_dir = Path(args.examples_dir).expanduser().resolve()

    if not submit_dir.is_dir():
        raise SystemExit(f"submit 目录不存在: {submit_dir}")
    if not examples_dir.is_dir():
        raise SystemExit(f"examples 目录不存在: {examples_dir}")

    run_rows: list[dict[str, Any]] = []

    for result_path in sorted(submit_dir.glob("**/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        tool = str(payload.get("tool") or "").strip()
        dataset = str(payload.get("dataset") or "").strip()
        equation_text = str(payload.get("equation") or "").strip()
        status = str(payload.get("status") or "")
        seed = payload.get("seed")
        feature_names = payload.get("feature_names") or []
        expected_n_features = len(feature_names) if isinstance(feature_names, list) and feature_names else None

        gt_expr, gt_feature_names, gt_error = _load_ground_truth(dataset, examples_dir)
        param_values = None
        param_source = None
        param_error = None
        if tool.lower() in {"llmsr", "drsr"} and equation_text:
            param_values, param_source = _load_best_params_for_equation(result_path, equation_text)
            if param_values is None:
                param_error = param_source
        equation_text = _resolve_optional_param_guards(equation_text, param_values)
        artifact, artifact_error = safe_build_canonical_artifact(
            tool_name=tool,
            equation=equation_text,
            expected_n_features=expected_n_features,
            parameter_values=param_values,
        )

        normalized_expr = None
        instantiated_expr = None
        ast_node_count = None
        tree_depth = None
        operator_set = None
        if artifact is not None:
            normalized_expr = artifact.get("normalized_expression")
            instantiated_expr = artifact.get("instantiated_expression") or normalized_expr
            ast_node_count = artifact.get("ast_node_count")
            tree_depth = artifact.get("tree_depth")
            operator_set = ",".join(artifact.get("operator_set") or [])

        size_raw = {}
        size_simplified = {}
        symbolic_solution = {}
        ted = {}
        metric_error = None

        metric_expr = instantiated_expr or normalized_expr
        try:
            if metric_expr:
                size_raw = srbench_model_size(metric_expr, simplify=False)
                size_simplified = srbench_model_size(metric_expr, simplify=True)
            if metric_expr and gt_expr:
                symbolic_solution = srbench_symbolic_solution(metric_expr, gt_expr)
                ted = normalized_tree_edit_distance(metric_expr, gt_expr)
        except Exception as exc:
            metric_error = f"{exc.__class__.__name__}: {exc}"

        run_rows.append(
            {
                "result_path": str(result_path.relative_to(submit_dir)),
                "seed": seed,
                "tool": tool,
                "dataset": dataset,
                "status": status,
                "seconds": payload.get("seconds"),
                "equation_count": payload.get("equation_count"),
                "gt_available": bool(gt_expr),
                "gt_expression": gt_expr,
                "gt_feature_names": ",".join(gt_feature_names),
                "gt_error": gt_error,
                "param_values_found": bool(param_values),
                "param_count": len(param_values) if param_values else None,
                "param_source": param_source,
                "param_error": param_error,
                "artifact_ok": artifact is not None,
                "artifact_error": artifact_error,
                "normalized_expression": normalized_expr,
                "instantiated_expression": instantiated_expr,
                "ast_node_count": ast_node_count,
                "tree_depth": tree_depth,
                "operator_set": operator_set,
                "model_size_raw": size_raw.get("size"),
                "operators_raw": size_raw.get("operators"),
                "features_raw": size_raw.get("features"),
                "constants_raw": size_raw.get("constants"),
                "model_size_simplified": size_simplified.get("size"),
                "operators_simplified": size_simplified.get("operators"),
                "features_simplified": size_simplified.get("features"),
                "constants_simplified": size_simplified.get("constants"),
                "symbolic_solution": symbolic_solution.get("is_symbolic_solution"),
                "symbolic_solution_relation": symbolic_solution.get("relation"),
                "tree_edit_distance": ted.get("tree_edit_distance"),
                "true_tree_size": ted.get("true_tree_size"),
                "ned": ted.get("ned"),
                "metric_error": metric_error,
            }
        )

    run_df = pd.DataFrame(run_rows)
    run_csv = submit_dir / "interpretability_metrics_runs.csv"
    run_df.to_csv(run_csv, index=False)

    summary_rows: list[dict[str, Any]] = []
    if not run_df.empty:
        for (tool, dataset), group in run_df.groupby(["tool", "dataset"], dropna=False):
            summary_rows.append(
                {
                    "tool": tool,
                    "dataset": dataset,
                    "runs": int(len(group)),
                    "gt_runs": int(group["gt_available"].fillna(False).sum()),
                    "artifact_ok_runs": int(group["artifact_ok"].fillna(False).sum()),
                    "param_found_runs": int(group["param_values_found"].fillna(False).sum()),
                    "symbolic_solution_rate": _safe_bool_mean(group["symbolic_solution"]),
                    "ned_mean": _safe_float_mean(group["ned"]),
                    "ned_std": _safe_float_std(group["ned"]),
                    "tree_edit_distance_mean": _safe_float_mean(group["tree_edit_distance"]),
                    "tree_edit_distance_std": _safe_float_std(group["tree_edit_distance"]),
                    "model_size_raw_mean": _safe_float_mean(group["model_size_raw"]),
                    "model_size_raw_std": _safe_float_std(group["model_size_raw"]),
                    "model_size_simplified_mean": _safe_float_mean(group["model_size_simplified"]),
                    "model_size_simplified_std": _safe_float_std(group["model_size_simplified"]),
                    "tree_depth_mean": _safe_float_mean(group["tree_depth"]),
                    "tree_depth_std": _safe_float_std(group["tree_depth"]),
                    "ast_node_count_mean": _safe_float_mean(group["ast_node_count"]),
                    "ast_node_count_std": _safe_float_std(group["ast_node_count"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "tool"])
    summary_csv = submit_dir / "interpretability_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"已写出逐次明细: {run_csv}")
    print(f"已写出聚合摘要: {summary_csv}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
