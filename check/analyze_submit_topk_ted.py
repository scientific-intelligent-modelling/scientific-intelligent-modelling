from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scientific_intelligent_modelling.benchmarks.metrics import (  # noqa: E402
    normalized_tree_edit_distance,
    srbench_model_size,
    srbench_symbolic_solution,
)
from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_build_canonical_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 submit 结果的 top10 公式 TED")
    parser.add_argument(
        "--submit-dir",
        required=True,
        help="submit 结果目录，例如 experiments/submit/three_tools_3seeds_2h_blt35_20260413_013856",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=8.0,
        help="每条候选公式计算 TED 的超时时间（秒）",
    )
    return parser.parse_args()


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


def _custom_ground_truth() -> dict[str, dict[str, str]]:
    return {
        "oscillator1": {
            "expression": "0.8*sin(x0) - 0.5*x1**3 - 0.2*x0**3 - 0.5*x0*x1 - x0*cos(x0)",
            "note": "x0=position, x1=velocity",
        },
        "oscillator2": {
            "expression": "0.3*sin(x0) - 0.5*x2**3 - 1.0*x1*x2 - 5.0*x1*exp(0.5*x1)",
            "note": "x0=time, x1=position, x2=velocity",
        },
        "stressstrain": {
            # 说明：
            # 1) 当前结果只有两个自变量，这里按 x0=strain(or strain-rate proxy), x1=temperature 做结构比较。
            # 2) 常数值在当前 TED 中会被折叠为 Const，因此这里使用非零占位数值以保留结构、避免 simplify 把结构直接消掉。
            "expression": "(1.3 + 0.7*exp(x0**2)) * (1 - ((x1 - 0.1) / (1.1 - 0.1))**2)",
            "note": "assume x0=strain_rate proxy, x1=temperature; constants are nonzero placeholders",
        },
    }


def _load_candidate_rows(result_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    tool = str(payload.get("tool") or "").strip()
    dataset = str(payload.get("dataset") or "").strip()
    seed = payload.get("seed")
    equation = str(payload.get("equation") or "").strip()
    feature_names = payload.get("feature_names") or []
    expected_n_features = len(feature_names) if isinstance(feature_names, list) and feature_names else None

    experiments_dir = result_path.parent / "experiments"
    top_files = sorted(experiments_dir.glob("**/samples/top*.json"))
    rows: list[dict[str, Any]] = []

    if top_files:
        for top_file in top_files:
            data = json.loads(top_file.read_text(encoding="utf-8"))
            function = str(data.get("function") or data.get("equation") or "").strip()
            params = data.get("params")
            if isinstance(params, list):
                try:
                    params = [float(x) for x in params]
                except Exception:
                    params = None
            else:
                params = None

            rows.append(
                {
                    "result_path": str(result_path),
                    "tool": tool,
                    "dataset": dataset,
                    "seed": seed,
                    "source": "top10",
                    "candidate_file": str(top_file),
                    "candidate_rank": top_file.name.split("_", 1)[0],
                    "sample_order": data.get("sample_order"),
                    "raw_equation": function,
                    "parameter_values": params,
                    "expected_n_features": expected_n_features,
                }
            )
        return rows

    rows.append(
        {
            "result_path": str(result_path),
            "tool": tool,
            "dataset": dataset,
            "seed": seed,
            "source": "best_only",
            "candidate_file": str(result_path),
            "candidate_rank": "best01",
            "sample_order": None,
            "raw_equation": equation,
            "parameter_values": None,
            "expected_n_features": expected_n_features,
        }
    )
    return rows


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


def _safe_bool_mean(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.astype(float).mean())


def _evaluate_candidate_worker(payload: dict[str, Any], queue: mp.Queue) -> None:
    equation_text = _resolve_optional_param_guards(payload["raw_equation"], payload.get("parameter_values"))
    artifact, artifact_error = safe_build_canonical_artifact(
        tool_name=payload["tool"],
        equation=equation_text,
        expected_n_features=payload.get("expected_n_features"),
        parameter_values=payload.get("parameter_values"),
    )

    normalized_expr = None
    instantiated_expr = None
    model_size = {}
    ted = {}
    symbolic_solution = {}
    metric_error = None
    if artifact is not None:
        normalized_expr = artifact.get("normalized_expression")
        instantiated_expr = artifact.get("instantiated_expression") or normalized_expr
        try:
            if instantiated_expr:
                model_size = srbench_model_size(instantiated_expr, simplify=True)
            if instantiated_expr and payload.get("gt_expression"):
                ted = normalized_tree_edit_distance(instantiated_expr, payload["gt_expression"])
                symbolic_solution = srbench_symbolic_solution(instantiated_expr, payload["gt_expression"])
        except Exception as exc:
            metric_error = f"{exc.__class__.__name__}: {exc}"

    queue.put(
        {
            "artifact_ok": artifact is not None,
            "artifact_error": artifact_error,
            "normalized_expression": normalized_expr,
            "instantiated_expression": instantiated_expr,
            "model_size_simplified": model_size.get("size"),
            "tree_edit_distance": ted.get("tree_edit_distance"),
            "true_tree_size": ted.get("true_tree_size"),
            "ned": ted.get("ned"),
            "symbolic_solution": symbolic_solution.get("is_symbolic_solution"),
            "symbolic_solution_relation": symbolic_solution.get("relation"),
            "metric_error": metric_error,
        }
    )


def _evaluate_candidate_with_timeout(payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_evaluate_candidate_worker, args=(payload, queue))
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "artifact_ok": False,
            "artifact_error": None,
            "normalized_expression": None,
            "instantiated_expression": None,
            "model_size_simplified": None,
            "tree_edit_distance": None,
            "true_tree_size": None,
            "ned": None,
            "symbolic_solution": None,
            "symbolic_solution_relation": None,
            "metric_error": f"TimeoutError: exceeded {timeout_seconds:.1f}s",
        }

    if not queue.empty():
        return queue.get()

    return {
        "artifact_ok": False,
        "artifact_error": "WorkerError: no result returned",
        "normalized_expression": None,
        "instantiated_expression": None,
        "model_size_simplified": None,
        "tree_edit_distance": None,
        "true_tree_size": None,
        "ned": None,
        "symbolic_solution": None,
        "symbolic_solution_relation": None,
        "metric_error": "WorkerError: queue empty",
    }


def main() -> None:
    args = parse_args()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    if not submit_dir.is_dir():
        raise SystemExit(f"submit 目录不存在: {submit_dir}")

    gt_map = _custom_ground_truth()
    candidate_rows: list[dict[str, Any]] = []

    for result_path in sorted(submit_dir.glob("**/result.json")):
        candidate_rows.extend(_load_candidate_rows(result_path))

    evaluated_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        dataset = row["dataset"]
        gt_info = gt_map.get(dataset) or {}
        gt_expr = gt_info.get("expression")
        gt_note = gt_info.get("note")
        metrics = _evaluate_candidate_with_timeout(
            {**row, "gt_expression": gt_expr},
            timeout_seconds=float(args.timeout_seconds),
        )

        evaluated_rows.append(
            {
                **row,
                "gt_expression": gt_expr,
                "gt_note": gt_note,
                **metrics,
            }
        )

    candidates_df = pd.DataFrame(evaluated_rows)
    candidates_df["result_path"] = candidates_df["result_path"].map(lambda x: str(Path(x).relative_to(submit_dir)))
    candidates_df["candidate_file"] = candidates_df["candidate_file"].map(
        lambda x: str(Path(x).relative_to(submit_dir))
    )
    candidates_csv = submit_dir / "topk_ted_candidates.csv"
    candidates_df.to_csv(candidates_csv, index=False)

    run_summary_rows: list[dict[str, Any]] = []
    for (tool, dataset, seed), group in candidates_df.groupby(["tool", "dataset", "seed"], dropna=False):
        valid_ted = pd.to_numeric(group["tree_edit_distance"], errors="coerce")
        valid_ned = pd.to_numeric(group["ned"], errors="coerce")
        best_idx = valid_ted.idxmin() if valid_ted.notna().any() else None
        best_row = group.loc[best_idx] if best_idx is not None else None
        run_summary_rows.append(
            {
                "tool": tool,
                "dataset": dataset,
                "seed": seed,
                "source": ",".join(sorted(set(group["source"].astype(str)))),
                "candidate_count": int(len(group)),
                "artifact_ok_runs": int(group["artifact_ok"].fillna(False).sum()),
                "best_ted": float(valid_ted.min()) if valid_ted.notna().any() else None,
                "mean_ted": _safe_float_mean(group["tree_edit_distance"]),
                "best_ned": float(valid_ned.min()) if valid_ned.notna().any() else None,
                "mean_ned": _safe_float_mean(group["ned"]),
                "symbolic_solution_rate": _safe_bool_mean(group["symbolic_solution"]),
                "best_model_size_simplified": best_row["model_size_simplified"] if best_row is not None else None,
                "best_candidate_rank": best_row["candidate_rank"] if best_row is not None else None,
                "best_candidate_file": best_row["candidate_file"] if best_row is not None else None,
            }
        )
    run_summary_df = pd.DataFrame(run_summary_rows).sort_values(["dataset", "tool", "seed"])
    run_summary_csv = submit_dir / "topk_ted_run_summary.csv"
    run_summary_df.to_csv(run_summary_csv, index=False)

    agg_rows: list[dict[str, Any]] = []
    for (tool, dataset), group in run_summary_df.groupby(["tool", "dataset"], dropna=False):
        agg_rows.append(
            {
                "tool": tool,
                "dataset": dataset,
                "runs": int(len(group)),
                "candidate_count_mean": _safe_float_mean(group["candidate_count"]),
                "best_ted_mean": _safe_float_mean(group["best_ted"]),
                "best_ted_std": _safe_float_std(group["best_ted"]),
                "mean_ted_mean": _safe_float_mean(group["mean_ted"]),
                "mean_ted_std": _safe_float_std(group["mean_ted"]),
                "best_ned_mean": _safe_float_mean(group["best_ned"]),
                "best_ned_std": _safe_float_std(group["best_ned"]),
                "mean_ned_mean": _safe_float_mean(group["mean_ned"]),
                "mean_ned_std": _safe_float_std(group["mean_ned"]),
                "symbolic_solution_rate_mean": _safe_float_mean(group["symbolic_solution_rate"]),
            }
        )
    agg_df = pd.DataFrame(agg_rows).sort_values(["dataset", "tool"])
    agg_csv = submit_dir / "topk_ted_aggregate_summary.csv"
    agg_df.to_csv(agg_csv, index=False)

    print(f"已写出候选明细: {candidates_csv}")
    print(f"已写出单次汇总: {run_summary_csv}")
    print(f"已写出聚合汇总: {agg_csv}")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
