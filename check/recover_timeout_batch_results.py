from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_artifacts import (
    safe_build_canonical_artifact,
)

try:
    from scipy.optimize import minimize

    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


def parse_args():
    parser = argparse.ArgumentParser(description="从已有实验目录恢复超时批次的最终结果")
    parser.add_argument("--batch-dir", required=True, help="批次目录，内部包含 seed*/tool/dataset/result.json")
    parser.add_argument("--summary-name", default="rescued_summary.json", help="恢复摘要文件名")
    parser.add_argument("--seed", type=int, action="append", help="只恢复指定 seed，可重复传参")
    parser.add_argument("--tool", action="append", choices=["pysr", "llmsr", "drsr"], help="只恢复指定算法，可重复传参")
    parser.add_argument("--dataset", action="append", help="只恢复指定数据集，可重复传参")
    parser.add_argument("--overwrite", action="store_true", help="回写覆盖 result.json；否则只生成 result.recovered.json")
    parser.add_argument("--force", action="store_true", help="即使原 result.json 不是 timeout error 也强制恢复")
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = regression_metrics(y_true, y_pred, acc_threshold=0.1)
    return {
        "rmse": safe_float(metrics["rmse"]),
        "r2": safe_float(metrics["r2"]),
        "nmse": safe_float(metrics["nmse"]),
        "acc_0_1": safe_float(metrics["acc_tau"]),
    }


def should_process(result: Dict[str, Any], args) -> bool:
    seed = result.get("seed")
    tool = result.get("tool")
    dataset = result.get("dataset")
    if args.seed and seed not in set(args.seed):
        return False
    if args.tool and tool not in set(args.tool):
        return False
    if args.dataset and dataset not in set(args.dataset):
        return False
    if args.force:
        return True
    status = result.get("status")
    error = str(result.get("error") or "")
    return status == "error" and "timeout" in error.lower()


def load_split(dataset_dir: Path, target_name: str, split_name: str):
    df = pd.read_csv(dataset_dir / split_name)
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    return X, y


def resolve_experiment_dir(result: Dict[str, Any]) -> Path:
    params = dict(result.get("params") or {})
    exp_path = params.get("exp_path")
    exp_name = params.get("exp_name")
    if isinstance(exp_path, str) and exp_path.strip() and isinstance(exp_name, str) and exp_name.strip():
        candidate = Path(exp_path) / exp_name
        if candidate.is_dir():
            return candidate

    result_path = Path(result["_result_path"])
    fallback_root = result_path.parent / "experiments"
    if isinstance(exp_name, str) and exp_name.strip():
        candidate = fallback_root / exp_name
        if candidate.is_dir():
            return candidate

    if fallback_root.is_dir():
        children = sorted(p for p in fallback_root.iterdir() if p.is_dir())
        if len(children) == 1:
            return children[0]

    raise FileNotFoundError(
        f"无法根据 result.json 定位实验目录: exp_path={exp_path!r}, exp_name={exp_name!r}, result={result_path}"
    )


def _extract_pysr_equations(equations_obj) -> List[str]:
    if equations_obj is None:
        return []
    if hasattr(equations_obj, "columns"):
        for col in ("sympy_format", "equation", "Equation", "expr", "expression"):
            if col in equations_obj.columns:
                return [str(item) for item in equations_obj[col].dropna().tolist()]
        try:
            return [str(row) for _, row in equations_obj.iterrows()]
        except Exception:
            return [str(equations_obj)]
    if isinstance(equations_obj, list):
        return [str(item) for item in equations_obj]
    return [str(equations_obj)]


def recover_pysr(exp_dir: Path, X_id: np.ndarray, X_ood: np.ndarray):
    from pysr import PySRRegressor as CorePySR

    model = CorePySR.from_file(run_directory=str(exp_dir))
    equations_obj = getattr(model, "equations_", None)
    equations = _extract_pysr_equations(equations_obj)
    best_fn = None
    best_equation = None
    best_loss = None
    if hasattr(equations_obj, "iterrows"):
        for _, row in equations_obj.iterrows():
            fn = row.get("lambda_format")
            if not callable(fn):
                continue
            try:
                id_pred = np.asarray(fn(X_id)).reshape(-1)
                ood_pred = np.asarray(fn(X_ood)).reshape(-1)
                if not (np.all(np.isfinite(id_pred)) and np.all(np.isfinite(ood_pred))):
                    continue
            except Exception:
                continue
            cur_loss = safe_float(row.get("loss"))
            if cur_loss is None:
                cur_loss = float("inf")
            if best_loss is None or cur_loss < best_loss:
                best_loss = cur_loss
                best_fn = fn
                best_equation = row.get("sympy_format") or row.get("equation") or row.get("Equation")
    if best_fn is not None:
        id_pred = np.asarray(best_fn(X_id)).reshape(-1)
        ood_pred = np.asarray(best_fn(X_ood)).reshape(-1)
        equation = str(best_equation)
    else:
        id_pred = np.asarray(model.predict(X_id)).reshape(-1)
        ood_pred = np.asarray(model.predict(X_ood)).reshape(-1)
        equation = str(model.sympy())
    return {
        "equation": equation,
        "equation_count": len(equations) or None,
        "id_pred": id_pred,
        "ood_pred": ood_pred,
    }


def _wrap_function_body(body: str, feature_names: List[str], include_params: bool) -> str:
    args = list(feature_names)
    if include_params:
        args.append("params")
    body = str(body or "").strip("\n")
    indented = "\n".join(("    " + line) if line.strip() else line for line in body.splitlines())
    return f"def equation({', '.join(args)}):\n{indented}\n"


def compile_equation_function(source: str, feature_names: List[str], include_params: bool):
    src = str(source or "").strip()
    if not src:
        raise ValueError("空公式源码")
    if not src.lstrip().startswith("def "):
        src = _wrap_function_body(src, feature_names, include_params=include_params)
    ns = {"np": np, "math": math}
    exec(src, ns)
    fn = ns.get("equation")
    if not callable(fn):
        raise ValueError("未找到 equation 可调用对象")
    return fn


def predict_with_function(func, X: np.ndarray, params: Optional[List[float]] = None) -> np.ndarray:
    cols = [X[:, i] for i in range(X.shape[1])]
    if params is not None:
        return np.asarray(func(*cols, np.asarray(params, dtype=float))).reshape(-1)
    try:
        return np.asarray(func(*cols)).reshape(-1)
    except TypeError:
        return np.asarray(func(*cols, np.ones(10, dtype=float))).reshape(-1)


def fit_params_if_needed(func, X_train: np.ndarray, y_train: np.ndarray, n_params: int = 10) -> Optional[List[float]]:
    if not _SCIPY_OK:
        return None

    def loss_fn(p):
        try:
            pred = predict_with_function(func, X_train, params=list(p))
            return float(np.mean((pred - y_train) ** 2))
        except Exception:
            return 1e6

    best_x = None
    best_loss = None
    rng = np.random.default_rng(0)
    for _ in range(5):
        x0 = rng.uniform(-1.0, 1.0, size=n_params)
        try:
            res = minimize(
                loss_fn,
                x0,
                method="BFGS",
                options={"maxiter": 200, "gtol": 1e-10, "eps": 1e-12, "disp": False},
            )
        except Exception:
            continue
        cur_loss = float(res.fun)
        if best_loss is None or cur_loss < best_loss:
            best_loss = cur_loss
            best_x = np.asarray(res.x, dtype=float)
    return best_x.tolist() if isinstance(best_x, np.ndarray) else None


def _sample_sort_key_for_llmsr(item: Dict[str, Any]):
    nmse = item.get("nmse")
    mse = item.get("mse")
    score = item.get("score")
    if isinstance(nmse, (int, float)):
        return (0, float(nmse))
    if isinstance(mse, (int, float)):
        return (1, float(mse))
    if isinstance(score, (int, float)):
        return (2, -float(score))
    return (3, float("inf"))


def load_best_llmsr_sample(exp_dir: Path) -> Dict[str, Any]:
    samples_dir = exp_dir / "samples"
    paths = sorted(samples_dir.glob("top*.json"))
    if not paths:
        raise FileNotFoundError(f"未找到 llmsr samples 目录: {samples_dir}")
    items = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("function"), str) and data["function"].strip():
            items.append(data)
    if not items:
        raise ValueError(f"llmsr 未找到可用候选: {samples_dir}")
    items.sort(key=_sample_sort_key_for_llmsr)
    return items[0]


def _sample_sort_key_for_drsr(entry: Dict[str, Any]):
    score = entry.get("score")
    if isinstance(score, (int, float)):
        return (0, -float(score))
    nmse = entry.get("nmse")
    if isinstance(nmse, (int, float)):
        return (1, float(nmse))
    mse = entry.get("mse")
    if isinstance(mse, (int, float)):
        return (2, float(mse))
    return (3, float("inf"))


def collect_drsr_entries(exp_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted((exp_dir / "best_history").glob("best_sample_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = data.get("function") or data.get("equation")
        if not isinstance(source, str) or not source.strip():
            continue
        entries.append(
            {
                "function": source,
                "params": data.get("fitted_params") or data.get("params"),
                "score": data.get("score"),
                "nmse": data.get("nmse"),
                "mse": data.get("mse"),
            }
        )

    exp_json = exp_dir / "equation_experiences" / "experiences.json"
    if exp_json.is_file():
        try:
            blob = json.loads(exp_json.read_text(encoding="utf-8"))
        except Exception:
            blob = {}
        for key in ("Good", "None", "Bad"):
            for data in blob.get(key, []):
                source = data.get("function") or data.get("equation")
                if not isinstance(source, str) or not source.strip():
                    continue
                entries.append(
                    {
                        "function": source,
                        "params": data.get("fitted_params") or data.get("params"),
                        "score": data.get("score"),
                        "nmse": data.get("nmse"),
                        "mse": data.get("mse"),
                    }
                )

    if not entries:
        raise FileNotFoundError(f"未找到 drsr 可恢复样本: {exp_dir}")
    entries.sort(key=_sample_sort_key_for_drsr)
    return entries


def recover_llmsr(
    exp_dir: Path,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_id: np.ndarray,
    X_ood: np.ndarray,
):
    sample = load_best_llmsr_sample(exp_dir)
    func = compile_equation_function(sample["function"], feature_names, include_params=True)
    params = sample.get("params")
    if not isinstance(params, list):
        params = fit_params_if_needed(func, X_train, y_train)
    return {
        "equation": str(sample.get("function") or ""),
        "equation_count": len(list((exp_dir / "samples").glob("top*.json"))) or None,
        "parameter_values": params,
        "id_pred": predict_with_function(func, X_id, params=params),
        "ood_pred": predict_with_function(func, X_ood, params=params),
    }


def recover_drsr(
    exp_dir: Path,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_id: np.ndarray,
    X_ood: np.ndarray,
):
    entries = collect_drsr_entries(exp_dir)
    chosen = None
    func = None
    params = None
    for entry in entries:
        try:
            cur_func = compile_equation_function(entry["function"], feature_names, include_params=True)
            cur_params = entry.get("params") if isinstance(entry.get("params"), list) else None
            predict_with_function(cur_func, X_train[: min(8, len(X_train))], params=cur_params)
        except Exception:
            continue
        chosen = entry
        func = cur_func
        params = cur_params
        break

    if func is None or chosen is None:
        raise ValueError(f"drsr 未找到可执行候选: {exp_dir}")
    if params is None:
        params = fit_params_if_needed(func, X_train, y_train)
    return {
        "equation": str(chosen.get("function") or ""),
        "equation_count": len(entries) or None,
        "parameter_values": params,
        "id_pred": predict_with_function(func, X_id, params=params),
        "ood_pred": predict_with_function(func, X_ood, params=params),
    }


def backup_original_result(result_path: Path):
    backup_path = result_path.with_name("result.timeout_backup.json")
    if not backup_path.exists():
        backup_path.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")


def recover_one(result_path: Path, args) -> Dict[str, Any]:
    original = json.loads(result_path.read_text(encoding="utf-8"))
    original["_result_path"] = str(result_path)

    started = time.time()
    dataset_dir = Path(original["dataset_dir"])
    target_name = str(original["target_name"])
    params = dict(original.get("params") or {})
    exp_dir = resolve_experiment_dir(original)

    X_train, y_train = load_split(dataset_dir, target_name, "train.csv")
    X_id, y_id = load_split(dataset_dir, target_name, "test_id.csv")
    X_ood, y_ood = load_split(dataset_dir, target_name, "test_ood.csv")
    feature_names = list(original.get("feature_names") or [])

    tool = original["tool"]
    if tool == "pysr":
        recovered = recover_pysr(exp_dir, X_id, X_ood)
    elif tool == "llmsr":
        recovered = recover_llmsr(exp_dir, feature_names, X_train, y_train, X_id, X_ood)
    elif tool == "drsr":
        recovered = recover_drsr(exp_dir, feature_names, X_train, y_train, X_id, X_ood)
    else:
        raise ValueError(f"不支持的算法: {tool}")

    canonical_artifact, canonical_artifact_error = safe_build_canonical_artifact(
        tool_name=str(tool),
        equation=recovered.get("equation"),
        expected_n_features=len(feature_names) if feature_names else X_train.shape[1],
        parameter_values=recovered.get("parameter_values"),
    )

    updated = dict(original)
    updated.pop("_result_path", None)
    updated.update(
        {
            "status": "ok",
            "error": None,
            "equation": recovered["equation"],
            "equation_count": recovered["equation_count"],
            "canonical_artifact": canonical_artifact,
            "canonical_artifact_error": canonical_artifact_error,
            "id_test": evaluate_predictions(y_id, recovered["id_pred"]),
            "ood_test": evaluate_predictions(y_ood, recovered["ood_pred"]),
            "recovered_from_timeout": True,
            "original_status": original.get("status"),
            "original_error": original.get("error"),
            "recovered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recovery_seconds": round(time.time() - started, 3),
            "recovery_experiment_dir": str(exp_dir),
        }
    )

    output_path = result_path if args.overwrite else result_path.with_name("result.recovered.json")
    if args.overwrite:
        backup_original_result(result_path)
    output_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "seed": updated.get("seed"),
        "tool": updated.get("tool"),
        "dataset": updated.get("dataset"),
        "status": "ok",
        "result_path": str(output_path),
        "equation_count": updated.get("equation_count"),
        "id_rmse": (updated.get("id_test") or {}).get("rmse"),
        "ood_rmse": (updated.get("ood_test") or {}).get("rmse"),
        "recovery_seconds": updated.get("recovery_seconds"),
    }


def iter_result_paths(batch_dir: Path) -> Iterable[Path]:
    return sorted(batch_dir.glob("seed*/**/result.json"))


def main():
    args = parse_args()
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    if not batch_dir.is_dir():
        raise SystemExit(f"批次目录不存在: {batch_dir}")

    summary: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for result_path in iter_result_paths(batch_dir):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(
                {
                    "result_path": str(result_path),
                    "status": "error",
                    "error": f"读取 result.json 失败: {exc!r}",
                }
            )
            continue

        if not should_process(result, args):
            continue

        try:
            recovered = recover_one(result_path, args)
            summary.append(recovered)
            print(
                f"[recovered] seed={recovered['seed']} tool={recovered['tool']} "
                f"dataset={recovered['dataset']} id_rmse={recovered['id_rmse']} ood_rmse={recovered['ood_rmse']}"
            )
        except Exception as exc:
            failures.append(
                {
                    "seed": result.get("seed"),
                    "tool": result.get("tool"),
                    "dataset": result.get("dataset"),
                    "result_path": str(result_path),
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"[failed] seed={result.get('seed')} tool={result.get('tool')} "
                f"dataset={result.get('dataset')} error={exc!r}"
            )

    payload = {
        "batch_dir": str(batch_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overwrite": bool(args.overwrite),
        "recovered_count": len(summary),
        "failure_count": len(failures),
        "recovered": summary,
        "failures": failures,
    }
    summary_path = batch_dir / args.summary_name
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recovered_count": len(summary), "failure_count": len(failures), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
