"""统一 benchmark/result runner。"""

from __future__ import annotations

import ast
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import sympy as sp

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_archive import write_result_payload
from scientific_intelligent_modelling.benchmarks.result_artifacts import (
    safe_build_canonical_artifact,
    safe_export_canonical_artifact,
)
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


_HIDDEN_PARAM_KEYS = {"api_key", "apikey", "token", "password", "secret"}
_PROGRESS_DIRNAME = "progress"
_SNAPSHOT_CAPABLE_TOOLS = {"llmsr", "drsr", "pysr", "dso", "pyoperon", "gplearn", "e2esr"}


@dataclass
class DatasetSplit:
    name: str
    X: np.ndarray
    y: np.ndarray
    rows: int


@dataclass
class LoadedDataset:
    dataset_dir: Path
    dataset_name: str
    metadata: dict[str, Any]
    target_name: str
    feature_names: list[str]
    feature_descriptions: list[str | None]
    target_description: str | None
    train: DatasetSplit
    valid: DatasetSplit | None
    id_test: DatasetSplit | None
    ood_test: DatasetSplit | None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"metadata.yaml 格式非法: {path}")
    return data


def _load_split(dataset_dir: Path, filename: str, target_name: str) -> DatasetSplit | None:
    split_path = dataset_dir / filename
    if not split_path.exists():
        return None
    df = pd.read_csv(split_path)
    if df.empty:
        return DatasetSplit(
            name=filename.removesuffix(".csv"),
            X=np.empty((0, 0), dtype=float),
            y=np.empty((0,), dtype=float),
            rows=0,
        )
    if target_name not in df.columns:
        raise ValueError(f"{split_path} 中缺少目标列 {target_name}")
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    return DatasetSplit(
        name=filename.removesuffix(".csv"),
        X=np.asarray(X),
        y=np.asarray(y).reshape(-1),
        rows=int(len(df)),
    )


def _build_background(dataset_meta: dict[str, Any], feature_names: list[str]) -> str:
    desc = str(dataset_meta.get("description") or "").strip()
    if desc:
        return desc

    features = dataset_meta.get("features") or []
    target = dataset_meta.get("target") or {}
    target_desc = str(target.get("description") or target.get("name") or "target").strip()
    feature_descs = []
    for idx, item in enumerate(features):
        if isinstance(item, dict):
            feature_descs.append(item.get("description") or item.get("name") or feature_names[idx])
        else:
            feature_descs.append(feature_names[idx])
    if not feature_descs:
        feature_descs = feature_names
    feature_text = ", ".join(str(x) for x in feature_descs if x)
    return f"Find the mathematical function skeleton that represents {target_desc}, given data on {feature_text}."


def load_canonical_dataset(dataset_dir: str | Path) -> LoadedDataset:
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")

    meta_root = _load_yaml(dataset_path / "metadata.yaml")
    dataset_meta = meta_root.get("dataset", meta_root)
    if not isinstance(dataset_meta, dict):
        raise ValueError(f"metadata.yaml 中 dataset 字段格式非法: {dataset_path}")

    target_meta = dataset_meta.get("target") or {}
    target_name = target_meta.get("name")
    if not isinstance(target_name, str) or not target_name.strip():
        train_path = dataset_path / "train.csv"
        if not train_path.exists():
            raise ValueError(f"metadata.yaml 缺少 target.name，且不存在 train.csv: {dataset_path}")
        train_df = pd.read_csv(train_path, nrows=1)
        if train_df.empty:
            raise ValueError(f"无法从空 train.csv 推断目标列: {train_path}")
        target_name = str(train_df.columns[-1])

    train = _load_split(dataset_path, "train.csv", target_name)
    if train is None:
        raise FileNotFoundError(f"缺少 train.csv: {dataset_path}")

    feature_names = list(pd.read_csv(dataset_path / "train.csv", nrows=1).drop(columns=[target_name]).columns)
    features_meta = dataset_meta.get("features") or []
    feature_descriptions: list[str | None] = []
    for idx, feature_name in enumerate(feature_names):
        item = features_meta[idx] if idx < len(features_meta) else {}
        if isinstance(item, dict):
            feature_descriptions.append(item.get("description") or item.get("name") or feature_name)
        else:
            feature_descriptions.append(feature_name)

    target_description = None
    if isinstance(target_meta, dict):
        target_description = target_meta.get("description") or target_meta.get("name")

    return LoadedDataset(
        dataset_dir=dataset_path,
        dataset_name=dataset_path.name,
        metadata=dataset_meta,
        target_name=target_name,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
        train=train,
        valid=_load_split(dataset_path, "valid.csv", target_name),
        id_test=_load_split(dataset_path, "id_test.csv", target_name),
        ood_test=_load_split(dataset_path, "ood_test.csv", target_name),
    )


def _evaluate_split(regressor: SymbolicRegressor, split: DatasetSplit | None) -> dict[str, float | None] | None:
    if split is None or split.rows == 0:
        return None
    pred = np.asarray(regressor.predict(split.X)).reshape(-1)
    metrics = regression_metrics(split.y, pred, acc_threshold=0.1)
    return {
        "rmse": _safe_float(metrics["rmse"]),
        "r2": _safe_float(metrics["r2"]),
        "nmse": _safe_float(metrics["nmse"]),
        "acc_0_1": _safe_float(metrics["acc_tau"]),
    }


def _evaluate_prediction(split: DatasetSplit | None, pred: np.ndarray | None) -> dict[str, float | None] | None:
    if split is None or split.rows == 0 or pred is None:
        return None
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    metrics = regression_metrics(split.y, pred_arr, acc_threshold=0.1)
    return {
        "rmse": _safe_float(metrics["rmse"]),
        "r2": _safe_float(metrics["r2"]),
        "nmse": _safe_float(metrics["nmse"]),
        "acc_0_1": _safe_float(metrics["acc_tau"]),
    }


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if str(key).lower() in _HIDDEN_PARAM_KEYS:
            continue
        sanitized[key] = value
    return sanitized


def _sympy_locals() -> dict[str, Any]:
    locals_map: dict[str, Any] = {
        "Abs": sp.Abs,
        "Max": sp.Max,
        "Min": sp.Min,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
    }
    for i in range(256):
        locals_map[f"x{i}"] = sp.Symbol(f"x{i}")
    return locals_map


def _predict_from_canonical_artifact(artifact: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """基于统一工件中的代值表达式做轻量预测。

    这里只服务 runner 的中间最优快照，不依赖具体算法 wrapper 或子进程环境。
    """
    expr_text = (
        artifact.get("instantiated_expression")
        or artifact.get("normalized_expression")
        or artifact.get("return_expression_source")
    )
    if not isinstance(expr_text, str) or not expr_text.strip():
        raise ValueError("canonical_artifact 中缺少可执行表达式")

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("中间快照预测要求二维输入")

    expr = sp.sympify(expr_text, locals=_sympy_locals())
    free_symbols = sorted(
        list(expr.free_symbols),
        key=lambda sym: int(str(sym)[1:]) if str(sym).startswith("x") and str(sym)[1:].isdigit() else str(sym),
    )

    if not free_symbols:
        value = float(expr)
        return np.full(X_arr.shape[0], value, dtype=float)

    args = []
    for sym in free_symbols:
        name = str(sym)
        if not name.startswith("x") or not name[1:].isdigit():
            raise ValueError(f"表达式包含非标准变量: {name}")
        idx = int(name[1:])
        if idx >= X_arr.shape[1]:
            raise ValueError(f"表达式变量索引越界: {name}, 输入维度={X_arr.shape[1]}")
        args.append(X_arr[:, idx])

    fn = sp.lambdify(free_symbols, expr, modules="numpy")
    pred = fn(*args)
    pred_arr = np.asarray(pred, dtype=float)
    if pred_arr.ndim == 0:
        pred_arr = np.full(X_arr.shape[0], float(pred_arr), dtype=float)
    else:
        pred_arr = np.broadcast_to(pred_arr, (X_arr.shape[0],)).astype(float)
    return pred_arr.reshape(-1)


def _is_equation_function_text(func_source: Any) -> bool:
    if not isinstance(func_source, str) or not func_source.strip():
        return False
    try:
        tree = ast.parse(func_source)
    except Exception:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "equation":
            body = list(node.body)
            while (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None
    return False


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_llmsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    samples_dir = Path(experiment_dir) / "samples"
    if not samples_dir.is_dir():
        return None
    candidates = sorted(samples_dir.glob("top01_*.json")) or sorted(samples_dir.glob("top*.json"))
    best_key = None
    best_item = None
    for path in candidates:
        item = _read_json_file(path)
        if not item:
            continue
        func = item.get("function")
        if not _is_equation_function_text(func):
            continue
        key_val = None
        nmse = item.get("nmse")
        mse = item.get("mse")
        score = item.get("score")
        if isinstance(nmse, (int, float)):
            key_val = float(nmse)
        elif isinstance(mse, (int, float)):
            key_val = float(mse)
        elif isinstance(score, (int, float)):
            key_val = -float(score)
        if key_val is None:
            continue
        if best_key is None or key_val < best_key:
            best_key = key_val
            best_item = item
    return best_item


def _extract_drsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    candidate_paths = []
    candidate_paths.extend(sorted((base_dir / "samples").glob("top*.json")))
    candidate_paths.extend(sorted((base_dir / "best_history").glob("best_sample_*.json")))
    candidate_paths.extend(sorted((base_dir / "samples").glob("samples_*.json")))
    best_key = None
    best_item = None
    for path in candidate_paths:
        item = _read_json_file(path)
        if not item:
            continue
        func = item.get("function")
        if not _is_equation_function_text(func):
            continue
        score = item.get("score")
        if not isinstance(score, (int, float)):
            continue
        key_val = -float(score)
        if best_key is None or key_val < best_key:
            best_key = key_val
            best_item = item
    return best_item


def _extract_pysr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    candidate_paths = [base_dir / "hall_of_fame.csv", base_dir / "hall_of_fame.csv.bak"]
    best_loss = None
    best_item = None
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "Equation" not in df.columns or "Loss" not in df.columns:
            continue
        for _, row in df.iterrows():
            equation = row.get("Equation")
            loss = row.get("Loss")
            if not isinstance(equation, str) or not equation.strip():
                continue
            try:
                loss_val = float(loss)
            except Exception:
                continue
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_item = {
                    "equation": equation,
                    "loss": loss_val,
                    "complexity": row.get("Complexity"),
                }
    return best_item


def _extract_dso_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    candidate_paths = sorted(base_dir.glob("*_hof.csv"))
    best_key = None
    best_item = None
    for path in candidate_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        expr_col = None
        for name in ("expression", "Equation", "equation", "sympy_format"):
            if name in df.columns:
                expr_col = name
                break
        score_col = None
        for name in ("r", "score", "reward"):
            if name in df.columns:
                score_col = name
                break
        if expr_col is None or score_col is None:
            continue
        for _, row in df.iterrows():
            equation = row.get(expr_col)
            score = row.get(score_col)
            if not isinstance(equation, str) or not equation.strip():
                continue
            try:
                score_val = float(score)
            except Exception:
                continue
            # DSO 的 reward 越大越好，这里统一转成“越小越优”的排序键。
            key_val = -score_val
            if best_key is None or key_val < best_key:
                best_key = key_val
                best_item = {
                    "equation": equation,
                    "score": score_val,
                    "complexity": row.get("complexity"),
                }
    return best_item


def _extract_pyoperon_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".pyoperon_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_gplearn_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".gplearn_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_e2esr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".e2esr_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_periodic_candidate(tool_name: str, experiment_dir: str | Path) -> dict[str, Any] | None:
    tool = str(tool_name).strip().lower()
    if tool == "llmsr":
        return _extract_llmsr_periodic_candidate(experiment_dir)
    if tool == "drsr":
        return _extract_drsr_periodic_candidate(experiment_dir)
    if tool == "pysr":
        return _extract_pysr_periodic_candidate(experiment_dir)
    if tool == "dso":
        return _extract_dso_periodic_candidate(experiment_dir)
    if tool == "pyoperon":
        return _extract_pyoperon_periodic_candidate(experiment_dir)
    if tool == "gplearn":
        return _extract_gplearn_periodic_candidate(experiment_dir)
    if tool == "e2esr":
        return _extract_e2esr_periodic_candidate(experiment_dir)
    return None


def _progress_snapshot_filename(payload: dict[str, Any]) -> str:
    elapsed_seconds = payload.get("elapsed_seconds")
    try:
        elapsed_minutes = max(0, int(round(float(elapsed_seconds) / 60.0)))
    except Exception:
        elapsed_minutes = 0
    return f"minute_{elapsed_minutes:04d}.json"


def _write_progress_payload(
    payload: dict[str, Any],
    *,
    primary_dir: str | Path,
    experiment_dir: str | Path | None = None,
) -> list[Path]:
    filename = _progress_snapshot_filename(payload)
    paths: list[Path] = [Path(primary_dir).resolve() / filename]
    if experiment_dir:
        paths.append(Path(experiment_dir).resolve() / _PROGRESS_DIRNAME / filename)

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for path in unique_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return unique_paths


def _resolve_progress_snapshot_interval_seconds(tool_name: str, params: dict[str, Any]) -> int | None:
    raw = params.pop("progress_snapshot_interval_seconds", None)
    if raw is None:
        if tool_name in _SNAPSHOT_CAPABLE_TOOLS:
            return 600
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _build_periodic_snapshot_payload(
    *,
    tool_name: str,
    dataset: LoadedDataset,
    params: dict[str, Any],
    seed: int,
    started_at: float,
    experiment_dir: str | Path,
    checkpoint_index: int,
) -> dict[str, Any] | None:
    candidate = _extract_periodic_candidate(tool_name, experiment_dir)
    if not candidate:
        return None

    parameter_values = candidate.get("params") if isinstance(candidate.get("params"), list) else None
    canonical_artifact, canonical_artifact_error = safe_build_canonical_artifact(
        tool_name=tool_name,
        equation=candidate.get("function") if "function" in candidate else candidate.get("equation"),
        expected_n_features=len(dataset.feature_names),
        parameter_values=parameter_values,
    )

    valid_metrics = None
    id_metrics = None
    ood_metrics = None
    error = None
    if canonical_artifact is not None:
        try:
            valid_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.valid.X) if dataset.valid else None
            id_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.id_test.X) if dataset.id_test else None
            ood_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.ood_test.X) if dataset.ood_test else None
            valid_metrics = _evaluate_prediction(dataset.valid, valid_pred)
            id_metrics = _evaluate_prediction(dataset.id_test, id_pred)
            ood_metrics = _evaluate_prediction(dataset.ood_test, ood_pred)
        except Exception as exc:
            error = repr(exc)
    else:
        error = canonical_artifact_error

    payload = build_result_payload(
        tool_name=tool_name,
        dataset=dataset,
        params=params,
        seed=seed,
        started_at=started_at,
        status="ok" if error is None else "error",
        error=error,
        equation=str(candidate.get("function") or candidate.get("equation") or ""),
        equation_count=1,
        canonical_artifact=canonical_artifact,
        canonical_artifact_error=canonical_artifact_error,
        valid_metrics=valid_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        experiment_dir=str(experiment_dir),
    )
    payload["record_type"] = "periodic_best"
    payload["checkpoint_index"] = int(checkpoint_index)
    payload["elapsed_seconds"] = round(time.time() - started_at, 3)
    payload["elapsed_minutes"] = max(0, int(round(payload["elapsed_seconds"] / 60.0)))
    payload["source_iteration"] = candidate.get("iteration")
    payload["source_sample_order"] = candidate.get("sample_order")
    payload["source_score"] = candidate.get("score")
    payload["source_loss"] = candidate.get("loss")
    payload["source_complexity"] = candidate.get("complexity")
    return payload


def _periodic_snapshot_loop(
    *,
    stop_event: threading.Event,
    interval_seconds: int,
    tool_name: str,
    dataset: LoadedDataset,
    params: dict[str, Any],
    seed: int,
    started_at: float,
    output_dir: Path,
    experiment_dir: str | Path,
) -> None:
    checkpoint_index = 0
    while not stop_event.wait(interval_seconds):
        checkpoint_index += 1
        payload = _build_periodic_snapshot_payload(
            tool_name=tool_name,
            dataset=dataset,
            params=params,
            seed=seed,
            started_at=started_at,
            experiment_dir=experiment_dir,
            checkpoint_index=checkpoint_index,
        )
        if payload is None:
            continue
        _write_progress_payload(
            payload,
            primary_dir=output_dir / _PROGRESS_DIRNAME,
            experiment_dir=experiment_dir,
        )


def build_runner_params(
    tool_name: str,
    dataset: LoadedDataset,
    output_dir: str | Path,
    *,
    seed: int,
    params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).resolve()
    params = dict(params_override or {})
    # runner 在外层统一传 seed，避免 params_override 中重复注入导致构造器冲突。
    params.pop("seed", None)
    params.setdefault("exp_path", str(output_path / "experiments"))
    params.setdefault("exp_name", f"{dataset.dataset_name}_{tool_name}_seed{seed}")

    if tool_name in {"llmsr", "drsr"}:
        params.setdefault("background", _build_background(dataset.metadata, dataset.feature_names))
        params.setdefault("metadata_path", str(dataset.dataset_dir / "metadata.yaml"))
        params.setdefault("feature_descriptions", dataset.feature_descriptions)
        if dataset.target_description:
            params.setdefault("target_description", dataset.target_description)

    return params


def build_result_payload(
    *,
    tool_name: str,
    dataset: LoadedDataset,
    params: dict[str, Any],
    seed: int,
    started_at: float,
    status: str,
    error: str | None,
    equation: str | None,
    equation_count: int | None,
    canonical_artifact: dict[str, Any] | None,
    canonical_artifact_error: str | None,
    valid_metrics: dict[str, float | None] | None,
    id_metrics: dict[str, float | None] | None,
    ood_metrics: dict[str, float | None] | None,
    experiment_dir: str | None,
) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "dataset": dataset.dataset_name,
        "dataset_dir": str(dataset.dataset_dir),
        "experiment_dir": str(Path(experiment_dir).resolve()) if experiment_dir else None,
        "status": status,
        "error": error,
        "seed": int(seed),
        "feature_names": dataset.feature_names,
        "target_name": dataset.target_name,
        "train_rows": dataset.train.rows,
        "valid_rows": dataset.valid.rows if dataset.valid else 0,
        "id_test_rows": dataset.id_test.rows if dataset.id_test else 0,
        "ood_test_rows": dataset.ood_test.rows if dataset.ood_test else 0,
        "seconds": round(time.time() - started_at, 3),
        "equation": equation,
        "equation_count": equation_count,
        "canonical_artifact": canonical_artifact,
        "canonical_artifact_error": canonical_artifact_error,
        "valid": valid_metrics,
        "id_test": id_metrics,
        "ood_test": ood_metrics,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": _sanitize_params(params),
    }


def run_benchmark_task(
    *,
    tool_name: str,
    dataset_dir: str | Path,
    output_root: str | Path,
    seed: int = 1314,
    params_override: dict[str, Any] | None = None,
) -> Path:
    dataset = load_canonical_dataset(dataset_dir)
    output_dir = Path(output_root).resolve() / tool_name / dataset.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    params = build_runner_params(
        tool_name,
        dataset,
        output_dir,
        seed=seed,
        params_override=params_override,
    )
    progress_snapshot_interval_seconds = _resolve_progress_snapshot_interval_seconds(tool_name, params)

    started_at = time.time()
    status = "ok"
    error = None
    equation = None
    equation_count = None
    canonical_artifact = None
    canonical_artifact_error = None
    valid_metrics = None
    id_metrics = None
    ood_metrics = None
    experiment_dir = None

    reg = SymbolicRegressor(
        tool_name,
        problem_name=dataset.dataset_name,
        seed=seed,
        **params,
    )
    experiment_dir = getattr(reg, "experiment_dir", None)
    snapshot_stop_event: threading.Event | None = None
    snapshot_thread: threading.Thread | None = None

    if progress_snapshot_interval_seconds and experiment_dir and tool_name in _SNAPSHOT_CAPABLE_TOOLS:
        snapshot_stop_event = threading.Event()
        snapshot_thread = threading.Thread(
            target=_periodic_snapshot_loop,
            kwargs={
                "stop_event": snapshot_stop_event,
                "interval_seconds": progress_snapshot_interval_seconds,
                "tool_name": tool_name,
                "dataset": dataset,
                "params": params,
                "seed": seed,
                "started_at": started_at,
                "output_dir": output_dir,
                "experiment_dir": experiment_dir,
            },
            daemon=True,
        )
        snapshot_thread.start()

    try:
        reg.fit(dataset.train.X, dataset.train.y)
        experiment_dir = getattr(reg, "experiment_dir", experiment_dir)
        equation = reg.get_optimal_equation()
        canonical_artifact, canonical_artifact_error = safe_export_canonical_artifact(reg)
        try:
            equations = reg.get_total_equations()
            equation_count = len(equations) if isinstance(equations, list) else None
        except Exception:
            equation_count = None
        valid_metrics = _evaluate_split(reg, dataset.valid)
        id_metrics = _evaluate_split(reg, dataset.id_test)
        ood_metrics = _evaluate_split(reg, dataset.ood_test)
    except Exception as exc:
        status = "error"
        error = repr(exc)
    finally:
        if snapshot_stop_event is not None:
            snapshot_stop_event.set()
        if snapshot_thread is not None:
            snapshot_thread.join(timeout=1.0)

    result = build_result_payload(
        tool_name=tool_name,
        dataset=dataset,
        params=params,
        seed=seed,
        started_at=started_at,
        status=status,
        error=error,
        equation=equation,
        equation_count=equation_count,
        canonical_artifact=canonical_artifact,
        canonical_artifact_error=canonical_artifact_error,
        valid_metrics=valid_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        experiment_dir=experiment_dir,
    )

    result_path = output_dir / "result.json"
    write_result_payload(result, primary_path=result_path, experiment_dir=experiment_dir)
    return result_path
