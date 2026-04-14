"""统一 benchmark/result runner。"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_archive import write_result_payload
from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_export_canonical_artifact
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


_HIDDEN_PARAM_KEYS = {"api_key", "apikey", "token", "password", "secret"}


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


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if str(key).lower() in _HIDDEN_PARAM_KEYS:
            continue
        sanitized[key] = value
    return sanitized


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
