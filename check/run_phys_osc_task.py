import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def load_split(dataset_dir: Path, filename: str, target_name: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = dataset_dir / filename
    df = pd.read_csv(file_path)
    if target_name not in df.columns:
        raise ValueError(f"{file_path} 中缺少目标列 {target_name}")
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    return X, y


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


def evaluate(reg, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(reg.predict(X)).reshape(-1)
    residual = pred - y
    mse = float(np.mean(np.square(residual)))
    rmse = math.sqrt(mse)
    y_mean = float(np.mean(y))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y - y_mean)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denom = float(np.mean(np.square(y))) if len(y) else float("nan")
    nmse = float(mse / denom) if denom and not math.isnan(denom) else float("nan")
    return {
        "rmse": safe_float(rmse),
        "r2": safe_float(r2),
        "nmse": safe_float(nmse),
    }


def build_background(meta: dict) -> str:
    dataset_meta = meta["dataset"]
    desc = (dataset_meta.get("description") or "").strip()
    features = dataset_meta.get("features") or []
    target = dataset_meta.get("target") or {}
    target_desc = (target.get("description") or target.get("name") or "target").strip()
    feature_descs = [f.get("description") or f.get("name") for f in features]
    feature_text = ", ".join(feature_descs)
    if desc:
        return desc
    return f"Find the mathematical function skeleton that represents {target_desc}, given data on {feature_text}."


def build_llm_config(output_dir: Path, model: str) -> Path:
    api_key = os.getenv("BLT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 BLT_API_KEY，无法运行 llmsr")

    llm_config = {
        "host": "api.bltcy.ai",
        "api_key": api_key,
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.3,
    }
    config_path = output_dir / "llm.config"
    config_path.write_text(json.dumps(llm_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def build_regressor(tool: str, dataset_dir: Path, output_dir: Path, meta: dict, api_model: str, params_override: Dict, seed: int):
    dataset_meta = meta["dataset"]
    problem_name = dataset_dir.name
    background = build_background(meta)
    common = {
        "problem_name": problem_name,
        "seed": seed,
    }

    if tool == "pysr":
        params = {
            "exp_path": str(output_dir / "experiments"),
            "niterations": 40,
            "population_size": 33,
            "populations": 8,
            "maxsize": 25,
            "progress": False,
            "verbosity": 0,
            "procs": 1,
            "random_state": seed,
        }
        params.update(params_override)
        return SymbolicRegressor(tool, **params, **common)

    if tool == "llmsr":
        llm_config_path = build_llm_config(output_dir, api_model)
        params = {
            "exp_path": str(output_dir / "experiments"),
            "exp_name": f"{problem_name}_{tool}",
            "background": background,
            "metadata_path": str(dataset_dir / "metadata.yaml"),
            "llm_config_path": str(llm_config_path),
            "niterations": 30,
            "samples_per_iteration": 2,
            "max_params": 12,
        }
        params.update(params_override)
        return SymbolicRegressor(tool, **params, **common)

    if tool == "drsr":
        llm_config_path = build_llm_config(output_dir, api_model)
        params = {
            "workdir": str(output_dir / "drsr_workdir"),
            "background": background,
            "metadata_path": str(dataset_dir / "metadata.yaml"),
            "llm_config_path": str(llm_config_path),
            "max_samples": 32,
            "samples_per_prompt": 4,
            "evaluate_timeout_seconds": 20,
            "wall_time_limit_seconds": 900,
        }
        params.update(params_override)
        return SymbolicRegressor(tool, **params, **common)

    raise ValueError(f"不支持的算法: {tool}")


def run_task(tool: str, dataset_dir: Path, output_root: Path, api_model: str, params_override: Dict, seed: int) -> Path:
    with open(dataset_dir / "metadata.yaml", "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    dataset_meta = meta["dataset"]
    target_name = dataset_meta["target"]["name"]
    output_dir = output_root / tool / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(dataset_dir, "train.csv", target_name)
    X_id, y_id = load_split(dataset_dir, "id_test.csv", target_name)
    X_ood, y_ood = load_split(dataset_dir, "ood_test.csv", target_name)

    started_at = time.time()
    status = "ok"
    error = None
    equation = None
    equations = None
    metrics_id = None
    metrics_ood = None

    try:
        reg = build_regressor(tool, dataset_dir, output_dir, meta, api_model, params_override=params_override, seed=seed)
        reg.fit(X_train, y_train)
        equation = reg.get_optimal_equation()
        try:
            equations = reg.get_total_equations()
        except Exception:
            equations = None
        metrics_id = evaluate(reg, X_id, y_id)
        metrics_ood = evaluate(reg, X_ood, y_ood)
    except Exception as exc:
        status = "error"
        error = repr(exc)

    result = {
        "tool": tool,
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir.resolve()),
        "status": status,
        "error": error,
        "background": build_background(meta),
        "feature_names": [f["name"] for f in dataset_meta.get("features") or []],
        "target_name": target_name,
        "train_rows": int(len(y_train)),
        "id_test_rows": int(len(y_id)),
        "ood_test_rows": int(len(y_ood)),
        "seconds": round(time.time() - started_at, 3),
        "equation": equation,
        "equation_count": len(equations) if isinstance(equations, list) else None,
        "id_test": metrics_id,
        "ood_test": metrics_ood,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params_override": params_override,
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, choices=["pysr", "llmsr", "drsr"])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--api-model", default="blt/gpt-3.5-turbo")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--params-json", default="{}")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    params_override = json.loads(args.params_json or "{}")
    if not isinstance(params_override, dict):
        raise ValueError("--params-json 必须解析为 JSON 对象")
    run_task(args.tool, dataset_dir, output_root, args.api_model, params_override=params_override, seed=args.seed)


if __name__ == "__main__":
    main()
