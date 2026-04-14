from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_artifacts import (
    safe_export_canonical_artifact,
)
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


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


def evaluate(reg, X, y):
    pred = np.asarray(reg.predict(X)).reshape(-1)
    metrics = regression_metrics(y, pred, acc_threshold=0.1)
    return {
        "rmse": safe_float(metrics["rmse"]),
        "r2": safe_float(metrics["r2"]),
        "nmse": safe_float(metrics["nmse"]),
        "acc_0_1": safe_float(metrics["acc_tau"]),
    }


def load_stressstrain():
    data_dir = Path(
        "scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/data/stressstrain"
    )
    train_df = pd.read_csv(data_dir / "train.csv")
    id_df = pd.read_csv(data_dir / "test_id.csv")
    ood_df = pd.read_csv(data_dir / "test_ood.csv")
    feature_cols = ["strain", "temp"]
    target_col = "stress"
    return {
        "data_dir": data_dir,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "X_train": train_df[feature_cols].values,
        "y_train": train_df[target_col].values,
        "X_id": id_df[feature_cols].values,
        "y_id": id_df[target_col].values,
        "X_ood": ood_df[feature_cols].values,
        "y_ood": ood_df[target_col].values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--niterations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--populations", type=int, default=32)
    parser.add_argument("--procs", type=int, default=32)
    parser.add_argument("--maxsize", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=1314)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--verbosity", type=int, default=1)
    args = parser.parse_args()

    ds = load_stressstrain()
    output_dir = Path(args.output_root).resolve() / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "exp_path": str(output_dir / "experiments"),
        "niterations": args.niterations,
        "population_size": args.population_size,
        "populations": args.populations,
        "maxsize": args.maxsize,
        "procs": args.procs,
        "progress": args.progress,
        "verbosity": args.verbosity,
        "random_state": args.random_state,
    }

    started = time.time()
    status = "ok"
    error = None
    equation = None
    equation_count = None
    canonical_artifact = None
    canonical_artifact_error = None
    id_metrics = None
    ood_metrics = None

    try:
        reg = SymbolicRegressor("pysr", problem_name="stressstrain", seed=args.random_state, **params)
        reg.fit(ds["X_train"], ds["y_train"])
        equation = reg.get_optimal_equation()
        canonical_artifact, canonical_artifact_error = safe_export_canonical_artifact(reg)
        try:
            equations = reg.get_total_equations()
            equation_count = len(equations) if isinstance(equations, list) else None
        except Exception:
            equation_count = None
        id_metrics = evaluate(reg, ds["X_id"], ds["y_id"])
        ood_metrics = evaluate(reg, ds["X_ood"], ds["y_ood"])
    except Exception as exc:
        status = "error"
        error = repr(exc)

    result = {
        "tool": "pysr",
        "dataset": "stressstrain",
        "label": args.label,
        "dataset_dir": str(ds["data_dir"].resolve()),
        "status": status,
        "error": error,
        "seconds": round(time.time() - started, 3),
        "equation": equation,
        "equation_count": equation_count,
        "canonical_artifact": canonical_artifact,
        "canonical_artifact_error": canonical_artifact_error,
        "id_test": id_metrics,
        "ood_test": ood_metrics,
        "feature_names": ds["feature_cols"],
        "target_name": ds["target_col"],
        "params": params,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
