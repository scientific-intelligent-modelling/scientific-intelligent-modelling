import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_archive import write_result_payload
from scientific_intelligent_modelling.benchmarks.result_artifacts import (
    safe_export_canonical_artifact,
)
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def load_dataset(dataset_dir: Path):
    meta = yaml.safe_load((dataset_dir / "metadata.yaml").read_text(encoding="utf-8"))["dataset"]
    target = meta["target"]["name"]

    def split(name: str):
        df = pd.read_csv(dataset_dir / name)
        return df.drop(columns=[target]).values, df[target].values

    X_train, y_train = split("train.csv")
    X_id, y_id = split("id_test.csv")
    X_ood, y_ood = split("ood_test.csv")
    return meta, target, X_train, y_train, X_id, y_id, X_ood, y_ood


def metrics(reg, X, y):
    pred = np.asarray(reg.predict(X)).reshape(-1)
    result = regression_metrics(y, pred, acc_threshold=0.1)
    return {
        "rmse": result["rmse"],
        "r2": result["r2"],
        "nmse": result["nmse"],
        "acc_0_1": result["acc_tau"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, choices=["llmsr", "drsr"])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--api-model", default="blt/gpt-3.5-turbo")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_dir = output_root / args.tool / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, target, X_train, y_train, X_id, y_id, X_ood, y_ood = load_dataset(dataset_dir)
    problem_name = f"{dataset_dir.name}_budget{args.budget}"
    background = meta["description"]
    llm_config_path = output_dir / "llm.config"
    llm_config_path.write_text(
        json.dumps(
            {
                "host": "api.bltcy.ai",
                "api_key": os.environ["BLT_API_KEY"],
                "model": args.api_model,
                "max_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.3,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.tool == "llmsr":
        reg = SymbolicRegressor(
            "llmsr",
            problem_name=problem_name,
            background=background,
            metadata_path=str(dataset_dir / "metadata.yaml"),
            llm_config_path=str(llm_config_path),
            exp_path=str(output_dir / "experiments"),
            exp_name=problem_name,
            niterations=max(1, args.budget // 2),
            samples_per_iteration=2,
            max_params=12,
            seed=1314,
        )
    else:
        reg = SymbolicRegressor(
            "drsr",
            problem_name=problem_name,
            background=background,
            metadata_path=str(dataset_dir / "metadata.yaml"),
            llm_config_path=str(llm_config_path),
            exp_path=str(output_dir / "experiments"),
            exp_name=problem_name,
            niterations=max(1, args.budget // 4),
            samples_per_iteration=4,
            evaluate_timeout_seconds=20,
            seed=1314,
        )

    started = time.time()
    reg.fit(X_train, y_train)
    experiment_dir = getattr(reg, "experiment_dir", None)
    canonical_artifact, canonical_artifact_error = safe_export_canonical_artifact(reg)
    result = {
        "tool": args.tool,
        "dataset": dataset_dir.name,
        "budget": args.budget,
        "seconds": time.time() - started,
        "experiment_dir": str(Path(experiment_dir).resolve()) if experiment_dir else None,
        "equation": reg.get_optimal_equation(),
        "canonical_artifact": canonical_artifact,
        "canonical_artifact_error": canonical_artifact_error,
        "id_test": metrics(reg, X_id, y_id),
        "ood_test": metrics(reg, X_ood, y_ood),
    }
    result_path = output_dir / "result.json"
    write_result_payload(result, primary_path=result_path, experiment_dir=experiment_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
