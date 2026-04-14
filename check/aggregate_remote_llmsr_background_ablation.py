#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_HOST = "iaaccn49"
DEFAULT_REMOTE_ROOT = "/data3/zhangziwen/experiments/1.llm_background_ablation"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "bench_results"
    / "derived"
    / "iaaccn49_llmsr_background_ablation_summary.csv"
)


REMOTE_SCAN_SCRIPT = r"""
from __future__ import annotations

import json
import pathlib
import sys


def load_json(path: pathlib.Path):
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


root = pathlib.Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(f"remote root not found: {root}")

for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    manifest = load_json(exp_dir / "manifest.json") or {}
    meta = load_json(exp_dir / "meta.json") or {}
    progress = load_json(exp_dir / "progress.json") or []
    config = manifest.get("config") or {}

    if isinstance(progress, list) and progress:
        last_progress = progress[-1]
    else:
        last_progress = {}

    row = {
        "remote_dir": str(exp_dir),
        "experiment_id": manifest.get("experiment_id") or exp_dir.name,
        "problem_name": manifest.get("problem_name") or meta.get("problem_name"),
        "algorithm": manifest.get("algorithm"),
        "status": manifest.get("status"),
        "seed": manifest.get("seed"),
        "created_at_local": manifest.get("created_at_local"),
        "created_at_utc": manifest.get("created_at_utc"),
        "train_path": config.get("train_path"),
        "dataset_name": config.get("dataset_name"),
        "prompts_type": config.get("prompts_type"),
        "background": config.get("background") or meta.get("background"),
        "llm_model": meta.get("llm_model"),
        "llm_config_path": config.get("llm_config_path") or meta.get("llm_config_path"),
        "wandb_project": config.get("wandb_project"),
        "wandb_name": config.get("wandb_name"),
        "wandb_group": config.get("wandb_group"),
        "wandb_tags": config.get("wandb_tags"),
        "niterations": config.get("niterations") or meta.get("niterations"),
        "samples_per_iteration": config.get("samples_per_iteration") or meta.get("samples_per_iteration"),
        "max_params": config.get("max_params") or meta.get("max_params"),
        "feature_names": meta.get("feature_names") or [],
        "target_name": meta.get("target_name"),
        "progress_entries": len(progress) if isinstance(progress, list) else 0,
        "final_iteration": last_progress.get("iteration"),
        "best_nmse": last_progress.get("best_nmse"),
        "best_mse": last_progress.get("best_mse"),
        "best_sample_order": last_progress.get("best_sample_order"),
        "llm_tokens": last_progress.get("llm_tokens") or {},
        "llm_time_seconds": last_progress.get("llm_time_seconds"),
    }
    print(json.dumps(row, ensure_ascii=False))
"""


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _split_dataset_name(dataset_name: str | None) -> tuple[str, str]:
    name = (dataset_name or "").strip()
    for prefix in ("llm-srbench_", "srbench1-0_", "srsd_"):
        if name.startswith(prefix):
            return prefix[:-1], name[len(prefix) :]
    if "_" in name:
        family, rest = name.split("_", 1)
        return family, rest
    return name, ""


def _parse_prompt_type(prompt_type: str | None) -> tuple[str, str, str]:
    parts = (prompt_type or "").split("_")
    level = parts[0] if parts else ""
    family = parts[1] if len(parts) >= 2 else ""
    naming = parts[2] if len(parts) >= 3 else ""
    return level, family, naming


def _augment_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset_name = row.get("dataset_name")
    dataset_family, dataset_key = _split_dataset_name(dataset_name)

    prompt_type = row.get("prompts_type") or ""
    prompt_level, prompt_family, prompt_naming = _parse_prompt_type(prompt_type)

    exp_id = row.get("experiment_id") or ""
    sample_setting_match = re.search(r"__(s[^_]+)_llmsr_seed", exp_id)
    prompt_setting = sample_setting_match.group(1) if sample_setting_match else ""

    model_alias_match = re.search(r"__llmsr__([^_]+)__", exp_id)
    model_alias = model_alias_match.group(1) if model_alias_match else ""

    feature_names = row.get("feature_names") or []
    if isinstance(feature_names, list):
        feature_names_joined = ",".join(str(name) for name in feature_names)
        feature_count = len(feature_names)
    else:
        feature_names_joined = str(feature_names)
        feature_count = 0

    wandb_tags = row.get("wandb_tags") or []
    if isinstance(wandb_tags, list):
        wandb_tags_joined = ",".join(str(tag) for tag in wandb_tags)
    else:
        wandb_tags_joined = str(wandb_tags)

    llm_tokens = row.get("llm_tokens") or {}
    prompt_tokens = llm_tokens.get("prompt")
    thinking_tokens = llm_tokens.get("thinking")
    content_tokens = llm_tokens.get("content")
    total_tokens = llm_tokens.get("total")

    background = row.get("background") or ""
    background_preview = _normalize_text(background)[:180]
    background_sha1 = hashlib.sha1(background.encode("utf-8")).hexdigest()[:12] if background else ""

    return {
        "experiment_id": exp_id,
        "remote_dir": row.get("remote_dir"),
        "status": row.get("status"),
        "created_at_local": row.get("created_at_local"),
        "created_at_utc": row.get("created_at_utc"),
        "seed": row.get("seed"),
        "problem_name": row.get("problem_name"),
        "dataset_name": dataset_name,
        "dataset_family": dataset_family,
        "dataset_key": dataset_key,
        "train_path": row.get("train_path"),
        "model_alias": model_alias,
        "llm_model": row.get("llm_model"),
        "llm_config_path": row.get("llm_config_path"),
        "prompts_type": prompt_type,
        "prompt_level": prompt_level,
        "prompt_family": prompt_family,
        "prompt_naming": prompt_naming,
        "prompt_setting": prompt_setting,
        "wandb_project": row.get("wandb_project"),
        "wandb_name": row.get("wandb_name"),
        "wandb_group": row.get("wandb_group"),
        "wandb_tags": wandb_tags_joined,
        "niterations": row.get("niterations"),
        "samples_per_iteration": row.get("samples_per_iteration"),
        "max_params": row.get("max_params"),
        "feature_count": feature_count,
        "feature_names": feature_names_joined,
        "target_name": row.get("target_name"),
        "progress_entries": row.get("progress_entries"),
        "final_iteration": row.get("final_iteration"),
        "best_nmse": row.get("best_nmse"),
        "best_mse": row.get("best_mse"),
        "best_sample_order": row.get("best_sample_order"),
        "llm_prompt_tokens": prompt_tokens,
        "llm_thinking_tokens": thinking_tokens,
        "llm_content_tokens": content_tokens,
        "llm_total_tokens": total_tokens,
        "llm_time_seconds": row.get("llm_time_seconds"),
        "background_sha1": background_sha1,
        "background_preview": background_preview,
    }


def fetch_remote_rows(host: str, remote_root: str, connect_timeout: int) -> list[dict[str, Any]]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        host,
        "python3",
        "-",
        remote_root,
    ]
    proc = subprocess.run(cmd, input=REMOTE_SCAN_SCRIPT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(_augment_row(json.loads(line)))
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "remote_dir",
        "status",
        "created_at_local",
        "created_at_utc",
        "seed",
        "problem_name",
        "dataset_name",
        "dataset_family",
        "dataset_key",
        "train_path",
        "model_alias",
        "llm_model",
        "llm_config_path",
        "prompts_type",
        "prompt_level",
        "prompt_family",
        "prompt_naming",
        "prompt_setting",
        "wandb_project",
        "wandb_name",
        "wandb_group",
        "wandb_tags",
        "niterations",
        "samples_per_iteration",
        "max_params",
        "feature_count",
        "feature_names",
        "target_name",
        "progress_entries",
        "final_iteration",
        "best_nmse",
        "best_mse",
        "best_sample_order",
        "llm_prompt_tokens",
        "llm_thinking_tokens",
        "llm_content_tokens",
        "llm_total_tokens",
        "llm_time_seconds",
        "background_sha1",
        "background_preview",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 iaaccn49 上的 llmsr background ablation 实验为 CSV")
    parser.add_argument("--host", default=DEFAULT_HOST, help="远端主机名")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="远端实验根目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出 CSV 路径")
    parser.add_argument("--connect-timeout", type=int, default=8, help="SSH 连接超时秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    rows = fetch_remote_rows(args.host, args.remote_root, args.connect_timeout)
    write_csv(rows, output_path)
    print(
        json.dumps(
            {
                "host": args.host,
                "remote_root": args.remote_root,
                "output": str(output_path),
                "rows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
