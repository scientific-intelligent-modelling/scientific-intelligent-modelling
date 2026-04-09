from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from common import load_json, load_manifest, repo_root


def parse_args():
    parser = argparse.ArgumentParser(description="校验符号回归工具 manifest 与接入产物")
    parser.add_argument("--manifest", required=True, help="manifest 路径")
    parser.add_argument("--manifest-only", action="store_true", help="只校验 manifest 结构，不校验生成产物")
    parser.add_argument("--require-source-path", action="store_true", help="要求 vendor_repo_relpath 真实存在")
    parser.add_argument("--runtime-check", action="store_true", help="执行 check/check_<tool>.py")
    parser.add_argument("--python-executable", default=sys.executable, help="运行时校验使用的 Python")
    return parser.parse_args()


def validate_registered_configs(manifest: dict, errors: list[str]) -> None:
    root = repo_root()
    toolbox = load_json(root / manifest["derived_paths"]["toolbox_config"])
    envs = load_json(root / manifest["derived_paths"]["envs_config"])

    tool_entry = (toolbox.get("tool_mapping") or {}).get(manifest["tool_name"])
    expected_tool_entry = {
        "env": manifest["env_name"],
        "regressor": manifest["wrapper_class_name"],
    }
    if tool_entry != expected_tool_entry:
        errors.append(
            f"toolbox_config 注册不匹配: 期望 {expected_tool_entry}，实际 {tool_entry}"
        )

    env_entry = (envs.get("env_list") or {}).get(manifest["env_name"])
    expected_env_entry = {
        "python_version": manifest["env"]["python_version"],
        "conda_packages": manifest["env"]["conda_packages"],
        "pip_packages": manifest["env"]["pip_packages"],
        "channels": manifest["env"]["channels"],
        "post_install_commands": manifest["env"]["post_install_commands"],
        "comments": manifest["env"]["comments"],
    }
    if env_entry != expected_env_entry:
        errors.append(
            f"envs_config 注册不匹配: 期望 {expected_env_entry}，实际 {env_entry}"
        )


def validate_generated_files(manifest: dict, errors: list[str]) -> None:
    root = repo_root()
    wrapper_file = root / manifest["derived_paths"]["wrapper_file"]
    check_file = root / manifest["derived_paths"]["check_file"]

    for path in (wrapper_file, check_file):
        if not path.exists():
            errors.append(f"缺少文件: {path}")

    if errors:
        return

    try:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        module = importlib.import_module(manifest["derived_paths"]["wrapper_module"])
    except Exception as exc:
        errors.append(f"包装器模块导入失败: {exc}")
        return

    if not hasattr(module, manifest["wrapper_class_name"]):
        errors.append(
            f"包装器类不存在: {manifest['derived_paths']['wrapper_module']}::{manifest['wrapper_class_name']}"
        )


def run_runtime_check(manifest: dict, python_executable: str, errors: list[str]) -> None:
    check_file = repo_root() / manifest["derived_paths"]["check_file"]
    env = dict(os.environ)
    root_str = str(repo_root())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        root_str if not existing_pythonpath else f"{root_str}:{existing_pythonpath}"
    )
    result = subprocess.run(
        [python_executable, str(check_file)],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        errors.append(
            f"运行时校验失败:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    errors: list[str] = []

    if args.require_source_path:
        source_path = repo_root() / manifest["vendor_repo_relpath"]
        if not source_path.exists():
            errors.append(f"外部仓库路径不存在: {source_path}")

    if not args.manifest_only:
        validate_generated_files(manifest, errors)
        validate_registered_configs(manifest, errors)
        if args.runtime_check and not errors:
            run_runtime_check(manifest, args.python_executable, errors)

    if errors:
        print("校验失败：")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)

    print("校验通过。")
    print(f"tool_name: {manifest['tool_name']}")
    print(f"manifest: {manifest['_manifest_path']}")
    if args.manifest_only:
        print("模式: manifest-only")
    elif args.runtime_check:
        print("模式: 结构 + 导入 + 运行时")
    else:
        print("模式: 结构 + 导入")


if __name__ == "__main__":
    main()
