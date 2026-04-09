from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
from string import Template

from common import load_json, load_manifest, repo_root, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="基于 manifest 生成符号回归工具接入脚手架")
    parser.add_argument("--manifest", required=True, help="manifest 路径")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有但内容不同的文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要写入的文件，不实际落盘")
    return parser.parse_args()


def render_template(template_path: Path, context: dict) -> str:
    template = Template(template_path.read_text(encoding="utf-8"))
    return template.substitute(context)


def write_text(path: Path, content: str, overwrite: bool, dry_run: bool, touched: list[str]) -> None:
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == content:
            print(f"[unchanged] {path}")
            return
        if not overwrite:
            raise FileExistsError(f"文件已存在且内容不同，请使用 --overwrite: {path}")

    if dry_run:
        print(f"[dry-run] {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[write] {path}")
    touched.append(str(path))


def update_toolbox_config(manifest: dict, overwrite: bool, dry_run: bool, touched: list[str]) -> None:
    path = repo_root() / manifest["derived_paths"]["toolbox_config"]
    data = load_json(path)
    tool_mapping = dict(data.get("tool_mapping") or {})
    desired = {
        "env": manifest["env_name"],
        "regressor": manifest["wrapper_class_name"],
    }
    existing = tool_mapping.get(manifest["tool_name"])
    if existing and existing != desired and not overwrite:
        raise ValueError(f"toolbox_config 中 '{manifest['tool_name']}' 已存在且与 manifest 不一致")
    tool_mapping[manifest["tool_name"]] = desired
    data["tool_mapping"] = tool_mapping
    if dry_run:
        print(f"[dry-run] {path}")
        touched.append(str(path))
        return
    save_json(path, data)
    print(f"[update] {path}")
    touched.append(str(path))


def update_envs_config(manifest: dict, overwrite: bool, dry_run: bool, touched: list[str]) -> None:
    path = repo_root() / manifest["derived_paths"]["envs_config"]
    data = load_json(path)
    env_list = dict(data.get("env_list") or {})
    desired = {
        "python_version": manifest["env"]["python_version"],
        "conda_packages": manifest["env"]["conda_packages"],
        "pip_packages": manifest["env"]["pip_packages"],
        "channels": manifest["env"]["channels"],
        "post_install_commands": manifest["env"]["post_install_commands"],
        "comments": manifest["env"]["comments"],
    }
    existing = env_list.get(manifest["env_name"])
    if existing and existing != desired and not overwrite:
        raise ValueError(f"envs_config 中 '{manifest['env_name']}' 已存在且与 manifest 不一致")
    env_list[manifest["env_name"]] = desired
    data["env_list"] = env_list
    if dry_run:
        print(f"[dry-run] {path}")
        touched.append(str(path))
        return
    save_json(path, data)
    print(f"[update] {path}")
    touched.append(str(path))


def build_context(manifest: dict) -> dict:
    adapter = manifest["adapter"]
    entrypoint = manifest["entrypoint"]
    smoke_test = manifest["smoke_test"]
    return {
        "tool_name_literal": repr(manifest["tool_name"]),
        "display_name": manifest["display_name"],
        "display_name_literal": repr(manifest["display_name"]),
        "wrapper_class_name": manifest["wrapper_class_name"],
        "integration_mode_literal": repr(manifest["integration_mode"]),
        "source_mode_literal": repr(manifest["source_mode"]),
        "vendor_repo_relpath_literal": repr(manifest["vendor_repo_relpath"]),
        "entry_module_literal": repr(entrypoint.get("module") or ""),
        "entry_object_literal": repr(entrypoint.get("object_name") or ""),
        "x_shape_literal": repr(adapter["x_shape"]),
        "supports_predict_literal": "True" if adapter["supports_predict"] else "False",
        "serialization_mode_literal": repr(adapter["serialization_mode"]),
        "meta_params_literal": pformat(tuple(adapter["meta_params"])),
        "constructor_allowlist_literal": pformat(tuple(adapter["constructor_param_allowlist"])),
        "fit_allowlist_literal": pformat(tuple(adapter["fit_param_allowlist"])),
        "equation_notes_literal": repr(adapter["equation_notes"]),
        "problem_name_literal": repr(smoke_test["problem_name"]),
        "predict_rows_literal": str(smoke_test["predict_rows"]),
        "default_params_literal": pformat(smoke_test["default_params"]),
    }


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    root = repo_root()
    touched: list[str] = []

    context = build_context(manifest)
    template_dir = root / "tools" / "sr_onboarder" / "templates"

    wrapper_content = render_template(template_dir / "wrapper_template.py.tmpl", context)
    init_content = (
        f"from .wrapper import {manifest['wrapper_class_name']}\n\n"
        f"__all__ = ['{manifest['wrapper_class_name']}']\n"
    )
    check_content = render_template(template_dir / "check_template.py.tmpl", context)

    wrapper_file = root / manifest["derived_paths"]["wrapper_file"]
    init_file = root / manifest["derived_paths"]["wrapper_init"]
    check_file = root / manifest["derived_paths"]["check_file"]

    write_text(wrapper_file, wrapper_content, args.overwrite, args.dry_run, touched)
    write_text(init_file, init_content, args.overwrite, args.dry_run, touched)
    write_text(check_file, check_content, args.overwrite, args.dry_run, touched)
    update_toolbox_config(manifest, args.overwrite, args.dry_run, touched)
    update_envs_config(manifest, args.overwrite, args.dry_run, touched)

    print("\n生成完成。")
    print(f"tool_name: {manifest['tool_name']}")
    print(f"wrapper: {wrapper_file}")
    print(f"check: {check_file}")
    if args.dry_run:
        print("当前为 dry-run，未实际写入文件。")
    else:
        print("请继续补齐生成的 wrapper.py 中工具特有逻辑，然后运行 validate_sr_tool.py。")


if __name__ == "__main__":
    main()
