from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


VALID_INTEGRATION_MODES = {"python_api", "cli"}
VALID_SOURCE_MODES = {"submodule", "vendored", "pip_editable", "external_path"}
VALID_X_SHAPES = {"samples_first", "features_first"}
VALID_SERIALIZATION_MODES = {"base_pickle", "json_state"}
DEFAULT_META_PARAMS = ["exp_name", "exp_path", "problem_name", "seed"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tool_file_stem(tool_name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", tool_name).strip("_").lower()
    return value or "tool"


def camelize_tool_name(tool_name: str) -> str:
    if "_" in tool_name or "-" in tool_name:
        parts = [part for part in re.split(r"[_\-]+", tool_name) if part]
        return "".join(part[:1].upper() + part[1:] for part in parts)
    return tool_name[:1].upper() + tool_name[1:]


def _require_string(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest 字段 '{key}' 必须是非空字符串")
    return value.strip()


def load_manifest(path_str: str) -> Dict[str, Any]:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        raise FileNotFoundError(f"manifest 不存在: {path}")
    data = load_json(path)
    data["_manifest_path"] = str(path)
    return normalize_manifest(data)


def normalize_manifest(raw_manifest: Dict[str, Any]) -> Dict[str, Any]:
    manifest = dict(raw_manifest)
    tool_name = _require_string(manifest, "tool_name")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tool_name):
        raise ValueError("tool_name 只允许字母、数字和下划线，且必须以字母开头")

    display_name = str(manifest.get("display_name") or tool_name)
    wrapper_class_name = str(
        manifest.get("wrapper_class_name") or f"{camelize_tool_name(tool_name)}Regressor"
    )
    integration_mode = str(manifest.get("integration_mode") or "python_api")
    source_mode = str(manifest.get("source_mode") or "submodule")
    vendor_repo_relpath = _require_string(manifest, "vendor_repo_relpath")

    if Path(vendor_repo_relpath).is_absolute():
        raise ValueError("vendor_repo_relpath 必须是仓库内相对路径")
    if integration_mode not in VALID_INTEGRATION_MODES:
        raise ValueError(f"integration_mode 仅支持: {', '.join(sorted(VALID_INTEGRATION_MODES))}")
    if source_mode not in VALID_SOURCE_MODES:
        raise ValueError(f"source_mode 仅支持: {', '.join(sorted(VALID_SOURCE_MODES))}")

    entrypoint = dict(manifest.get("entrypoint") or {})
    if integration_mode == "python_api":
        _require_string(entrypoint, "module")
        _require_string(entrypoint, "object_name")
    else:
        entrypoint.setdefault("module", "")
        entrypoint.setdefault("object_name", "")

    env = dict(manifest.get("env") or {})
    env_name = _require_string(env, "name")
    _require_string(env, "python_version")
    env["conda_packages"] = list(env.get("conda_packages") or [])
    env["pip_packages"] = list(env.get("pip_packages") or [])
    env["channels"] = list(env.get("channels") or [])
    env["post_install_commands"] = list(env.get("post_install_commands") or [])
    env["comments"] = str(env.get("comments") or "")

    adapter = dict(manifest.get("adapter") or {})
    x_shape = str(adapter.get("x_shape") or "samples_first")
    serialization_mode = str(adapter.get("serialization_mode") or "base_pickle")
    if x_shape not in VALID_X_SHAPES:
        raise ValueError(f"adapter.x_shape 仅支持: {', '.join(sorted(VALID_X_SHAPES))}")
    if serialization_mode not in VALID_SERIALIZATION_MODES:
        raise ValueError(
            f"adapter.serialization_mode 仅支持: {', '.join(sorted(VALID_SERIALIZATION_MODES))}"
        )
    adapter["x_shape"] = x_shape
    adapter["supports_predict"] = bool(adapter.get("supports_predict", True))
    adapter["serialization_mode"] = serialization_mode
    adapter["meta_params"] = list(adapter.get("meta_params") or DEFAULT_META_PARAMS)
    adapter["constructor_param_allowlist"] = list(adapter.get("constructor_param_allowlist") or [])
    adapter["fit_param_allowlist"] = list(adapter.get("fit_param_allowlist") or [])
    adapter["equation_notes"] = str(adapter.get("equation_notes") or "")

    smoke_test = dict(manifest.get("smoke_test") or {})
    smoke_test["problem_name"] = str(smoke_test.get("problem_name") or f"ci_{tool_file_stem(tool_name)}")
    smoke_test["predict_rows"] = int(smoke_test.get("predict_rows") or 4)
    smoke_test["default_params"] = dict(smoke_test.get("default_params") or {})

    wrapper_dir = Path("scientific_intelligent_modelling") / "algorithms" / f"{tool_name}_wrapper"
    manifest.update(
        {
            "tool_name": tool_name,
            "display_name": display_name,
            "wrapper_class_name": wrapper_class_name,
            "integration_mode": integration_mode,
            "source_mode": source_mode,
            "vendor_repo_relpath": vendor_repo_relpath,
            "entrypoint": entrypoint,
            "env": env,
            "adapter": adapter,
            "smoke_test": smoke_test,
            "derived_paths": {
                "wrapper_dir": str(wrapper_dir),
                "wrapper_file": str(wrapper_dir / "wrapper.py"),
                "wrapper_init": str(wrapper_dir / "__init__.py"),
                "check_file": str(Path("check") / f"check_{tool_file_stem(tool_name)}.py"),
                "toolbox_config": "scientific_intelligent_modelling/config/toolbox_config.json",
                "envs_config": "scientific_intelligent_modelling/config/envs_config.json",
                "wrapper_module": f"scientific_intelligent_modelling.algorithms.{tool_name}_wrapper.wrapper",
            },
            "tool_file_stem": tool_file_stem(tool_name),
            "env_name": env_name,
        }
    )
    return manifest
