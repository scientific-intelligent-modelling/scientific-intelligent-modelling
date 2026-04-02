import json
from pathlib import Path

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor


def test_drsr_wrapper_supports_llm_config_path(tmp_path: Path):
    llm_config = {
        "host": "api.bltcy.ai",
        "api_key": {
            "blt": "key-from-provider",
            "blt/gpt-3.5-turbo": "key-from-model",
        },
        "model": "blt/gpt-3.5-turbo",
        "max_tokens": 2048,
        "temperature": 0.4,
        "top_p": 0.8,
    }
    config_path = tmp_path / "llm.config"
    config_path.write_text(json.dumps(llm_config, ensure_ascii=False, indent=2), encoding="utf-8")

    reg = DRSRRegressor(llm_config_path=str(config_path))
    runtime = reg._resolve_llm_client_config()

    assert runtime["client_config"]["model"] == "blt/gpt-3.5-turbo"
    assert runtime["client_config"]["api_key"] == "key-from-model"
    assert runtime["client_config"]["base_url"] == "https://api.bltcy.ai/v1"
    assert runtime["generation_overrides"]["max_tokens"] == 2048
    assert runtime["generation_overrides"]["temperature"] == 0.4
    assert runtime["generation_overrides"]["top_p"] == 0.8


def test_drsr_wrapper_explicit_params_override_llm_config(tmp_path: Path):
    llm_config = {
        "host": "api.bltcy.ai",
        "api_key": "config-key",
        "model": "blt/gpt-3.5-turbo",
        "temperature": 0.6,
    }
    config_path = tmp_path / "llm.config"
    config_path.write_text(json.dumps(llm_config, ensure_ascii=False, indent=2), encoding="utf-8")

    reg = DRSRRegressor(
        llm_config_path=str(config_path),
        api_model="blt/gpt-4o-mini",
        api_key="explicit-key",
        api_base="https://example.com/v1",
        temperature=0.2,
        api_params={"top_p": 0.5},
    )
    runtime = reg._resolve_llm_client_config()

    assert runtime["client_config"]["model"] == "blt/gpt-4o-mini"
    assert runtime["client_config"]["api_key"] == "explicit-key"
    assert runtime["client_config"]["base_url"] == "https://example.com/v1"
    assert runtime["generation_overrides"]["temperature"] == 0.2
    assert runtime["generation_overrides"]["top_p"] == 0.5
