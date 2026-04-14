from pathlib import Path

import numpy as np
import yaml

from scientific_intelligent_modelling.algorithms.drsr_wrapper.drsr.drsr_420.prompt_config import PromptContext
from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor


def test_prompt_context_uses_x_variables_and_metadata_semantics():
    ctx = PromptContext(
        n_features=2,
        feature_names=["x0", "x1"],
        dependent_name="y",
        problem_name="MatSci0",
        background="Calculate Stress given Strain and Temperature",
        feature_descriptions=["Strain", "Temperature"],
        target_description="Stress",
    )

    rendered = "\n".join(
        [
            ctx.render_head(),
            ctx.render_instruction(),
            ctx.render_residual_block_title(),
            ctx.render_residual_analysis_prompt("{}", "[[1, 2, 3, 4]]", "return params[0]"),
        ]
    )

    assert "with driving force" not in rendered
    assert "col0" not in rendered
    assert "col1" not in rendered
    assert "MatSci0" not in rendered
    assert "x0 (Strain)" in rendered
    assert "x1 (Temperature)" in rendered
    assert "y (Stress)" in rendered


def test_drsr_wrapper_prompt_semantics_follow_metadata(tmp_path: Path):
    metadata = {
        "dataset": {
            "features": [
                {"name": "epsilon", "description": "Strain"},
                {"name": "T", "description": "Temperature"},
            ],
            "target": {"name": "sigma", "description": "Stress"},
        }
    }
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata, allow_unicode=True), encoding="utf-8")

    reg = DRSRRegressor(
        problem_name="MatSci0",
        background="Calculate Stress given Strain and Temperature",
        metadata_path=str(metadata_path),
    )

    feature_names, feature_descriptions, target_description = reg._resolve_prompt_semantics(2)
    assert feature_names == ["x0", "x1"]
    assert feature_descriptions == ["Strain", "Temperature"]
    assert target_description == "Stress"

    spec = reg._build_spec_from_background(np.zeros((8, 2)), np.zeros(8), "Calculate Stress given Strain and Temperature")
    assert "  - x0: Strain" in spec
    assert "  - x1: Temperature" in spec
    assert "  - y: Stress" in spec
