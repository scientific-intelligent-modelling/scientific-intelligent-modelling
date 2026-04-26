from scientific_intelligent_modelling.algorithms.udsr_wrapper.wrapper import UDSRRegressor
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_udsr_artifact


def test_udsr_config_drops_component_metadata_before_dso():
    config = UDSRRegressor._build_config(
        {
            "benchmark_variant": "udsr_trunk_dso_poly_gp_meld",
            "component_flags": {"aif": False},
            "component_notes": "metadata only",
        }
    )

    assert "benchmark_variant" not in config
    assert "component_flags" not in config
    assert "component_notes" not in config
    assert config["gp_meld"]["run_gp_meld"] is True


def test_udsr_artifact_marks_trunk_components():
    artifact = normalize_udsr_artifact("x1 + 1", expected_n_features=2)

    assert artifact["tool_name"] == "udsr"
    assert artifact["benchmark_variant"] == "udsr_trunk_dso_poly_gp_meld"
    assert artifact["component_flags"] == {
        "aif": False,
        "dsr": True,
        "lspt": False,
        "gp_meld": True,
        "linear_poly": True,
    }
    assert "without AIF" in artifact["component_notes"]
