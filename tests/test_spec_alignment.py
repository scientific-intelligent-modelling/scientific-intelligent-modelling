import numpy as np

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor
from scientific_intelligent_modelling.algorithms.llmsr_wrapper.llmsr.llmsr import prompts


def test_llmsr_and_drsr_spec_align_text_semantics():
    background = "Calculate Acceleration in Nonl-linear Harmonic Oscillator given Position at time t, Time and Velocity at time t"
    features = ["x0", "x1", "x2"]

    llmsr_spec = prompts.build_specification(
        background=background,
        features=features,
        target="y",
        max_params=12,
        problem="PO0",
    )

    reg = DRSRRegressor(problem_name="PO0", background=background, max_params=12)
    drsr_spec = reg._build_spec_from_background(np.zeros((8, 3)), np.zeros(8), background)

    for text in (llmsr_spec, drsr_spec):
        assert "Find the mathematical function skeleton that fits the data." in text
        assert "represents PO0" not in text
        assert "Variables:" in text
        assert "- Independents: x0, x1, x2" in text
        assert "- Dependent: y" in text
        assert "MAX_NPARAMS = 12" in text
        assert "def equation(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:" in text
        assert "return params[0] + params[1] * x0 + params[2] * x1 + params[3] * x2" in text

    assert "from scipy.optimize import minimize" in llmsr_spec
    assert "from scipy.optimize import minimize" not in drsr_spec
    assert "BFGS_PARAMS = None" in llmsr_spec
    assert "BFGS_PARAMS = None" not in drsr_spec


def test_spec_injects_feature_and_target_descriptions():
    background = "demo background"
    features = ["x0", "x1", "x2"]
    feature_descriptions = ["Position at time t", "Time", "Velocity at time t"]
    target_description = "Acceleration in Nonl-linear Harmonic Oscillator"

    llmsr_spec = prompts.build_specification(
        background=background,
        features=features,
        target="y",
        max_params=12,
        problem="PO0",
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    reg = DRSRRegressor(
        problem_name="PO0",
        background=background,
        max_params=12,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )
    drsr_spec = reg._build_spec_from_background(np.zeros((8, 3)), np.zeros(8), background)

    for text in (llmsr_spec, drsr_spec):
        assert "- Independents:" in text
        assert "  - x0: Position at time t" in text
        assert "  - x1: Time" in text
        assert "  - x2: Velocity at time t" in text
        assert "- Dependent:" in text
        assert "  - y: Acceleration in Nonl-linear Harmonic Oscillator" in text


def test_llmsr_prompt_default_max_params_is_10():
    spec = prompts.build_specification(
        background="demo background",
        features=["x0", "x1"],
        target="y",
        problem="demo_problem",
    )

    assert "MAX_NPARAMS = 10" in spec
    assert "MAX_NPARAMS = 12" not in spec


def test_spec_builder_clamps_max_params_to_linear_seed_requirement():
    spec = prompts.build_specification(
        background="demo background",
        features=["x0", "x1"],
        target="y",
        max_params=2,
        problem="demo_problem",
    )

    assert "MAX_NPARAMS = 3" in spec
    assert "return params[0] + params[1] * x0 + params[2] * x1" in spec
