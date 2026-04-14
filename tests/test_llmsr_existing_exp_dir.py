import json
from pathlib import Path

import numpy as np

from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import LLMSRRegressor


def test_llmsr_fit_reuses_existing_experiment_dir(tmp_path):
    exp_dir = tmp_path / "llmsr_exp"
    exp_dir.mkdir()
    (exp_dir / "samples").mkdir()
    (exp_dir / "meta.json").write_text(
        json.dumps(
            {
                "problem_name": "demo",
                "feature_names": ["x0", "x1"],
                "target_name": "y",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reg = LLMSRRegressor(existing_exp_dir=str(exp_dir))
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 6.0])

    fitted = reg.fit(X, y)

    assert fitted is reg
    assert reg._core is None
    assert reg._exp_dir == str(exp_dir.resolve())
    assert reg._n_features == 2


def test_llmsr_existing_exp_dir_supports_multidoc_and_multiline_return(tmp_path):
    exp_dir = tmp_path / "llmsr_exp"
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(parents=True)
    (exp_dir / "meta.json").write_text(
        json.dumps(
            {
                "problem_name": "demo",
                "feature_names": ["x0", "x1"],
                "target_name": "y",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (samples_dir / "top01_demo.json").write_text(
        json.dumps(
            {
                "nmse": 0.1,
                "mse": 0.1,
                "params": [2.0, 3.0],
                "function": (
                    "def equation(x0: np.ndarray, x1: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
                    "    \"\"\"first doc\"\"\"\n"
                    "    \"\"\"second doc\"\"\"\n"
                    "    return (\n"
                    "        params[0] * x0\n"
                    "        + params[1] * x1\n"
                    "    )\n"
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reg = LLMSRRegressor(existing_exp_dir=str(exp_dir))
    pred = reg.predict(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert np.allclose(pred, np.array([8.0, 18.0]))
    assert "params[0]" in reg.get_optimal_equation()
