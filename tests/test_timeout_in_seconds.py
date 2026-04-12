import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor
from scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper import GPLearnRegressor
from scientific_intelligent_modelling.algorithms.operon_wrapper.wrapper import OperonRegressor
from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor
from scientific_intelligent_modelling.srkit import subprocess_runner
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def test_strict_wrappers_accept_timeout_meta_param():
    pysr = PySRRegressor(timeout_in_seconds=7, niterations=5, population_size=16)
    assert pysr.params["timeout_in_seconds"] == 7

    operon = OperonRegressor(timeout_in_seconds=9, niterations=4, population_size=16)
    assert operon.params["max_time"] == 9
    assert operon.params["generations"] == 4

    gplearn = GPLearnRegressor(timeout_in_seconds=11, generations=3, population_size=32)
    assert "timeout_in_seconds" not in gplearn.params

    dso = DSORegressor(timeout_in_seconds=13)
    assert "timeout_in_seconds" not in dso.params


def test_subprocess_runner_handle_fit_strips_timeout_meta_param():
    seen = {}

    class DummyRegressor:
        def __init__(self, **kwargs):
            seen["kwargs"] = dict(kwargs)

        def fit(self, X, y):
            seen["shape"] = (tuple(X.shape), tuple(y.shape))

        def serialize(self):
            return "dummy-model"

    result = subprocess_runner.handle_fit(
        DummyRegressor,
        {
            "tool_name": "dummy",
            "data": {"X": [[1.0], [2.0]], "y": [3.0, 4.0]},
            "params": {"timeout_in_seconds": 5, "alpha": 1},
        },
    )

    assert result["serialized_model"] == "dummy-model"
    assert seen["kwargs"] == {"alpha": 1}
    assert seen["shape"] == ((2, 1), (2,))


def test_symbolic_regressor_timeout_updates_manifest(monkeypatch):
    monkeypatch.setattr(
        "scientific_intelligent_modelling.srkit.regressor.env_manager.get_env_python",
        lambda env_name: sys.executable,
    )
    monkeypatch.setattr(
        SymbolicRegressor,
        "_execute_subprocess",
        lambda self, command: (_ for _ in ()).throw(TimeoutError("boom")),
    )

    reg = SymbolicRegressor("gplearn", timeout_in_seconds=3, generations=2, population_size=8)

    with pytest.raises(TimeoutError):
        reg.fit(np.array([[1.0], [2.0]]), np.array([3.0, 4.0]))

    manifest_path = Path(reg.experiment_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "timed_out"
    assert manifest["timeout_in_seconds"] == 3


def test_symbolic_regressor_prefers_explicit_fit_timeout(monkeypatch):
    monkeypatch.setattr(
        "scientific_intelligent_modelling.srkit.regressor.env_manager.get_env_python",
        lambda env_name: sys.executable,
    )
    reg = SymbolicRegressor("gplearn", timeout_in_seconds=12)
    timeout = reg._resolve_subprocess_timeout_seconds(
        {"action": "fit", "params": {"timeout_in_seconds": 12}}
    )
    assert timeout == 12
