import os
import sys
import types

import numpy as np

from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor


def test_pysr_fit_forces_thread_env_to_four():
    old_module = sys.modules.get("pysr")
    captured = {}

    fake_module = types.ModuleType("pysr")

    class FakePySRRegressor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            captured["env"] = {
                key: os.environ.get(key)
                for key in PySRRegressor._THREAD_ENV_VARS
            }

        def fit(self, X, y):
            captured["shape"] = (tuple(X.shape), tuple(y.shape))

    fake_module.PySRRegressor = FakePySRRegressor
    sys.modules["pysr"] = fake_module

    try:
        reg = PySRRegressor(niterations=5, population_size=32, procs=8)
        reg.fit(np.zeros((4, 2)), np.zeros(4))
    finally:
        if old_module is not None:
            sys.modules["pysr"] = old_module
        else:
            sys.modules.pop("pysr", None)

    assert captured["kwargs"]["procs"] == 8
    assert captured["shape"] == ((4, 2), (4,))
    for key, value in captured["env"].items():
        assert value == "4", f"{key} 未固定为 4，而是 {value}"


def test_pysr_can_restore_from_existing_experiment_dir(tmp_path):
    old_module = sys.modules.get("pysr")
    captured = {}

    fake_module = types.ModuleType("pysr")

    class FakeRecoveredModel:
        def __init__(self):
            self.equations_ = ["x0 + x1"]
            self.n_features_in_ = 2

        def predict(self, X):
            return X[:, 0] + X[:, 1]

        def sympy(self):
            return "x0 + x1"

    class FakePySRRegressor:
        @staticmethod
        def from_file(run_directory):
            captured["run_directory"] = run_directory
            return FakeRecoveredModel()

    fake_module.PySRRegressor = FakePySRRegressor
    sys.modules["pysr"] = fake_module

    exp_dir = tmp_path / "experiments" / "demo_run"
    exp_dir.mkdir(parents=True)

    try:
        reg = PySRRegressor(existing_exp_dir=str(exp_dir), niterations=5, population_size=32)
        serialized = reg.serialize()
        restored = PySRRegressor.deserialize(serialized)
        preds = restored.predict(np.array([[1.0, 2.0], [3.0, 4.0]]))
        equation = restored.get_optimal_equation()
    finally:
        if old_module is not None:
            sys.modules["pysr"] = old_module
        else:
            sys.modules.pop("pysr", None)

    assert captured["run_directory"] == str(exp_dir.resolve())
    assert equation == "x0 + x1"
    assert preds.tolist() == [3.0, 7.0]
