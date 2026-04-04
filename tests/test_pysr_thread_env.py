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
