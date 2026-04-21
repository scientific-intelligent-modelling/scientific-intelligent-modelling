import importlib.util
import importlib
import sys
import types

import numpy as np


def _load_tpsr_wrapper_module():
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda *args, **kwargs: ("device", args, kwargs)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.set_num_threads = lambda *args, **kwargs: None
    torch_stub.set_num_interop_threads = lambda *args, **kwargs: None

    normalizers_stub = types.ModuleType("scientific_intelligent_modelling.benchmarks.normalizers")
    normalizers_stub.normalize_tpsr_artifact = lambda *args, **kwargs: {}

    base_wrapper_stub = types.ModuleType(
        "scientific_intelligent_modelling.algorithms.base_wrapper"
    )

    class _BaseWrapper:
        pass

    base_wrapper_stub.BaseWrapper = _BaseWrapper

    previous_modules = {}
    stubs = {
        "torch": torch_stub,
        "scientific_intelligent_modelling.benchmarks.normalizers": normalizers_stub,
        "scientific_intelligent_modelling.algorithms.base_wrapper": base_wrapper_stub,
    }
    for name, module in stubs.items():
        previous_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        sys.modules.pop("scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper", None)
        return importlib.import_module(
            "scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper"
        )
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_downsample_reward_arrays_caps_rows_and_preserves_alignment():
    module = _load_tpsr_wrapper_module()
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)

    sampled_X, sampled_y = module.TPSRRegressor._downsample_reward_arrays(X, y, 4)

    assert sampled_X.shape == (4, 3)
    assert sampled_y.shape == (4,)
    np.testing.assert_array_equal(sampled_y, sampled_X[:, 0] / 3.0)


def test_downsample_reward_arrays_keeps_small_inputs_unchanged():
    module = _load_tpsr_wrapper_module()
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.arange(4, dtype=float)

    sampled_X, sampled_y = module.TPSRRegressor._downsample_reward_arrays(X, y, 16)

    np.testing.assert_array_equal(sampled_X, X)
    np.testing.assert_array_equal(sampled_y, y)


def test_tpsr_wrapper_defaults_align_official_bagging_config():
    module = _load_tpsr_wrapper_module()

    reg = module.TPSRRegressor()

    assert reg.params["max_input_points"] == 200
    assert reg.params["max_number_bags"] == 10
    assert reg.params["n_trees_to_refine"] == 10
    assert reg.params["no_seq_cache"] is False
    assert reg.params["no_prefix_cache"] is True
    assert reg.params["width"] == 3
    assert reg.params["num_beams"] == 1
    assert reg.params["rollout"] == 3
    assert reg.params["horizon"] == 200
    assert reg.params["lam"] == 0.1
