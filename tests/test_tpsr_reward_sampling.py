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


def test_tpsr_runtime_feature_context_tracks_current_dataset():
    module = _load_tpsr_wrapper_module()

    reg = module.TPSRRegressor(max_input_dimension=10)
    X = np.arange(20, dtype=float).reshape(5, 4)

    n_features = reg._capture_runtime_feature_context(X)

    assert n_features == 4
    assert reg._n_features == 4
    assert reg._predict_variable_names == ["x_0", "x_1", "x_2", "x_3"]


def test_tpsr_progress_state_projects_out_of_range_variables_to_zero():
    module = _load_tpsr_wrapper_module()

    reg = module.TPSRRegressor()
    reg._n_features = 4
    written = []
    reg._write_progress_state = written.append

    reg._emit_progress_equation(
        equation="x_0 + x_9",
        score=1.0,
        complexity=3,
        source="unit_test",
    )
    assert len(written) == 1
    assert written[0]["equation"] == "x_0 + 0"

    written.clear()
    reg._emit_progress_equation(
        equation="x_0 + x_3",
        score=1.0,
        complexity=3,
        source="unit_test",
    )
    assert len(written) == 1
    assert written[0]["equation"] == "x_0 + x_3"


def test_tpsr_variable_budget_accepts_one_based_tokens():
    module = _load_tpsr_wrapper_module()

    assert module.TPSRRegressor._equation_within_feature_budget("x_1 + x_4", 4) is True
    assert module.TPSRRegressor._equation_within_feature_budget("x_1 + x_5", 4) is False


def test_tpsr_projects_out_of_range_one_based_tokens_to_zero():
    module = _load_tpsr_wrapper_module()

    assert module.TPSRRegressor._project_equation_to_feature_budget("x_1 + x_5", 4) == "x_1 + 0"
