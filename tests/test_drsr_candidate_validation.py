import numpy as np

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor


def test_restore_skips_non_executable_candidate():
    reg = DRSRRegressor()
    reg._n_features = 3

    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
            [0.4, 0.8, 1.2],
        ],
        dtype=float,
    )
    y = X[:, 0] + X[:, 2]

    bad_equation = """
    p = np.zeros(MAX_NPARAMS)
    n_params = min(len(params), MAX_NPARAMS)
    p[:n_params] = params[:n_params]
    return p[0] * col0 + p[1] * col2
    """
    good_equation = """
    return params[0] * col0 + params[1] * col2
    """

    entries = [
        {
            "equation": bad_equation,
            "params": [1.0, 1.0],
            "score": 10.0,
            "category": "Good",
            "sample_order": 1,
        },
        {
            "equation": good_equation,
            "params": [1.0, 1.0],
            "score": 9.0,
            "category": "Good",
            "sample_order": 2,
        },
    ]

    ok = reg._restore_from_experiences(X, y, [bad_equation, good_equation], entries)

    assert ok is True
    assert reg._equation_func is not None
    assert "MAX_NPARAMS" not in reg.get_optimal_equation()

    pred = reg.predict(X[:2])
    assert pred.shape == (2,)
    assert np.allclose(pred, y[:2])
