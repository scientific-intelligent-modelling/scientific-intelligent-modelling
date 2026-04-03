from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor


def test_pysr_defaults_enable_progress_and_basic_verbosity():
    reg = PySRRegressor(niterations=5, population_size=32)
    assert reg.params["progress"] is True
    assert reg.params["verbosity"] == 1


def test_pysr_explicit_progress_and_verbosity_override_defaults():
    reg = PySRRegressor(
        niterations=5,
        population_size=32,
        progress=False,
        verbosity=0,
    )
    assert reg.params["progress"] is False
    assert reg.params["verbosity"] == 0
