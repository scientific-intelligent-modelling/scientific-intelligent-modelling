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


def test_pysr_maps_exp_layout_to_native_output_directory(tmp_path):
    reg = PySRRegressor(
        niterations=5,
        population_size=32,
        exp_path=str(tmp_path / "experiments"),
        exp_name="demo_run",
    )

    assert reg.params["output_directory"] == str((tmp_path / "experiments").resolve())
    assert reg.params["run_id"] == "demo_run"


def test_pysr_accepts_advanced_probe_params():
    constraints = {"/": (-1, 1), "pow": (1, 1)}
    nested_constraints = {"pow": {"pow": 0}}
    complexity_of_operators = {"pow": 3, "sin": 2}

    reg = PySRRegressor(
        niterations=5,
        population_size=32,
        max_evals=1000,
        early_stop_condition="stop_if(loss, complexity) = loss < 1e-6",
        constraints=constraints,
        nested_constraints=nested_constraints,
        complexity_of_operators=complexity_of_operators,
        complexity_of_constants=2,
        complexity_of_variables=[1, 1],
        precision=32,
        deterministic=True,
    )

    assert reg.params["max_evals"] == 1000
    assert reg.params["early_stop_condition"] == "stop_if(loss, complexity) = loss < 1e-6"
    assert reg.params["constraints"] == constraints
    assert reg.params["nested_constraints"] == nested_constraints
    assert reg.params["complexity_of_operators"] == complexity_of_operators
    assert reg.params["complexity_of_constants"] == 2
    assert reg.params["complexity_of_variables"] == [1, 1]
    assert reg.params["precision"] == 32
    assert reg.params["deterministic"] is True


def test_pysr_accepts_deterministic_parallelism_combo():
    reg = PySRRegressor(
        niterations=5,
        population_size=32,
        deterministic=True,
        parallelism="serial",
        seed=1314,
    )

    assert reg.params["deterministic"] is True
    assert reg.params["parallelism"] == "serial"
    assert reg.params["random_state"] == 1314
