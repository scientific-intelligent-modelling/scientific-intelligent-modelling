from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor


def test_drsr_budget_maps_to_llmsr_style_parameters():
    reg = DRSRRegressor(
        niterations=50,
        samples_per_iteration=4,
    )
    niterations, samples_per_iteration, max_samples = reg._resolve_search_budget()

    assert niterations == 50
    assert samples_per_iteration == 4
    assert max_samples == 200


def test_drsr_exp_layout_prefers_exp_path_and_exp_name(tmp_path):
    reg = DRSRRegressor(
        exp_path=str(tmp_path / "experiments"),
        exp_name="demo_run",
        workdir=str(tmp_path / "legacy_workdir"),
    )

    experiments_root, exp_name, workdir = reg._resolve_experiment_layout()

    assert exp_name == "demo_run"
    assert workdir == str(tmp_path / "experiments" / "demo_run")
    assert experiments_root == str(tmp_path / "experiments")


def test_drsr_default_exp_layout_uses_experiments_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reg = DRSRRegressor(problem_name="demo_problem")

    experiments_root, exp_name, workdir = reg._resolve_experiment_layout()

    assert experiments_root == str(tmp_path / "experiments")
    assert workdir == str(tmp_path / "experiments" / exp_name)
    assert exp_name.startswith("drsr_demo_problem_")
