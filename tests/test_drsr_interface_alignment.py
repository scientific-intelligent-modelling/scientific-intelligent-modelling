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


def test_drsr_anonymize_renames_features():
    """anonymize=True 时变量名变为 x1..xN，目标名为 y，不加载描述。"""
    reg = DRSRRegressor(
        feature_names=["mu", "Nn"],
        target_name="output",
        anonymize=True,
    )
    assert reg._feature_names == ["x1", "x2"]
    assert reg._target_name == "y"
    names, descs, tdesc = reg._resolve_prompt_semantics(2)
    assert names == ["x1", "x2"]
    assert descs is None
    assert tdesc is None


def test_drsr_anonymize_fallback_no_feature_names():
    """anonymize=True 且未传入 feature_names 时按 n_features 生成 x1..xN。"""
    reg = DRSRRegressor(
        n_features=3,
        anonymize=True,
    )
    assert reg._feature_names == ["x1", "x2", "x3"]
    assert reg._target_name == "y"
    names, descs, tdesc = reg._resolve_prompt_semantics(3)
    assert names == ["x1", "x2", "x3"]


def test_drsr_anonymize_default_false():
    """默认 anonymize=False，保持原始变量名。"""
    reg = DRSRRegressor(
        feature_names=["mu", "Nn"],
        target_name="output",
    )
    assert reg._feature_names == ["mu", "Nn"]
    assert reg._target_name == "output"
