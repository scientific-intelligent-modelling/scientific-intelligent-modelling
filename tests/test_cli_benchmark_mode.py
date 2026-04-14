from pathlib import Path

from scientific_intelligent_modelling import cli


def test_is_canonical_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "demo_dataset"
    dataset_dir.mkdir()
    assert cli._is_canonical_dataset_dir(str(dataset_dir)) is False

    (dataset_dir / "metadata.yaml").write_text("dataset: {}\n", encoding="utf-8")
    (dataset_dir / "train.csv").write_text("x0,y\n1,2\n", encoding="utf-8")
    assert cli._is_canonical_dataset_dir(str(dataset_dir)) is True


def test_cli_uses_runner_for_canonical_dataset(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "demo_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.yaml").write_text("dataset:\n  target:\n    name: y\n", encoding="utf-8")
    (dataset_dir / "train.csv").write_text("x0,y\n1,2\n", encoding="utf-8")

    captured = {}

    def fake_run_benchmark_task(*, tool_name, dataset_dir, output_root, seed, params_override):
        captured["tool_name"] = tool_name
        captured["dataset_dir"] = str(dataset_dir)
        captured["output_root"] = str(output_root)
        captured["seed"] = seed
        captured["params_override"] = dict(params_override)
        out = Path(output_root) / tool_name / Path(dataset_dir).name / "result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return out

    monkeypatch.setattr(cli, "run_benchmark_task", fake_run_benchmark_task)

    cli.main(
        [
            "--algorithm",
            "pysr",
            "--train-path",
            str(dataset_dir),
            "--seed",
            "1316",
            "--output-root",
            str(tmp_path / "bench_results"),
            "--niterations",
            "5",
        ]
    )

    assert captured["tool_name"] == "pysr"
    assert captured["dataset_dir"] == str(dataset_dir.resolve())
    assert captured["output_root"] == str((tmp_path / "bench_results").resolve())
    assert captured["seed"] == 1316
    assert captured["params_override"]["niterations"] == 5
