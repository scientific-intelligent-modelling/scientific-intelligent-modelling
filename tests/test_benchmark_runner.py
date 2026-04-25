import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scientific_intelligent_modelling.benchmarks import runner


class _FakeRegressor:
    def __init__(self, tool_name, problem_name=None, seed=1314, **kwargs):
        self.tool_name = tool_name
        self.problem_name = problem_name
        self.seed = seed
        self.params = dict(kwargs)
        exp_path = Path(self.params["exp_path"])
        exp_name = self.params["exp_name"]
        self.experiment_dir = str((exp_path / exp_name).resolve())
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

    def fit(self, X, y):
        self._n_features = X.shape[1]
        return self

    def predict(self, X):
        return X[:, 0]

    def get_optimal_equation(self):
        return "x0"

    def get_total_equations(self):
        return ["x0", "x0 + 1"]

    def export_canonical_symbolic_program(self):
        return {
            "version": "csp_v1",
            "tool_name": self.tool_name,
            "raw_equation": "x0",
            "raw_equation_kind": "plain_expression",
            "python_function_source": "def equation(x0, params):\n    return x0\n",
            "return_expression_source": "x0",
            "normalized_expression": "x0",
            "instantiated_expression": "x0",
            "variables": ["x0"],
            "parameter_symbols": [],
            "parameter_values": None,
            "expected_n_features": self._n_features,
            "operator_set": [],
            "ast_node_count": 1,
            "tree_depth": 1,
            "normalization_mode": "fake",
            "normalization_notes": [],
            "artifact_valid": True,
            "validation_errors": [],
            "fidelity_check": {},
        }


class _TimeoutFakeRegressor(_FakeRegressor):
    def fit(self, X, y):
        raise TimeoutError("fit timeout")


class _TimeoutRecoverableFakeRegressor(_FakeRegressor):
    def fit(self, X, y):
        raise TimeoutError("fit timeout")


class BenchmarkRunnerTest(unittest.TestCase):
    def test_predict_from_canonical_artifact_reports_uninstantiated_params(self):
        artifact = {
            "normalized_expression": "c0 + x0",
            "instantiated_expression": "c0 + x0",
        }
        with self.assertRaisesRegex(ValueError, "非标准变量: c0"):
            runner._predict_from_canonical_artifact(artifact, np.asarray([[1.0], [2.0]]))

    def test_run_benchmark_task_writes_outer_and_experiment_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  description: demo dataset
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "x0,y\n1,1\n2,2\n",
                "valid.csv": "x0,y\n3,3\n",
                "id_test.csv": "x0,y\n4,4\n",
                "ood_test.csv": "x0,y\n5,5\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _FakeRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="gplearn",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                )
            finally:
                runner.SymbolicRegressor = original_cls

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["tool"], "gplearn")
            self.assertEqual(result["dataset"], dataset_dir.name)
            self.assertEqual(result["equation"], "x0")
            self.assertEqual(result["equation_count"], 2)
            self.assertIsNotNone(result["valid"])
            self.assertIsNotNone(result["id_test"])
            self.assertIsNotNone(result["ood_test"])
            self.assertTrue(result["experiment_dir"])

            experiment_result = Path(result["experiment_dir"]) / "result.json"
            self.assertTrue(experiment_result.exists())
            self.assertEqual(
                json.loads(experiment_result.read_text(encoding="utf-8")),
                result,
            )

    def test_run_benchmark_task_ignores_seed_in_params_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n", encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _FakeRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="iMCTS",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                    params_override={"seed": 2025},
                )
            finally:
                runner.SymbolicRegressor = original_cls

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["seed"], 1314)
            self.assertNotIn("seed", result["params"])

    def test_run_benchmark_task_maps_timeout_to_timed_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n", encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _TimeoutFakeRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="pysr",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                )
            finally:
                runner.SymbolicRegressor = original_cls

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "timed_out")
            self.assertIn("TimeoutError", result["error"])
            self.assertTrue(result["experiment_dir"])

    def test_run_benchmark_task_recovers_timeout_result_from_candidate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "x0,y\n1,1\n2,2\n",
                "valid.csv": "x0,y\n3,3\n",
                "id_test.csv": "x0,y\n4,4\n",
                "ood_test.csv": "x0,y\n5,5\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            original_extract = runner._extract_periodic_candidate
            runner.SymbolicRegressor = _TimeoutRecoverableFakeRegressor
            runner._extract_periodic_candidate = lambda tool, exp_dir: {"equation": "x0", "loss": 0.1}
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="pysr",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                )
            finally:
                runner.SymbolicRegressor = original_cls
                runner._extract_periodic_candidate = original_extract

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "timed_out")
            self.assertEqual(result["equation"], "x0")
            self.assertIsNotNone(result["canonical_artifact"])
            self.assertIsNotNone(result["valid"])
            self.assertIsNotNone(result["id_test"])
            self.assertIsNotNone(result["ood_test"])
            self.assertEqual(result["equation_count"], 1)

    def test_extract_drsr_candidate_backfills_params_from_matching_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "drsr_exp"
            samples_dir = exp_dir / "samples"
            best_history_dir = exp_dir / "best_history"
            samples_dir.mkdir(parents=True)
            best_history_dir.mkdir(parents=True)
            function = (
                "def equation(beta, alpha, params):\n"
                "    return params[0] + params[1] * beta + params[2] * alpha\n"
            )
            (samples_dir / "top01_samples_0.json").write_text(
                json.dumps({"function": function, "score": 2.0}),
                encoding="utf-8",
            )
            (best_history_dir / "best_sample_0.json").write_text(
                json.dumps({"function": function, "score": 1.0, "params": [1.0, 2.0, 3.0]}),
                encoding="utf-8",
            )

            candidate = runner._extract_drsr_periodic_candidate(exp_dir)

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["params"], [1.0, 2.0, 3.0])
            self.assertEqual(candidate["params_source"], "matched_drsr_candidate")

    def test_recover_drsr_timeout_payload_uses_backfilled_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: beta
    - name: alpha
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "beta,alpha,y\n1,2,9\n2,3,14\n",
                "valid.csv": "beta,alpha,y\n3,4,19\n",
                "id_test.csv": "beta,alpha,y\n4,5,24\n",
                "ood_test.csv": "beta,alpha,y\n5,6,29\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            exp_dir = root / "drsr_exp"
            samples_dir = exp_dir / "samples"
            best_history_dir = exp_dir / "best_history"
            samples_dir.mkdir(parents=True)
            best_history_dir.mkdir(parents=True)
            function = (
                "def equation(beta, alpha, params):\n"
                "    return params[0] + params[1] * beta + params[2] * alpha\n"
            )
            (samples_dir / "top01_samples_0.json").write_text(
                json.dumps({"function": function, "score": 2.0}),
                encoding="utf-8",
            )
            (best_history_dir / "best_sample_0.json").write_text(
                json.dumps({"function": function, "score": 1.0, "params": [1.0, 2.0, 3.0]}),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._recover_timeout_payload_from_candidate(
                tool_name="drsr",
                dataset=dataset,
                experiment_dir=exp_dir,
            )

            self.assertIsNotNone(payload)
            self.assertIsNone(payload["canonical_artifact_error"])
            self.assertEqual(payload["canonical_artifact"]["parameter_values"], [1.0, 2.0, 3.0])
            self.assertEqual(payload["canonical_artifact"]["normalized_expression"], "c0 + c1*x0 + c2*x1")
            self.assertEqual(payload["id_metrics"]["nmse"], 0.0)
            self.assertEqual(payload["ood_metrics"]["nmse"], 0.0)

    def test_build_runner_params_can_disable_prompt_semantics_for_llmsr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  description: hidden semantic description
  target:
    name: y
    description: hidden target
  features:
    - name: x0
      description: hidden feature
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n", encoding="utf-8")

            dataset = runner.load_canonical_dataset(dataset_dir)
            params = runner.build_runner_params(
                "llmsr",
                dataset,
                root / "bench_results",
                seed=1314,
                params_override={"inject_prompt_semantics": False},
            )

            self.assertFalse(params["inject_prompt_semantics"])
            self.assertNotIn("metadata_path", params)
            self.assertNotIn("feature_descriptions", params)
            self.assertNotIn("target_description", params)
            self.assertEqual(
                params["background"],
                "This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.",
            )

    def test_build_runner_params_injects_explicit_dataset_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  description: explicit contract dataset
  target:
    name: output
  features:
    - name: feature_a
    - name: feature_b
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("feature_a,feature_b,output\n1,2,3\n", encoding="utf-8")

            dataset = runner.load_canonical_dataset(dataset_dir)
            params = runner.build_runner_params(
                "gplearn",
                dataset,
                root / "bench_results",
                seed=1314,
            )

            self.assertEqual(params["n_features"], 2)
            self.assertEqual(params["feature_names"], ["feature_a", "feature_b"])
            self.assertEqual(params["target_name"], "output")


if __name__ == "__main__":
    unittest.main()
