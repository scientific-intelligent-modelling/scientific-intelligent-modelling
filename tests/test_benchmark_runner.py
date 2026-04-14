import json
import tempfile
import unittest
from pathlib import Path

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


class BenchmarkRunnerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
