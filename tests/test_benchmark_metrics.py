import importlib.util
import unittest

import numpy as np

from scientific_intelligent_modelling.benchmarks.metrics import (
    acc_within_threshold,
    llm_srbench_acc_tau,
    llm_srbench_nmse,
    llm_srbench_numeric_metrics,
    normalized_tree_edit_distance,
    regression_metrics,
    srbench_model_size,
    srbench_symbolic_solution,
)


HAS_SYMPY = importlib.util.find_spec("sympy") is not None


class BenchmarkMetricsTest(unittest.TestCase):
    def test_regression_metrics_basic_fields(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.05, 2.95])
        metrics = regression_metrics(y_true, y_pred, acc_threshold=0.1)
        self.assertIsNotNone(metrics["rmse"])
        self.assertIsNotNone(metrics["r2"])
        self.assertIsNotNone(metrics["nmse"])
        self.assertAlmostEqual(metrics["acc_tau"], 1.0)

    @unittest.skipUnless(HAS_SYMPY, "需要 sympy")
    def test_srbench_symbolic_solution_additive_constant(self):
        result = srbench_symbolic_solution("x + 2", "x + 5")
        self.assertTrue(result["is_symbolic_solution"])
        self.assertEqual(result["relation"], "additive_constant")

    @unittest.skipUnless(HAS_SYMPY, "需要 sympy")
    def test_srbench_symbolic_solution_scalar_multiple(self):
        result = srbench_symbolic_solution("2*x", "6*x")
        self.assertTrue(result["is_symbolic_solution"])
        self.assertEqual(result["relation"], "scalar_multiple")

    @unittest.skipUnless(HAS_SYMPY, "需要 sympy")
    def test_srbench_symbolic_solution_reject_constant_prediction(self):
        result = srbench_symbolic_solution("3", "x + 1")
        self.assertFalse(result["is_symbolic_solution"])
        self.assertEqual(result["relation"], "constant_prediction")

    @unittest.skipUnless(HAS_SYMPY, "需要 sympy")
    def test_srbench_model_size_counts_tokens(self):
        result = srbench_model_size("x + 2*y")
        self.assertGreaterEqual(result["operators"], 2)
        self.assertEqual(result["features"], 2)
        self.assertGreaterEqual(result["constants"], 1)
        self.assertEqual(
            result["size"],
            result["operators"] + result["features"] + result["constants"],
        )

    @unittest.skipUnless(HAS_SYMPY, "需要 sympy")
    def test_srsd_ned_zero_for_equivalent_simplified_expressions(self):
        result = normalized_tree_edit_distance("x + x + x", "3*x")
        self.assertEqual(result["ned"], 0.0)

    @unittest.skipUnless(HAS_SYMPY, "需要 sympy")
    def test_srsd_ned_detects_structural_difference(self):
        result = normalized_tree_edit_distance("x + y", "x * y")
        self.assertGreater(result["ned"], 0.0)

    def test_acc_within_threshold(self):
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([0.05, 1.02, 1.85, 3.2])
        score = acc_within_threshold(y_true, y_pred, threshold=0.1)
        self.assertAlmostEqual(score, 0.5)

    def test_llm_srbench_acc_tau_is_sequence_level(self):
        y_true = np.array([1.0, 2.0, 4.0])
        y_pred = np.array([1.01, 2.02, 4.03])
        self.assertEqual(llm_srbench_acc_tau(y_true, y_pred, tau=0.1), 1.0)
        y_pred_bad = np.array([1.01, 2.02, 5.0])
        self.assertEqual(llm_srbench_acc_tau(y_true, y_pred_bad, tau=0.1), 0.0)

    def test_llm_srbench_nmse_matches_variance_normalization(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = 1.0 / 2.0
        self.assertAlmostEqual(llm_srbench_nmse(y_true, y_pred), expected)

    def test_llm_srbench_numeric_metrics_bundle(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = llm_srbench_numeric_metrics(y_true, y_pred, tau=0.1)
        self.assertEqual(metrics["acc_tau"], 1.0)
        self.assertEqual(metrics["nmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
