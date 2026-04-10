import sys
import types
import unittest

from scientific_intelligent_modelling.benchmarks.normalizers import (
    normalize_drsr_artifact,
    normalize_e2esr_artifact,
    normalize_gplearn_artifact,
    normalize_llmsr_artifact,
    normalize_operon_artifact,
    normalize_pysr_artifact,
    normalize_qlattice_artifact,
    normalize_tpsr_artifact,
)


if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.ModuleType("pandas")
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor
from scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper import E2ESRRegressor
from scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper import GPLearnRegressor
from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import LLMSRRegressor
from scientific_intelligent_modelling.algorithms.operon_wrapper.wrapper import OperonRegressor
from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor
from scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper import QLatticeRegressor
from scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper import TPSRRegressor


class SymbolicNormalizersTest(unittest.TestCase):
    def test_normalize_pysr_artifact(self):
        artifact = normalize_pysr_artifact("x0 + 2*x1")
        self.assertEqual(artifact["tool_name"], "pysr")
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_qlattice_artifact(self):
        artifact = normalize_qlattice_artifact("x0 + 2*x1")
        self.assertEqual(artifact["tool_name"], "QLattice")
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_gplearn_artifact(self):
        artifact = normalize_gplearn_artifact("add(X0, mul(X1, X1))")
        self.assertEqual(artifact["tool_name"], "gplearn")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_e2esr_artifact(self):
        artifact = normalize_e2esr_artifact("x_0 + x_1**2")
        self.assertEqual(artifact["tool_name"], "e2esr")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_tpsr_artifact(self):
        artifact = normalize_tpsr_artifact("x_0 + x_1**2")
        self.assertEqual(artifact["tool_name"], "tpsr")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_tpsr_artifact_marks_variable_mismatch(self):
        artifact = normalize_tpsr_artifact("x_0 + x_2**2", expected_n_features=2)
        self.assertTrue(artifact["sympy_parse_ok"])
        self.assertFalse(artifact["artifact_valid"])
        self.assertTrue(artifact["validation_errors"])

    def test_normalize_operon_artifact(self):
        artifact = normalize_operon_artifact("X1 + X2^2")
        self.assertEqual(artifact["tool_name"], "pyoperon")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_llmsr_artifact(self):
        raw = (
            "def equation(x0, x1, params):\n"
            "    return params[0] + params[1] * x0 + params[2] * x1\n"
        )
        artifact = normalize_llmsr_artifact(raw, parameter_values=[1.0, 2.0, 3.0])
        self.assertEqual(artifact["tool_name"], "llmsr")
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")
        self.assertEqual(artifact["parameter_symbols"], ["c0", "c1", "c2"])
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_drsr_artifact(self):
        raw = (
            "def equation(col0, col1, params):\n"
            "    return params[0] + params[1] * col0 + params[2] * col1\n"
        )
        artifact = normalize_drsr_artifact(raw, parameter_values=[1.0, 2.0, 3.0])
        self.assertEqual(artifact["tool_name"], "drsr")
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_drsr_legacy_xyv_artifact(self):
        raw = (
            "def equation(col0, col1, params):\n"
            "    return params[0] * x + params[1] * v + params[2]\n"
        )
        artifact = normalize_drsr_artifact(raw, parameter_values=[1.0, 2.0, 3.0])
        self.assertEqual(artifact["normalized_expression"], "c0*x0 + c1*x1 + c2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_wrapper_export_pysr(self):
        model = PySRRegressor()
        model.model = object()
        model.get_optimal_equation = lambda: "x0 + 2*x1"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["tool_name"], "pysr")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_wrapper_export_qlattice(self):
        model = QLatticeRegressor()
        model.model = True
        model.get_optimal_equation = lambda: "x0 + 2*x1"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["tool_name"], "QLattice")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_wrapper_export_gplearn(self):
        model = GPLearnRegressor()
        model.model = object()
        model.get_optimal_equation = lambda: "add(X0, mul(X1, X1))"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_e2esr(self):
        model = E2ESRRegressor.__new__(E2ESRRegressor)
        model.best_tree = object()
        model.get_optimal_equation = lambda: "x_0 + x_1**2"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_tpsr(self):
        model = TPSRRegressor()
        model.best_tree = "x_0 + x_1**2"
        model._n_features = 2
        model.get_optimal_equation = lambda: "x_0 + x_1**2"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_tpsr_marks_variable_mismatch(self):
        model = TPSRRegressor()
        model.best_tree = "x_0 + x_2**2"
        model._n_features = 2
        model.get_optimal_equation = lambda: "x_0 + x_2**2"
        artifact = model.export_canonical_symbolic_program()
        self.assertFalse(artifact["artifact_valid"])
        self.assertTrue(artifact["validation_errors"])

    def test_wrapper_export_operon(self):
        model = OperonRegressor()
        model.best_model_str = "X1 + X2^2"
        model.model = None
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_llmsr(self):
        model = LLMSRRegressor()
        model._load_best_sample = lambda: {
            "function": (
                "def equation(x0, x1, params):\n"
                "    return params[0] + params[1] * x0 + params[2] * x1\n"
            ),
            "params": [1.0, 2.0, 3.0],
        }
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")

    def test_wrapper_export_drsr(self):
        model = DRSRRegressor()
        model.get_optimal_equation = lambda: (
            "def equation(col0, col1, params):\n"
            "    return params[0] + params[1] * col0 + params[2] * col1\n"
        )
        model.get_fitted_params = lambda: [1.0, 2.0, 3.0]
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")


if __name__ == "__main__":
    unittest.main()
