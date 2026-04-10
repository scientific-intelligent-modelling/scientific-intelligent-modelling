import unittest
import sys
import types

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import (
    DRSRRegressor,
)

if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.ModuleType("pandas")

from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import (
    _is_single_line_formula_function,
)


class SingleLineFormulaFiltersTest(unittest.TestCase):
    def test_llmsr_accepts_single_line_return_function(self):
        func = (
            "def equation(x0, x1, params):\n"
            "    return params[0] + params[1] * x0 + params[2] * x1\n"
        )
        self.assertTrue(_is_single_line_formula_function(func))

    def test_llmsr_rejects_multiple_statements(self):
        func = (
            "def equation(x0, x1, params):\n"
            "    tmp = params[0] + x0\n"
            "    return tmp + x1\n"
        )
        self.assertFalse(_is_single_line_formula_function(func))

    def test_llmsr_rejects_multiline_return(self):
        func = (
            "def equation(x0, x1, params):\n"
            "    return (\n"
            "        params[0] + x0 + x1\n"
            "    )\n"
        )
        self.assertFalse(_is_single_line_formula_function(func))

    def test_drsr_accepts_single_line_return_body(self):
        body = "    return params[0] + params[1] * col0 + params[2] * col1\n"
        self.assertEqual(
            DRSRRegressor._clean_equation_body(body),
            "return params[0] + params[1] * col0 + params[2] * col1\n",
        )

    def test_drsr_rejects_assignment_then_return(self):
        body = "tmp = params[0] + col0\nreturn tmp + col1\n"
        self.assertEqual(DRSRRegressor._clean_equation_body(body), "")

    def test_drsr_rejects_multiline_return(self):
        body = "return (\n    params[0] + col0 + col1\n)\n"
        self.assertEqual(DRSRRegressor._clean_equation_body(body), "")


if __name__ == "__main__":
    unittest.main()
