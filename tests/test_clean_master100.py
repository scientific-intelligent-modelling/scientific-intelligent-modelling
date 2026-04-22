import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("/home/family/workplace/scientific-intelligent-modelling/check/clean_master100.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("clean_master100_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_feynman_name_merges_separator_variants():
    mod = _load_module()
    assert mod.normalize_feynman_name("feynman-i.14.3") == "feynman.i.14.3"
    assert mod.normalize_feynman_name("feynman_I_14_3") == "feynman.i.14.3"
    assert mod.normalize_feynman_name("feynman_III_17_37") == "feynman.iii.17.37"


def test_canonicalize_formula_file_renames_args_to_canonical_positions(tmp_path):
    mod = _load_module()
    src_a = tmp_path / "a.py"
    src_b = tmp_path / "b.py"
    src_a.write_text("def f(m, g, z):\n    return m * g * z\n", encoding="utf-8")
    src_b.write_text("def y(alpha, beta, gamma):\n    return alpha * beta * gamma\n", encoding="utf-8")

    a = mod.canonicalize_formula_file(src_a)
    b = mod.canonicalize_formula_file(src_b)

    assert a["formula_parse_ok"] is True
    assert b["formula_parse_ok"] is True
    assert a["formula_hash"] is not None
    assert b["formula_hash"] is not None
    assert a["formula_normalized_expression"] == "x0*x1*x2"
    assert b["formula_normalized_expression"] == "x0*x1*x2"
    assert a["formula_hash"] == b["formula_hash"]
