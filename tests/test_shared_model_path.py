import os
import sys
import types
from pathlib import Path

def test_tpsr_prefers_shared_symbolic_model_path(tmp_path: Path):
    torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    sys.modules.setdefault("torch", torch_stub)

    from scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper import TPSRRegressor

    shared_model = tmp_path / "model.pt"
    shared_model.write_bytes(b"dummy")

    reg = TPSRRegressor(symbolicregression_model_path=str(shared_model))
    resolved = reg._ensure_e2e_model()

    assert resolved == str(shared_model.resolve())
    assert reg.params["symbolicregression_model_path"] == str(shared_model.resolve())
    assert os.environ["SIM_SYMBOLICREGRESSION_MODEL_PATH"] == str(shared_model.resolve())
