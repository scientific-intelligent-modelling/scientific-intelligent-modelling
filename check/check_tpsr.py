import os
import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))
TPSR_ROOT = os.path.join(
    REPO_ROOT,
    "scientific_intelligent_modelling",
    "algorithms",
    "tpsr_wrapper",
    "tpsr",
)
SHARED_E2E_MODEL = os.path.join(
    REPO_ROOT,
    "scientific_intelligent_modelling",
    "algorithms",
    "e2esr_wrapper",
    "model.pt",
)


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _resolve_path(rel_path: str):
    if not rel_path:
        return None
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(TPSR_ROOT, rel_path)


def _model_file(backbone: str):
    if backbone == "e2e":
        return SHARED_E2E_MODEL
    if backbone == "nesymres":
        default_candidates = [
            _resolve_path(os.path.join("nesymres", "weights", "10MCompleted.ckpt")),
            _resolve_path(os.path.join("nesymres", "weights", "10M.ckpt")),
        ]
        for p in default_candidates:
            if os.path.isfile(p):
                return p
        return default_candidates[0]
    return None


def _assert_pretrained_file(backbone: str):
    path = _model_file(backbone)
    env_key = "TPSR_E2E_MODEL_PATH" if backbone == "e2e" else "TPSR_NESYMRES_MODEL_PATH"
    env_path = os.getenv(env_key)
    if env_path:
        path = env_path
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            f"未找到 {backbone} 预训练权重。请在运行前设置 {env_key} 指向可用 ckpt。"
        )
    return path


def run_backbone(backbone: str):
    rng = np.random.RandomState(123)
    X = rng.rand(24, 2)
    y = 0.5 * X[:, 0] + 2.0 * X[:, 1] + 0.01 * rng.randn(24)

    reg = SymbolicRegressor(
        "tpsr",
        symbolicregression_model_path=_assert_pretrained_file("e2e") if backbone == "e2e" else None,
        nesymres_model_path=_assert_pretrained_file("nesymres") if backbone == "nesymres" else None,
        backbone_model=backbone,
        max_input_points=80,
        beam_size=6,
        num_beams=1,
        width=3,
        rollout=2,
        horizon=20,
        no_seq_cache=True,
        no_prefix_cache=True,
        force_cpu=True,
    )

    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])

    assert_ok(isinstance(eq, str) and len(eq) > 0, f"{backbone}: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, f"{backbone}: 方程列表为空")
    assert_ok(pred.shape == (4,), f"{backbone}: 预测形状异常 {pred.shape}")

    print(f"[check_tpsr] {backbone} OK")
    print(f"eq: {eq}")
    print(f"equations: {len(equations)}")
    print(f"pred_head: {pred[:3].tolist()}")


def run():
    run_backbone("e2e")
    # run_backbone("nesymres")


if __name__ == "__main__":
    run()
