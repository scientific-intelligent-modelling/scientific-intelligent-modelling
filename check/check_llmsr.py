import glob
import json
import os
import numpy as np

from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import LLMSRRegressor
from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import _llmsr_root_dir
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def latest_llmsr_experiment():
    root = os.path.join(_llmsr_root_dir(), "experiments")
    candidates = sorted(glob.glob(os.path.join(root, "*")))
    for p in reversed(candidates):
        meta = os.path.join(p, "meta.json")
        samples = os.path.join(p, "samples")
        if os.path.isfile(meta) and os.path.isdir(samples):
            return p
    return None


def get_feature_count(exp_dir: str) -> int:
    with open(os.path.join(exp_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta.get("feature_names") or []
    return len(feature_names)


def has_api_credentials() -> bool:
    # 仅在用户显式开启在线模式时才尝试联网，避免在 CI/无网络环境下误触发 API 调用。
    if os.getenv("LLMSR_ALLOW_ONLINE", "").lower() not in {"1", "true", "yes", "on"}:
        return False

    for k in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "BLT_API_KEY", "OPENAI_API_KEY"):
        v = os.getenv(k)
        if v and str(v).strip():
            return True
    return False


def run_offline_replay():
    exp_dir = latest_llmsr_experiment()
    if not exp_dir:
        raise RuntimeError("未找到可复用的 llmsr 实验目录，无法执行离线验收")

    n_features = get_feature_count(exp_dir)
    assert_ok(n_features > 0, "离线验收: 实验元信息未包含 feature_names")

    reg = LLMSRRegressor(existing_exp_dir=exp_dir)
    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()

    rng = np.random.RandomState(1)
    X = rng.rand(5, n_features)
    pred = reg.predict(X)

    assert_ok(isinstance(eq, str) and len(eq) > 0, "llmsr 离线验收: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "llmsr 离线验收: 候选方程列表为空")
    assert_ok(pred.shape == (5,), f"llmsr 离线验收: 预测形状异常 {pred.shape}")

    print("[check_llmsr] offline OK")
    print("equation_head:", eq[:120])
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


def run_online_smoke():
    if not has_api_credentials():
        print("[check_llmsr] skip online: 未检测到 API Key，已跳过 llmsr 在线闭环")
        return

    rng = np.random.RandomState(2)
    X = rng.rand(20, 2)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.1 * rng.randn(20)

    reg = SymbolicRegressor(
        "llmsr",
        problem_name="ci_llmsr_check",
        background="y 与 x0,x1 的线性关系：y = 2*x0 - 3*x1",
        niterations=20,
        samples_per_iteration=1,
    )

    reg.fit(X, y)
    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:3])

    assert_ok(isinstance(eq, str), "llmsr 在线验收: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "llmsr 在线验收: 候选方程列表为空")
    assert_ok(pred.shape == (3,), f"llmsr 在线验收: 预测形状异常 {pred.shape}")

    print("[check_llmsr] online OK")
    print("equation_head:", eq[:120])
    print("equations:", len(equations))
    print("pred_head:", pred.tolist())


def run():
    # 离线复用既有实验，保证本地环境下可复现
    run_offline_replay()
    # 在线真实训练仅在有凭证时执行，避免 CI/本地无 API Key 时误报失败
    run_online_smoke()


if __name__ == "__main__":
    run()
