import glob
import os

import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


DRSR_EXP_ROOT = os.path.join(
    "scientific_intelligent_modelling",
    "algorithms",
    "drsr_wrapper",
    "drsr",
    "experiments",
)


def assert_ok(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def latest_drsr_experiment(problem_prefix: str = "oscillator1") -> str | None:
    candidates = sorted(glob.glob(os.path.join(DRSR_EXP_ROOT, "*")))
    for p in reversed(candidates):
        if problem_prefix and problem_prefix not in os.path.basename(p):
            continue
        if os.path.isfile(os.path.join(p, "experiences.json")):
            return p
    return None


def load_oscillator_data():
    data_path = os.path.join(
        "scientific_intelligent_modelling",
        "algorithms",
        "drsr_wrapper",
        "drsr",
        "data",
        "oscillator1",
        "train.csv",
    )
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    X = data[:, :2]
    y = data[:, 2]
    return X, y


def has_api_credentials() -> bool:
    if os.getenv("DRSR_ALLOW_ONLINE", "").lower() not in {"1", "true", "yes", "on"}:
        return False

    for k in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "BLT_API_KEY", "OPENAI_API_KEY"):
        v = os.getenv(k)
        if v and str(v).strip():
            return True
    return False


def run_offline_replay():
    exp_dir = latest_drsr_experiment("oscillator1")
    if not exp_dir:
        raise RuntimeError("未找到可复用的 DRSR 实验目录，无法执行离线验收")

    X, y = load_oscillator_data()
    reg = SymbolicRegressor(
        "drsr",
        existing_exp_dir=exp_dir,
        problem_name="ci_drsr_replay",
        max_samples=1,
    )
    reg.fit(X[:120], y[:120])

    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:4])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "drsr 离线验收: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "drsr 离线验收: 方程列表为空")
    assert_ok(pred.shape == (4,), f"drsr 离线验收: 预测形状异常 {pred.shape}")

    print("[check_drsr] offline OK")
    print("equation_head:", eq[:160])
    print("equations:", len(equations))
    print("pred_head:", pred[:3].tolist())


def run_online_smoke():
    if not has_api_credentials():
        print("[check_drsr] skip online: 未检测到 API Key，已跳过 DRSR 在线闭环")
        return

    rng = np.random.RandomState(3)
    X = rng.rand(24, 2)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.05 * rng.randn(24)

    reg = SymbolicRegressor(
        "drsr",
        problem_name="ci_drsr_check",
        background="y 与 x0, x1 的线性关系：y = 2*x0 - 3*x1",
        use_api=True,
        api_model="blt/gpt-4o-mini",
        max_samples=20,
        samples_per_prompt=1,
        evaluate_timeout_seconds=10,
    )

    reg.fit(X, y)
    eq = reg.get_optimal_equation()
    equations = reg.get_total_equations()
    pred = reg.predict(X[:3])

    assert_ok(isinstance(eq, str) and len(eq) > 0, "drsr 在线验收: 最优方程为空")
    assert_ok(isinstance(equations, list) and len(equations) > 0, "drsr 在线验收: 方程列表为空")
    assert_ok(pred.shape == (3,), f"drsr 在线验收: 预测形状异常 {pred.shape}")

    print("[check_drsr] online OK")
    print("equation_head:", eq[:160])
    print("equations:", len(equations))
    print("pred_head:", pred.tolist())


def run():
    run_offline_replay()
    run_online_smoke()


if __name__ == "__main__":
    run()
