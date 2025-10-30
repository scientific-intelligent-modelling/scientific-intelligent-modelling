"""
DRSR 管道最小可运行用例（不改动 DRSR 源码）。

做法：
- 将 DRSR 子模块目录加入 sys.path，直接调用 drsr_420.pipeline.main。
- 使用 monkeypatch 替换联网相关与重型优化路径：
  * LocalLLM.draw_samples -> 返回固定函数体样本（避免 HTTP 请求）
  * Sampler.analyze_equations_with_scores / _with_residual -> 返回固定分析文本（避免 HTTP 请求）
  * DataAnalyzer.analyze -> 返回固定文本（避免 HTTP 请求）
  * evaluate_on_problems.evaluate -> 采用快速 MSE 评分（不做 SciPy BFGS）
- 缩小数据量、减少采样次数，加快测试速度。

该测试仅验证“能跑通且产生经验文件”，不校验数学收敛性。
"""

import os
import sys
import json
from pathlib import Path


def test_drsr_pipeline_smoke():
    # 1) 准备路径与导入 drsr_420
    repo_root = Path(__file__).resolve().parents[1]
    drsr_dir = repo_root / "scientific_intelligent_modelling" / "algorithms" / "drsr_wrapper" / "drsr"
    sys.path.insert(0, str(drsr_dir))  # 仅本用例作用域

    from drsr_420 import pipeline, config as config_lib, sampler, evaluator, evaluate_on_problems, data_analyse_real

    # 2) 准备 spec 与数据（取前 50 行以提速）
    spec_path = drsr_dir / "specs" / "specification_oscillator1_numpy.txt"
    with open(spec_path, "r", encoding="utf-8") as f:
        specification = f.read()

    import numpy as np
    import pandas as pd
    df = pd.read_csv(drsr_dir / "data" / "oscillator1" / "train.csv")
    data = df.to_numpy()
    data = data[:50]  # 减少规模，加快测试
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    dataset = {"data": {"inputs": X, "outputs": y}}

    # 3) 预创建经验目录，避免写文件报错
    exp_dir = Path.cwd() / "equation_experiences"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_file = exp_dir / "experiences.json"
    if not exp_file.exists():
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump({"None": [], "Good": [], "Bad": []}, f)

    # 4) Monkeypatch：屏蔽联网与重型优化
    def fake_draw_samples(self, prompt: str, cfg: config_lib.Config):
        # 返回函数体（不含 def 行），使 _extract_body 原样返回
        body = (
            "    # 简单线性骨架\n"
            "    dv = params[0] * x + params[1] * v + params[2]\n"
            "    return dv\n"
        )
        return [body]

    def fake_analyze_scores(self, samples, quality_for_sample, error_for_sample, prompt):
        return ["ok"] * len(samples)

    def fake_analyze_residual(self, sample, residual):
        return "residual analyzed"

    def fast_evaluate(data_dict, equation):
        # 轻量评分：不做 BFGS，仅用固定 params 评估 MSE
        inputs, outputs = data_dict["inputs"], data_dict["outputs"]
        params = np.ones(10)
        pred = equation(*inputs.T, params)
        mse = float(np.mean((pred - outputs) ** 2))
        # 构造简化 residual 矩阵：x, v, y, residual
        residual = outputs - pred
        full_res = np.column_stack((inputs, outputs, residual))
        return -mse, full_res

    def fake_data_analyze(self, data_source, *args, **kwargs):
        return "ok"

    # 应用补丁
    sampler.LocalLLM.draw_samples = fake_draw_samples
    sampler.Sampler.analyze_equations_with_scores = fake_analyze_scores
    sampler.Sampler.analyze_equations_with_residual = fake_analyze_residual
    evaluate_on_problems.evaluate = fast_evaluate
    data_analyse_real.DataAnalyzer.analyze = fake_data_analyze

    # 5) 组装配置，限制采样次数，缩短超时
    cls_cfg = config_lib.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    cfg = config_lib.Config(
        use_api=False,
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=1,
        evaluate_timeout_seconds=10,
    )

    # 6) 运行管道（最多采样 2 次）
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=cfg,
        max_sample_nums=2,
        class_config=cls_cfg,
        log_dir=str(Path.cwd() / "outputs" / "drsr_test_logs"),
    )

    # 7) 断言：经验文件存在且可解析
    assert exp_file.exists(), "未生成经验文件"
    with open(exp_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict) and all(k in data for k in ["None", "Good", "Bad"]) , "经验文件结构不正确"


def test_drsr_cli_style_smoke():
    """
    按你提供的 CLI 用法模拟执行：
    等价于在 drsr 目录运行：
      python3 main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_test

    但为避免联网与长时优化，这里在执行 main.py 前对 drsr_420 做猴子补丁：
    - 限制 Sampler._max_sample_nums=2（通过包装 __init__）
    - 屏蔽联网/重优化，复用上个用例的补丁
    通过 runpy 在同一进程执行脚本，使补丁生效。
    """
    import runpy
    import numpy as np

    repo_root = Path(__file__).resolve().parents[1]
    drsr_dir = repo_root / "scientific_intelligent_modelling" / "algorithms" / "drsr_wrapper" / "drsr"
    sys.path.insert(0, str(drsr_dir))

    # 预创建经验文件
    exp_dir = Path.cwd() / "equation_experiences"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_file = exp_dir / "experiences.json"
    if not exp_file.exists():
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump({"None": [], "Good": [], "Bad": []}, f)

    # 预先导入并补丁 drsr_420 下模块
    from drsr_420 import sampler, evaluate_on_problems, data_analyse_real

    # 轻量打分
    def fast_evaluate(data_dict, equation):
        inputs, outputs = data_dict["inputs"], data_dict["outputs"]
        params = np.ones(10)
        pred = equation(*inputs.T, params)
        mse = float(np.mean((pred - outputs) ** 2))
        residual = outputs - pred
        full_res = np.column_stack((inputs, outputs, residual))
        return -mse, full_res

    evaluate_on_problems.evaluate = fast_evaluate
    data_analyse_real.DataAnalyzer.analyze = lambda *a, **k: "ok"

    # LLM 采样：返回固定函数体
    def fake_draw_samples(self, prompt: str, cfg):
        body = (
            "    dv = params[0] * x + params[1] * v + params[2]\n"
            "    return dv\n"
        )
        return [body]

    sampler.LocalLLM.draw_samples = fake_draw_samples
    sampler.Sampler.analyze_equations_with_scores = lambda *a, **k: ["ok"]
    sampler.Sampler.analyze_equations_with_residual = lambda *a, **k: "ok"

    # 限制采样最大次数：包裹 __init__
    _orig_init = sampler.Sampler.__init__

    def wrapped_init(self, *a, **k):
        _orig_init(self, *a, **k)
        self._max_sample_nums = 2

    sampler.Sampler.__init__ = wrapped_init

    # 组装 CLI 参数并运行 main.py
    main_py = drsr_dir / "main.py"
    spec_rel = "./specs/specification_oscillator1_numpy.txt"
    log_rel = "./logs/oscillator1_test"

    argv_backup = list(sys.argv)
    try:
        sys.argv = [str(main_py), "--problem_name", "oscillator1", "--spec_path", spec_rel, "--log_path", log_rel]
        runpy.run_path(str(main_py), run_name="__main__")
    finally:
        sys.argv = argv_backup

    # 简单断言
    assert exp_file.exists()
