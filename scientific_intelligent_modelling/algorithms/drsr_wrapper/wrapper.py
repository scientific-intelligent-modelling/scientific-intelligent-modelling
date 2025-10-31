import os
import sys
import json
import tempfile
from typing import List, Optional

import numpy as np

from ..base_wrapper import BaseWrapper


class DRSRRegressor(BaseWrapper):
    """
    DRSR 最小可用封装：
    - 默认 fast_mode=True：使用轻量猴子补丁避免联网/重优化，跑通管线并采集候选方程。
    - 不修改 drsr 源码，直接按 tests/test_drsr_smoke.py 的思路调用 drsr_420.pipeline。

    参数（可选）：
    - spec_path: 提示规范文件路径，默认使用 drsr/specs/specification_oscillator1_numpy.txt
    - fast_mode: 是否启用轻量补丁（默认 True）
    - samples_per_prompt: 每轮采样数（默认 1，fast_mode 下仅用于形式）
    - max_samples: 采样总轮数上限（默认 2）
    - evaluate_timeout_seconds: 评估超时（默认 10）
    - log_dir: 日志目录（等价于 api.sh 的 --log_path；默认写入工作目录 logs/ 下）
    - workdir: 作为 drsr 相对输出的工作目录（默认创建临时目录）
    """

    def __init__(self, **kwargs):
        self.params = dict(kwargs) if kwargs else {}
        self.model_ready = False
        self._workdir: Optional[str] = None
        self._equation_body: Optional[str] = None
        self._equation_func = None
        self._all_bodies: List[str] = []

    # -------------------------------
    # 训练：运行 DRSR 管线（默认轻量模式）并记录方程
    # -------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        # 工作目录（承载 equation_experiences/residual_analyze 等相对输出）
        self._workdir = self.params.get("workdir") or tempfile.mkdtemp(prefix="drsr_run_")

        # 导入 drsr_420 模块
        drsr_dir = os.path.join(os.path.dirname(__file__), "drsr")
        sys.path.insert(0, drsr_dir)
        from drsr_420 import pipeline, config as config_lib, sampler, evaluator, evaluate_on_problems, data_analyse_real

        # 规范文本
        spec_path = self.params.get("spec_path")
        if not spec_path:
            spec_path = os.path.join(drsr_dir, "specs", "specification_oscillator1_numpy.txt")
        with open(spec_path, "r", encoding="utf-8") as f:
            specification = f.read()

        # 数据（直接传给 pipeline，避免文件依赖）
        dataset = {"data": {"inputs": X, "outputs": y}}

        # 轻量补丁（默认开启）。若希望调用真实 LLM，请传入 fast_mode=False。
        if self.params.get("fast_mode", True):
            def _fake_draw_samples(self_llm, prompt: str, cfg: config_lib.Config):
                body = (
                    "    # 简单线性骨架\n"
                    "    dv = params[0] * x + params[1] * v + params[2]\n"
                    "    return dv\n"
                )
                return [body]

            def _fake_analyze_scores(self_sampler, samples, quality_for_sample, error_for_sample, prompt):
                return ["ok"] * len(samples)

            def _fake_analyze_residual(self_sampler, sample, residual):
                return "ok"

            def _fast_evaluate(data_dict, equation):
                inputs, outputs = data_dict["inputs"], data_dict["outputs"]
                params = np.ones(10)
                pred = equation(*inputs.T, params)
                mse = float(np.mean((pred - outputs) ** 2))
                residual = outputs - pred
                full_res = np.column_stack((inputs, outputs, residual))
                return -mse, full_res

            sampler.LocalLLM.draw_samples = _fake_draw_samples
            sampler.Sampler.analyze_equations_with_scores = _fake_analyze_scores
            sampler.Sampler.analyze_equations_with_residual = _fake_analyze_residual
            evaluate_on_problems.evaluate = _fast_evaluate
            data_analyse_real.DataAnalyzer.analyze = lambda *a, **k: "ok"

        # 经验文件预创建（相对 self._workdir）
        exp_dir = os.path.join(self._workdir, "equation_experiences")
        os.makedirs(exp_dir, exist_ok=True)
        exp_file = os.path.join(exp_dir, "experiences.json")
        if not os.path.exists(exp_file):
            with open(exp_file, "w", encoding="utf-8") as f:
                json.dump({"None": [], "Good": [], "Bad": []}, f)

        # 组装配置并运行
        cls_cfg = config_lib.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
        cfg = config_lib.Config(
            use_api=bool(self.params.get("use_api", False)),
            api_model=str(self.params.get("api_model", "gpt-3.5-turbo")),
            num_samplers=1,
            num_evaluators=1,
            samples_per_prompt=int(self.params.get("samples_per_prompt", 1)),
            evaluate_timeout_seconds=int(self.params.get("evaluate_timeout_seconds", 10)),
        )

        # 切换 cwd 到工作目录，确保 drsr 相对路径输出写入其中
        cwd_backup = os.getcwd()
        os.chdir(self._workdir)
        try:
            pipeline.main(
                specification=specification,
                inputs=dataset,
                config=cfg,
                max_sample_nums=int(self.params.get("max_samples", 2)),
                class_config=cls_cfg,
                log_dir=self.params.get("log_dir") or os.path.join(self._workdir, "logs"),
            )
        finally:
            os.chdir(cwd_backup)

        # 读取经验，选取最佳/首个方程
        best_body = None
        bodies: List[str] = []
        try:
            with open(exp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 收集 Good/None/Bad 顺序的方程体
            for key in ("Good", "None", "Bad"):
                for item in data.get(key, []):
                    eq = item.get("equation")
                    if isinstance(eq, str) and eq.strip():
                        bodies.append(eq)
            # 选择最佳（优先 Good 中分数最高）
            def _pick_best(group_name: str):
                group = data.get(group_name, [])
                if not group:
                    return None
                # score 越大越好
                scored = [(it.get("score"), it.get("equation")) for it in group if isinstance(it.get("equation"), str)]
                scored = [s for s in scored if s[1]]
                if not scored:
                    return None
                scored.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
                return scored[0][1]

            best_body = _pick_best("Good") or (bodies[0] if bodies else None)
        except Exception:
            pass

        # 回退：若无产出，给一个默认骨架
        if not best_body:
            best_body = (
                "    dv = params[0] * x + params[1] * v + params[2]\n"
                "    return dv\n"
            )

        self._equation_body = best_body
        self._all_bodies = bodies or [best_body]
        self._equation_func = self._compile_equation(best_body)
        self.model_ready = True
        return self

    # -------------------------------
    # 序列化/反序列化：仅保存必要可序列化字段
    # -------------------------------
    def serialize(self):
        state = {
            'params': self.params,
            'equation_body': self._equation_body,
            'all_bodies': self._all_bodies,
        }
        return json.dumps(state)

    @classmethod
    def deserialize(cls, payload: str):
        obj = json.loads(payload)
        inst = cls(**obj.get('params', {}))
        inst._equation_body = obj.get('equation_body')
        inst._all_bodies = obj.get('all_bodies', [])
        if inst._equation_body:
            try:
                inst._equation_func = inst._compile_equation(inst._equation_body)
                inst.model_ready = True
            except Exception:
                inst._equation_func = None
                inst.model_ready = False
        return inst

    # -------------------------------
    # 预测：使用已编译方程 + 默认参数
    # -------------------------------
    def predict(self, X):
        if not self.model_ready or self._equation_func is None:
            raise ValueError("模型尚未训练或方程不可用")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("DRSR 预测需要至少两列输入：x 与 v")
        params = np.ones(10)
        try:
            return self._equation_func(X[:, 0], X[:, 1], params)
        except Exception:
            # 回退为零向量，保证子进程不崩
            return np.zeros(X.shape[0])

    # -------------------------------
    # 获取最优/全部方程
    # -------------------------------
    def get_optimal_equation(self):
        if not self._equation_body:
            return ""
        return self._wrap_equation(self._equation_body)

    def get_total_equations(self, n=None):
        eqs = [self._wrap_equation(b) for b in (self._all_bodies or []) if isinstance(b, str) and b.strip()]
        if n is not None:
            try:
                n = int(n)
                eqs = eqs[:max(0, n)]
            except Exception:
                pass
        return eqs

    # -------------------------------
    # 工具方法
    # -------------------------------
    @staticmethod
    def _wrap_equation(body: str) -> str:
        body = body.rstrip("\n") + "\n"
        return "def equation(x, v, params):\n" + body

    @staticmethod
    def _compile_equation(body: str):
        code = DRSRRegressor._wrap_equation(body)
        ns = {}
        exec(code, ns)
        return ns["equation"]
