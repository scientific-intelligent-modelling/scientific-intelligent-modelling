import os
import sys
import json
import tempfile
import time
import ast
from typing import List, Optional

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.srkit.llm import ClientFactory, parse_provider_model
from typing import Tuple
try:
    # 可选：用于参数拟合（与评估一致的 BFGS）
    from scipy.optimize import minimize
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


class DRSRRegressor(BaseWrapper):
    """
    DRSR 封装改造：
    - 优先改造"采样用的 LLM API"到统一的 llm.ClientFactory，其他执行逻辑尽量保持不变。

    关键参数（与 llmsr 封装对齐）：
    - spec_path: 规范文件（默认 drsr/specs/specification_oscillator1_numpy.txt）
    - samples_per_prompt: 每提示采样数（默认 1）
    - max_samples: 采样上限（默认 2）
    - evaluate_timeout_seconds: 评估超时（默认 10）
    - log_dir: 日志目录（默认写入 workdir/logs）
    - workdir: DRSR 相对输出目录（默认临时创建）
    - use_api, api_model, api_key, api_base, temperature, api_params: API 相关设置
    """

    def __init__(self, **kwargs):
        self.params = dict(kwargs) if kwargs else {}
        self.model_ready = False
        self._workdir: Optional[str] = None
        self._equation_body: Optional[str] = None
        self._equation_func = None
        self._all_bodies: List[str] = []
        self._best_params: Optional[np.ndarray] = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        # 工作目录（承载 equation_experiences/residual_analyze 等相对输出）
        # 默认：使用当前工作目录下的 `drsr_<problem>_<timestamp>`，可通过参数覆盖
        user_workdir = self.params.get("workdir")
        if user_workdir and str(user_workdir).strip():
            self._workdir = user_workdir
        else:
            default_base = os.getcwd()
            problem = str(self.params.get("problem_name") or "problem").strip()
            ts = time.strftime('%Y%m%d-%H%M%S')
            dir_name = f"drsr_{problem}_{ts}"
            # 默认前缀 outputs/
            self._workdir = os.path.join(default_base, "outputs", dir_name)
        os.makedirs(self._workdir, exist_ok=True)
        try:
            print(f"[DRSR] 使用工作目录: {os.path.abspath(self._workdir)}")
        except Exception:
            pass

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

        # 由 Wrapper 构建并注入单例 LLM 客户端；LocalLLM 仅负责 prompt 组织
        api_model = str(self.params.get('api_model'))
        provider, _model = parse_provider_model(api_model)
        api_key = self.params.get('api_key')
        if not api_key:
            if provider == 'deepseek':
                api_key = os.getenv('DEEPSEEK_API_KEY', '')
            elif provider in ('siliconflow', 'silicon-flow', 'sflow'):
                api_key = os.getenv('SILICONFLOW_API_KEY', '')
            elif provider in ('blt', 'bltcy', 'plato'):
                api_key = os.getenv('BLT_API_KEY', '')
            elif provider == 'ollama':
                api_key = ''
        client = ClientFactory.from_config({
            'model': api_model,
            'api_key': api_key,
            'base_url': self.params.get('api_base')
        })
        # 透传生成参数
        if self.params.get('temperature') is not None:
            client.kwargs['temperature'] = self.params['temperature']
        if isinstance(self.params.get('api_params'), dict):
            client.kwargs.update(self.params['api_params'])

        # 注入到 drsr 的 sampler 模块（共享使用）
        sampler.set_shared_llm_client(client)
        data_analyse_real.set_shared_llm_client(client)
        llm_class = sampler.LocalLLM

        # 经验文件预创建（相对 self._workdir）
        exp_dir = os.path.join(self._workdir, "equation_experiences")
        os.makedirs(exp_dir, exist_ok=True)
        exp_file = os.path.join(exp_dir, "experiences.json")
        if not os.path.exists(exp_file):
            with open(exp_file, "w", encoding="utf-8") as f:
                json.dump({"None": [], "Good": [], "Bad": []}, f)

        # 组装配置并运行
        cls_cfg = config_lib.ClassConfig(llm_class=llm_class, sandbox_class=evaluator.LocalSandbox)
        cfg = config_lib.Config(
            use_api=bool(self.params.get("use_api", False)),
            api_model=str(self.params.get("api_model", "deepseek/deepseek-chat")),
            num_samplers=1,
            num_evaluators=1,
            samples_per_prompt=int(self.params.get("samples_per_prompt", 1)),
            evaluate_timeout_seconds=int(self.params.get("evaluate_timeout_seconds", 10)),
            results_root=self._workdir,
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
            for key in ("Good", "None", "Bad"):
                for item in data.get(key, []):
                    eq = item.get("equation")
                    if isinstance(eq, str) and eq.strip():
                        bodies.append(eq)
            # 构建 entries（方程+训练期拟合参数+分数）并排序
            entries: List[dict] = []
            for key in ("Good", "Bad", "None"):
                for item in data.get(key, []):
                    eq = item.get("equation")
                    if not isinstance(eq, str) or not eq.strip():
                        continue
                    params = item.get("fitted_params")
                    try:
                        params = np.asarray(params).tolist() if params is not None else None
                    except Exception:
                        params = None
                    score = item.get("score")
                    entries.append({
                        'equation': eq,
                        'params': params,
                        'score': score,
                        'category': key,
                        'sample_order': item.get('sample_order')
                    })
            # 分数为 None 的排最后；分数越大越靠前
            entries.sort(key=lambda e: ((e['score'] is not None), e['score'] if e['score'] is not None else -1e18), reverse=True)
            self._equation_entries = entries
            def _pick_best(group_name: str):
                group = data.get(group_name, [])
                if not group:
                    return None
                scored = [(it.get("score"), it.get("equation")) for it in group if isinstance(it.get("equation"), str)]
                scored = [s for s in scored if s[1]]
                if not scored:
                    return None
                scored.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
                return scored[0][1]
            best_body = _pick_best("Good") or (bodies[0] if bodies else None)

            # 若经验中包含训练期拟合参数，优先取出
            best_params = None
            if best_body:
                for key in ("Good", "None", "Bad"):
                    for item in data.get(key, []):
                        if item.get("equation") == best_body and item.get("fitted_params") is not None:
                            try:
                                best_params = np.asarray(item.get("fitted_params"), dtype=float)
                                break
                            except Exception:
                                best_params = None
                    if best_params is not None:
                        break
        except Exception:
            pass

        if not best_body:
            best_body = (
                "    dv = params[0] * x + params[1] * v + params[2]\n"
                "    return dv\n"
            )

        self._equation_body = best_body
        self._all_bodies = bodies or [best_body]
        self._equation_func = self._compile_equation(best_body)
        # 注入训练期参数（若存在），否则置为空等待调用侧显式处理
        if 'best_params' in locals() and best_params is not None:
            self._best_params = best_params
        else:
            self._best_params = None
        self.model_ready = True
        return self

    def serialize(self):
        state = {
            'params': self.params,
            'equation_body': self._equation_body,
            'all_bodies': self._all_bodies,
            'best_params': self._best_params.tolist() if isinstance(self._best_params, np.ndarray) else None,
        }
        return json.dumps(state)

    @classmethod
    def deserialize(cls, payload: str):
        obj = json.loads(payload)
        inst = cls(**obj.get('params', {}))
        inst._equation_body = obj.get('equation_body')
        inst._all_bodies = obj.get('all_bodies', [])
        best_params = obj.get('best_params')
        inst._best_params = np.array(best_params) if best_params is not None else None
        if inst._equation_body:
            try:
                inst._equation_func = inst._compile_equation(inst._equation_body)
                inst.model_ready = True
            except Exception:
                inst._equation_func = None
                inst.model_ready = False
        return inst

    def predict(self, X):
        if not self.model_ready or self._equation_func is None:
            raise ValueError("模型尚未训练或方程不可用")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("DRSR 预测需要至少两列输入：x 与 v")
        if not isinstance(self._best_params, np.ndarray):
            raise RuntimeError("DRSRRegressor: 未找到训练期最优参数（fitted_params）。请检查 experiences.json 是否包含 fitted_params，或训练流程是否按期望运行。")
        params = self._best_params
        return self._equation_func(X[:, 0], X[:, 1], params)

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

    def get_total_equations_with_params(self, n: Optional[int] = None) -> List[dict]:
        """
        返回包含方程、训练期拟合参数、分数等的列表（按分数降序）。
        每个元素包含：{'equation': def字符串, 'params': List[float]|None, 'score': float|None, 'category': str, 'sample_order': int|None}
        """
        items = self._equation_entries or []
        if n is not None:
            try:
                n = int(n)
                items = items[:max(0, n)]
            except Exception:
                pass
        out: List[dict] = []
        for e in items:
            out.append({
                'equation': self._wrap_equation(e.get('equation', '')),
                'params': e.get('params'),
                'score': e.get('score'),
                'category': e.get('category'),
                'sample_order': e.get('sample_order')
            })
        return out

    def get_fitted_params(self):
        """返回最佳方程的训练期拟合参数（列表形式），若不存在则返回 None。"""
        try:
            if isinstance(self._best_params, np.ndarray):
                return self._best_params.tolist()
        except Exception:
            pass
        # 尝试从 entries 的首项获取
        if getattr(self, '_equation_entries', None):
            p = self._equation_entries[0].get('params')
            return p
        return None

    def __str__(self) -> str:
        lines = [f"DRSRRegressor(tool='drsr')"]
        try:
            eq_str = self.get_optimal_equation()
            if eq_str:
                lines.append("最佳方程:")
                lines.append(eq_str.rstrip())
        except Exception:
            pass
        try:
            if isinstance(self._best_params, np.ndarray):
                np.set_printoptions(precision=8, suppress=False)
                lines.append(f"最佳参数: {np.array2string(self._best_params, precision=8, suppress_small=False)}")
        except Exception:
            pass
        if self._equation_entries:
            k = min(3, len(self._equation_entries))
            lines.append(f"Top-{k} 候选（方程+分数简要）:")
            for i, e in enumerate(self._equation_entries[:k], 1):
                score = e.get('score')
                eq = e.get('equation') or ''
                eq_one_line = " ".join(eq.strip().split())
                if len(eq_one_line) > 120:
                    eq_one_line = eq_one_line[:120] + '...'
                lines.append(f"  {i}. score={score}, eq={eq_one_line}")
        return "\n".join(lines)

    @staticmethod
    def _wrap_equation(body: str) -> str:
        body = body.rstrip("\n") + "\n"
        return "def equation(x, v, params):\n" + body

    @staticmethod
    def _compile_equation(body: str):
        code = DRSRRegressor._wrap_equation(body)
        ns = {}
        # 提供 numpy 命名以支持方程体中的 np.sin/np.cos 等写法
        try:
            import numpy as _np
            ns["np"] = _np
        except Exception:
            pass
        exec(code, ns)
        return ns["equation"]

    def _fit_params(self, X: np.ndarray, y: np.ndarray, n_params: int = 10, n_starts: int = 5) -> np.ndarray:
        """
        使用与评估相同思想的 BFGS 在训练集上拟合参数，并返回最优参数。
        """
        eq = self._equation_func
        if not callable(eq):
            return np.ones(n_params)

        def loss_fn(p: np.ndarray) -> float:
            try:
                y_pred = eq(X[:, 0], X[:, 1], p)
                return float(np.mean((y_pred - y) ** 2))
            except Exception:
                # 不可导的坏点，返回大损失
                return 1e6

        best_p = None
        best_loss = None
        rng = np.random.default_rng(0)
        for _ in range(max(1, n_starts)):
            x0 = rng.uniform(low=-1.0, high=1.0, size=n_params)
            try:
                res = minimize(loss_fn, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-10, 'eps': 1e-12, 'disp': False})
                cur_loss = float(res.fun)
                if best_loss is None or cur_loss < best_loss:
                    best_loss = cur_loss
                    best_p = np.array(res.x)
            except Exception:
                continue
        return best_p if isinstance(best_p, np.ndarray) else np.ones(n_params)
