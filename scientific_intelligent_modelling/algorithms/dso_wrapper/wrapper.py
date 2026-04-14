# tools/gplearn_wrapper/wrapper.py
import os
import sys
from typing import Any, Dict

import numpy as np
from sympy import sympify, lambdify
from sympy.core.sympify import SympifyError

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_dso_artifact

class DSORegressor(BaseWrapper):
    _DEFAULT_EXPERIMENT = {
        "logdir": None,
    }
    _DEFAULT_TASK = {
        "task_type": "regression",
        "function_set": ["add", "sub", "mul", "div"],
        "metric": "inv_nrmse",
        "metric_params": [1.0],
        "threshold": 1e-12,
        "protected": False,
    }
    _DEFAULT_TRAINING = {
        "batch_size": 1,
        "n_samples": 20,
        "epsilon": 0.05,
    }
    _EXPERIMENT_KEYS = {"logdir", "exp_name", "seed"}
    _TASK_KEYS = {"task_type", "function_set", "metric", "metric_params", "threshold", "protected"}
    _TRAINING_KEYS = {"batch_size", "n_samples", "epsilon"}

    def __init__(self, **kwargs):
        # 延迟导入，避免环境问题
        self.params = self._build_config(kwargs)
        self.model = None
        self._dso_equation = None
        self._dso_expression = None
        self._dso_pred_fn = None
        self._dso_var_count = 0
        self._dso_input_indices = []
        self._dso_n_features = None

    @classmethod
    def _build_config(cls, raw_kwargs):
        params = dict(raw_kwargs or {})
        experiment = dict(cls._DEFAULT_EXPERIMENT)
        task = dict(cls._DEFAULT_TASK)
        training = dict(cls._DEFAULT_TRAINING)

        for key in list(params.keys()):
            value = params[key]
            if key == "experiment" and isinstance(value, dict):
                experiment.update(value)
                params.pop(key)
            elif key == "task" and isinstance(value, dict):
                task.update(value)
                params.pop(key)
            elif key == "training" and isinstance(value, dict):
                training.update(value)
                params.pop(key)

        for key in list(params.keys()):
            if key in cls._EXPERIMENT_KEYS:
                experiment[key] = params.pop(key)
            elif key in cls._TASK_KEYS:
                task[key] = params.pop(key)
            elif key in cls._TRAINING_KEYS:
                training[key] = params.pop(key)

        exp_path = params.pop("exp_path", None)
        exp_name = params.pop("exp_name", None)
        problem_name = params.pop("problem_name", None)
        seed = params.pop("seed", None)
        params.pop("timeout_in_seconds", None)

        if seed is not None and "seed" not in experiment:
            experiment["seed"] = int(seed)
        if exp_name and "exp_name" not in experiment:
            experiment["exp_name"] = str(exp_name)
        if exp_path and experiment.get("logdir") is None:
            # DSO 内部会再用 exp_name 组装 save_path，logdir 这里只传实验根目录，
            # 避免最终路径变成 exp_path/exp_name/exp_name 的双层结构。
            experiment["logdir"] = os.path.abspath(str(exp_path))

        config = dict(params)
        config["experiment"] = experiment
        config["task"] = task
        config["training"] = training
        return config
    
    def fit(self, X, y):
        # 优先使用子仓库源码，避免环境可复现性差异导致的 editable 安装问题
        repo_root = os.path.dirname(os.path.abspath(__file__))
        local_dso_path = os.path.join(repo_root, "dso", "dso")
        if os.path.isdir(local_dso_path) and local_dso_path not in sys.path:
            sys.path.insert(0, local_dso_path)

        # 仅在需要时导入
        from dso import DeepSymbolicRegressor
        import warnings
        # 过滤掉特定的FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning, 
                                message="`BaseEstimator._validate_data` is deprecated")
        
        if not hasattr(__import__("dso"), "DeepSymbolicRegressor"):
            raise ImportError("当前 dso 包未提供 DeepSymbolicRegressor，请检查 dso 源码或安装版本。")

        # 创建并训练模型
        self.model = DeepSymbolicRegressor(config=self.params)
        self.model.fit(X, y)
        x_arr = np.asarray(X)
        self._dso_n_features = int(x_arr.shape[1]) if x_arr.ndim == 2 else 1
        self._cache_post_fit_state()
        return self

    def _cache_post_fit_state(self):
        if self.model is None or not hasattr(self.model, "program_"):
            self._dso_equation = None
            self._dso_expression = None
            self._dso_pred_fn = None
            self._dso_var_count = 0
            return

        self._dso_expression = str(self.model.program_.sympy_expr)
        self._dso_equation = str(self.model.program_.pretty())
        # 仅在运行时可进行反序列化后的预测，不在主进程训练过程里触发额外解析成本
        self._build_predict_fn_from_equation(self._dso_expression)

    def _build_predict_fn_from_equation(self, equation: str):
        try:
            expr = sympify(equation.strip(), locals={"x": None})
        except (SympifyError, TypeError, SyntaxError, ValueError):
            self._dso_var_count = 0
            self._dso_pred_fn = None
            return

        raw_indices = []
        for token in expr.free_symbols:
            name = str(token)
            if not name.startswith("x"):
                continue
            suffix = name[1:]
            if suffix.isdigit():
                raw_indices.append(int(suffix))

        raw_indices = sorted(set(raw_indices))
        if not raw_indices:
            self._dso_var_count = 0
            self._dso_pred_fn = None
            self._dso_input_indices = []
            return

        one_based = 0 not in raw_indices
        input_indices = []
        for idx in raw_indices:
            mapped = idx - 1 if one_based else idx
            if mapped < 0:
                self._dso_var_count = 0
                self._dso_pred_fn = None
                self._dso_input_indices = []
                return
            input_indices.append(mapped)

        self._dso_var_count = max(input_indices) + 1
        self._dso_pred_fn = None
        self._dso_input_indices = input_indices

        if self._dso_var_count <= 0:
            self._dso_input_indices = []
            return

        symbols = [f"x{idx}" for idx in raw_indices]
        self._dso_pred_fn = lambdify(symbols, expr, modules="numpy")

    def _predict_with_cached_fn(self, X):
        if self._dso_pred_fn is None:
            raise RuntimeError("DSO 反序列化模型不包含可执行方程，无法继续执行 predict")

        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        n_features = x_arr.shape[1] if x_arr.ndim > 1 else 0
        if n_features < self._dso_var_count:
            raise ValueError("DSO 反序列化状态下的输入特征维度不足")

        args = [x_arr[:, idx] for idx in self._dso_input_indices]
        y = self._dso_pred_fn(*args)
        return np.asarray(y, dtype=float).reshape(-1)
    
    def predict(self, X):
        if self.model is None:
            if self._dso_pred_fn is not None:
                return self._predict_with_cached_fn(X)
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def get_optimal_equation(self):
        """返回模型拟合的数学方程"""
        if self.model is None:
            if self._dso_equation is not None:
                return self._dso_equation
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        return str(self.model.program_.pretty())

    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        if self.model is None:
            if self._dso_equation is not None:
                return [self._dso_equation]
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        return [str(self.model.program_.pretty())]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # 规避 RLock/进程上下文等不可 pickle 对象，保留方程文本供反序列化恢复预测
        if state.get("model") is not None:
            self._cache_post_fit_state()
        state["model"] = None
        state["_dso_input_indices"] = self._dso_input_indices
        state["_dso_pred_fn"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        if self.model is not None:
            self.model = None
        if self._dso_expression:
            self._build_predict_fn_from_expression()
            return
        if self._dso_equation:
            self._build_predict_fn_from_equation(self._dso_equation)

    def _build_predict_fn_from_expression(self):
        if not self._dso_expression:
            return
        self._build_predict_fn_from_equation(self._dso_expression)

    def export_canonical_symbolic_program(self):
        raw_equation = self._dso_expression
        if raw_equation is None and self.model is not None and hasattr(self.model, "program_"):
            try:
                raw_equation = str(self.model.program_.sympy_expr)
            except Exception:
                raw_equation = None
        if raw_equation is None:
            raise ValueError("DSO 当前没有可导出的标准表达式")
        expected_n_features = self._dso_n_features
        if expected_n_features is None and self._dso_var_count:
            expected_n_features = int(self._dso_var_count)
        return normalize_dso_artifact(
            raw_equation,
            expected_n_features=expected_n_features,
        )
    
