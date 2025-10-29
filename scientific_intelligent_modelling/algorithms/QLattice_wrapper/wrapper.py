"""QLattice 回归器包装器（原生实现）

说明：
- 适配 srkit/subprocess_runner 的统一调用：fit/predict/get_optimal_equation/get_total_equations
- 训练期依赖在线 feyn.QLattice()（社区版）
- 序列化采用稳健 JSON：保存最优表达式与候选方程字符串，反序列化后可离线预测
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import List, Optional

from ..base_wrapper import BaseWrapper


class QLatticeRegressor(BaseWrapper):
    """QLattice 的回归器封装（回归任务）。

    参数（常用）：
    - n_epochs: int = 100，自动搜索轮数
    - kind: str = 'regression'，任务类型
    - signif: int = 4，表达式输出的有效数字（用于 sympify 展示）
    - 其他 QLattice.auto_run 支持的参数可透传
    """

    def __init__(self, **kwargs) -> None:
        self.params = dict(kwargs)
        self.model = None

        # QLattice 相关缓存
        self._ql = None
        self._models = []
        self._best_model = None

        # 表达式/预测相关缓存
        self._expr_str: Optional[str] = None  # 最优表达式字符串
        self._input_vars: List[str] = []
        self._output_name: str = 'y'
        self._lambdified = None
        # 候选方程字符串列表（便于序列化后仍可获取多个解）
        self._equations: List[str] = []

    # ---------------- 公共接口 ----------------
    def fit(self, X, y):
        """训练 QLattice 模型并缓存最优与候选表达式。"""
        import feyn

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 构造 DataFrame
        n_features = X.shape[1]
        self._input_vars = [f"x{i}" for i in range(n_features)]
        self._output_name = self.params.get('output_name', 'y')
        df = pd.DataFrame(X, columns=self._input_vars)
        df[self._output_name] = y

        # 连接 QLattice（社区版需要联网）
        self._ql = feyn.QLattice()

        # 组装 auto_run 参数
        auto_args = {
            'data': df,
            'output_name': self._output_name,
            'kind': self.params.get('kind', 'regression'),
            'n_epochs': int(self.params.get('n_epochs', 100)),
        }
        # 尽量覆盖 QLattice.auto_run 的可选参数（按需透传）
        for k in [
            'stypes', 'threads', 'max_complexity', 'query_string',
            'loss_function', 'criterion', 'sample_weights',
            'function_names', 'starting_models'
        ]:
            if k in self.params:
                auto_args[k] = self.params[k]

        models = list(self._ql.auto_run(**auto_args))
        if not models:
            raise RuntimeError('QLattice.auto_run 未返回任何模型，请检查数据与参数。')

        self._models = models
        self._best_model = models[0]
        self.model = True

        # 提取最优与候选表达式
        signif = int(self.params.get('signif', 4))
        try:
            self._expr_str = str(self._best_model.sympify(signif=signif))
        except Exception:
            self._expr_str = str(self._best_model)

        equations: List[str] = []
        for m in self._models:
            try:
                equations.append(str(m.sympify(signif=signif)))
            except Exception:
                equations.append(str(m))
        self._equations = equations

        # 构建 lambdify 预测器（便于序列化回放）
        self._build_lambdify()
        return self

    def predict(self, X):
        """使用模型进行预测。"""
        if self.model is None:
            raise ValueError('模型尚未训练，请先调用 fit 方法。')

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self._input_vars):
            raise ValueError(f'特征维度不匹配：期望 {len(self._input_vars)} 列，实际 {X.shape[1]} 列。')

        # 优先使用原生 feyn 模型（更稳健），其次使用 lambdify
        if self._best_model is not None:
            import pandas as pd
            df = pd.DataFrame(X, columns=self._input_vars)
            return np.asarray(self._best_model.predict(data=df))

        if self._lambdified is not None:
            cols = [X[:, i] for i in range(X.shape[1])]
            y_pred = self._lambdified(*cols)
            return np.asarray(y_pred)

        raise RuntimeError('未找到可用的预测器（表达式或 QLattice 模型）。')

    def get_optimal_equation(self):
        if self.model is None:
            raise ValueError('模型尚未训练，请先调用 fit 方法。')
        if self._expr_str:
            return self._expr_str
        try:
            signif = int(self.params.get('signif', 4))
            return str(self._best_model.sympify(signif=signif))
        except Exception:
            return '未找到可用的方程'

    def get_total_equations(self, n: int | None = None):
        """返回候选模型的表达式列表（字符串）。

        参数:
            n: 返回的方程数量上限；若为 None 或无效，则返回全部。
        """
        if self.model is None:
            raise ValueError('模型尚未训练，请先调用 fit 方法。')
        results: List[str] = []
        signif = int(self.params.get('signif', 4))
        if self._models:
            for m in self._models:
                try:
                    results.append(str(m.sympify(signif=signif)))
                except Exception:
                    results.append(str(m))
        elif self._equations:
            results = list(self._equations)
        elif self._expr_str:
            results = [self._expr_str]
        if isinstance(n, int) and n > 0:
            return results[:n]
        return results

    # ---------------- 序列化/反序列化 ----------------
    def serialize(self):
        state = {
            'params': self.params,
            'expr': self._expr_str,
            'input_vars': self._input_vars,
            'output_name': self._output_name,
            'equations': self._equations,
        }
        return json.dumps(state, ensure_ascii=False)

    @classmethod
    def deserialize(cls, payload: str) -> 'QLatticeRegressor':
        obj = json.loads(payload)
        inst = cls(**obj.get('params', {}))
        inst.model = True
        inst._expr_str = obj.get('expr')
        inst._input_vars = obj.get('input_vars') or []
        inst._output_name = obj.get('output_name') or 'y'
        inst._equations = obj.get('equations') or []
        inst._build_lambdify()
        return inst

    # ---------------- 内部工具 ----------------
    def _build_lambdify(self):
        if not self._expr_str or not self._input_vars:
            self._lambdified = None
            return
        try:
            import sympy as sp
            syms = sp.symbols(self._input_vars)
            expr = sp.sympify(self._expr_str)
            self._lambdified = sp.lambdify(syms, expr, modules='numpy')
        except Exception:
            self._lambdified = None

__all__ = ["QLatticeRegressor"]
