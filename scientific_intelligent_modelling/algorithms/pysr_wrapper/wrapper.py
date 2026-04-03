# tools/pysr_wrapper/wrapper.py
import pickle
import base64
import numpy as np

from ..base_wrapper import BaseWrapper 

class PySRRegressor(BaseWrapper):
    _META_PARAMS = {"exp_name", "exp_path", "problem_name", "seed"}
    _ALLOWED_PARAMS = {
        "niterations",
        "population_size",
        "populations",
        "ncycles_per_iteration",
        "maxsize",
        "maxdepth",
        "parsimony",
        "binary_operators",
        "unary_operators",
        "elementwise_loss",
        "model_selection",
        "loss_function",
        "warm_start",
        "random_state",
        "procs",
        "n_jobs",
        "verbosity",
        "progress",
    }
    
    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        self.params = self._validate_and_normalize_params(kwargs)
        self.model = None

    @classmethod
    def _validate_and_normalize_params(cls, raw_params):
        params = {}
        raw_params = dict(raw_params)
        # 先保存元参数，便于后续复现性映射
        seed = raw_params.get("seed")
        # 剥离系统参数
        for key in cls._META_PARAMS:
            raw_params.pop(key, None)

        # 复现性参数透传映射
        if "random_state" not in raw_params and seed is not None:
            params["random_state"] = int(seed)

        unknown = sorted(set(raw_params) - cls._ALLOWED_PARAMS)
        if unknown:
            raise ValueError(
                "PySR 参数不受支持: {}。当前允许的参数有: {}。".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )

        for key, value in raw_params.items():
            if key == "n_jobs":
                # 与 pysr 主参数兼容：统一使用 procs
                params.setdefault("procs", int(value))
                continue
            if key == "seed":
                continue
            params[key] = value

        # 默认打开进度与基础日志，便于远程实验观测。
        params.setdefault("progress", True)
        params.setdefault("verbosity", 1)

        return params
    
    def fit(self, X, y):
        # 仅在需要时导入
        from pysr import PySRRegressor
        
        # 创建并训练模型
        self.model = PySRRegressor(**self.params)
        self.model.fit(X, y)

        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def get_optimal_equation(self):
        """返回模型拟合的数学方程"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        # self.model.best()
        return str(self.model.sympy())
    
    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        equations = self.model.equations_
        if hasattr(equations, "to_dict"):
            # pandas.DataFrame 常见于 PySR：优先提取可读表达式列并保序
            if hasattr(equations, "columns"):
                for col in ("sympy_format", "equation", "expr", "expression"):
                    if col in equations.columns:
                        return [str(item) for item in equations[col].dropna().tolist()]
                # 最后兜底：每行转为字符串字典
                return [
                    {k: str(v) for k, v in row.items()}
                    for row in equations.to_dict(orient="records")
                ]
        if isinstance(equations, list):
            return [str(eq) for eq in equations]
        return [str(equations)]
  


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    # 生成示例数据
    X = np.random.rand(100, 2)
    y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.1*np.random.randn(100)

    # 创建并训练模型
    model = PySRRegressor(niterations=5, population_size=1000)
    model.fit(X, y)

    # 获取最优方程
    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")

    equations = model.get_total_equations()
    print(f"所有方程: {equations}")
