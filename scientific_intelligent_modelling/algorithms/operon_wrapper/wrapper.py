# algorithms/operon_wrapper/wrapper.py
import pickle
import base64
import numpy as np
from ..base_wrapper import BaseWrapper 
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_operon_artifact

class OperonRegressor(BaseWrapper):
    _META_PARAMS = {"exp_name", "exp_path", "problem_name", "seed"}
    _ALLOWED_PARAMS = {
        "allowed_symbols",
        "symbolic_mode",
        "crossover_probability",
        "crossover_internal_probability",
        "mutation",
        "mutation_probability",
        "offspring_generator",
        "reinserter",
        "objectives",
        "optimizer",
        "optimizer_likelihood",
        "optimizer_batch_size",
        "optimizer_iterations",
        "local_search_probability",
        "lamarckian_probability",
        "sgd_update_rule",
        "sgd_learning_rate",
        "sgd_beta",
        "sgd_beta2",
        "sgd_epsilon",
        "sgd_debias",
        "max_length",
        "max_depth",
        "initialization_method",
        "initialization_max_length",
        "initialization_max_depth",
        "female_selector",
        "male_selector",
        "population_size",
        "pool_size",
        "generations",
        "max_evaluations",
        "max_selection_pressure",
        "comparison_factor",
        "brood_size",
        "tournament_size",
        "irregularity_bias",
        "epsilon",
        "model_selection_criterion",
        "add_model_scale_term",
        "add_model_intercept_term",
        "uncertainty",
        "n_threads",
        "max_time",
        "random_state",
        "n_jobs",
        "niterations",
        "niteration",
        "population",
    }

    def __init__(self, **kwargs):
        self.params = self._validate_and_normalize_params(dict(kwargs))
        self.model = None

    @classmethod
    def _validate_and_normalize_params(cls, raw_params):
        params = {}
        seed = raw_params.get("seed")
        for key in cls._META_PARAMS:
            raw_params.pop(key, None)

        if "random_state" not in raw_params and seed is not None:
            params["random_state"] = int(seed)

        # 常见别名
        alias_map = {
            "n_jobs": "n_threads",
            "niterations": "generations",
            "niteration": "generations",
            "population": "population_size",
        }

        for old, new in alias_map.items():
            if old in raw_params and new not in raw_params:
                raw_params[new] = raw_params.pop(old)

        # 剔除 seed，防止与上面映射重复
        raw_params.pop("seed", None)

        unknown = sorted(set(raw_params) - cls._ALLOWED_PARAMS)
        if unknown:
            raise ValueError(
                "PyOperon 参数不受支持: {}。当前允许的参数有: {}。".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )

        params.update(raw_params)
        # 避免 float/int 混用
        if "n_threads" in params:
            try:
                params["n_threads"] = int(params["n_threads"])
            except Exception:
                pass

        if "generations" in params:
            try:
                params["generations"] = int(params["generations"])
            except Exception:
                pass

        if "population_size" in params:
            try:
                params["population_size"] = int(params["population_size"])
            except Exception:
                pass

        return params
    
    def fit(self, X, y):
        # 仅在需要时导入
        from pyoperon.sklearn import SymbolicRegressor
        
        # 创建并训练模型
        self.model = SymbolicRegressor(**self.params)
        self.model.fit(X, y)
        # 保存关键元信息，避免 pickle 非法对象导致子进程间反序列化失败
        self.best_model_str = self.model.get_model_string(self.model.model_)
        self.pareto_models = self._extract_pareto_models(self.model.pareto_front_)
        self.n_features_ = int(X.shape[1])

        return self
    
    def predict(self, X):
        if self.model is None:
            # 如果子进程反序列化恢复到轻量级状态，则通过表达式回退预测
            if hasattr(self, "best_model_str") and getattr(self, "best_model_str"):
                return self._predict_from_expression(X)
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def get_optimal_equation(self):
        """返回模型拟合的数学方程"""
        if self.model is None:
            if hasattr(self, "best_model_str") and getattr(self, "best_model_str"):
                return self.best_model_str
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        return self.best_model_str if hasattr(self, "best_model_str") else str(self.model)
    
    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        if self.model is None:
            if hasattr(self, "pareto_models") and self.pareto_models is not None:
                return self.pareto_models
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        return self.pareto_models or [self.best_model_str]

    @staticmethod
    def _extract_pareto_models(pareto_front):
        models = []
        if not pareto_front:
            return models
        for item in pareto_front:
            if isinstance(item, dict) and "model" in item:
                models.append(str(item["model"]))
        return models
    
    def _build_predict_fn(self):
        if not hasattr(self, "best_model_str") or not getattr(self, "best_model_str", None):
            return None
        import sympy as sp

        n_features = getattr(self, "n_features_", 0)
        if n_features <= 0:
            return None

        symbols = [sp.Symbol(f"X{i+1}") for i in range(n_features)]
        try:
            expr = sp.sympify(self.best_model_str)
            fn = sp.lambdify(symbols, expr, modules=["numpy"])
        except Exception:
            try:
                expr = sp.sympify(self.best_model_str.replace(" ", ""))
                fn = sp.lambdify(symbols, expr, modules=["numpy"])
            except Exception:
                return None
        return fn

    def _predict_from_expression(self, X):
        fn = self._build_predict_fn()
        if fn is None:
            raise ValueError("无法从模型表达式构建预测函数，当前状态不支持直接预测")
        X = np.asarray(X)
        cols = [X[:, i] for i in range(X.shape[1])]
        return np.asarray(fn(*cols))

    def serialize(self):
        payload = {
            "params": self.params,
            "best_model_str": getattr(self, "best_model_str", None),
            "pareto_models": getattr(self, "pareto_models", None),
            "n_features_": getattr(self, "n_features_", None),
        }
        return base64.b64encode(pickle.dumps(payload)).decode("utf-8")

    @classmethod
    def deserialize(cls, instance_b64):
        payload = pickle.loads(base64.b64decode(instance_b64))
        inst = cls(**payload.get("params", {}))
        inst.best_model_str = payload.get("best_model_str")
        inst.pareto_models = payload.get("pareto_models")
        inst.n_features_ = payload.get("n_features_")
        inst.model = None
        return inst

    def export_canonical_symbolic_program(self):
        equation = self.get_optimal_equation()
        return normalize_operon_artifact(
            equation,
            expected_n_features=getattr(self, "n_features_", None),
        )
  
if __name__ == "__main__":
    # 测试代码
    import numpy as np

    # 生成示例数据
    X = np.random.rand(100, 2)
    y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.1*np.random.randn(100)

    # 创建并训练模型
    model = OperonRegressor(niterations=100, population_size=1000)
    model.fit(X, y)

    # 获取最优方程
    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")

    # 获取所有方程
    total_equations = model.get_total_equations()
    print(f"所有方程: {total_equations}")
