# tools/gplearn_wrapper/wrapper.py
import os
import sys
import pickle
import base64
import numpy as np
from ..base_wrapper import BaseWrapper 

class DSORegressor(BaseWrapper):
    def __init__(self, **kwargs):
        # 延迟导入，避免环境问题
        self.params = kwargs
        self.model = None
    
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
        return str(self.model.program_.pretty())

    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        return [str(self.model.program_.pretty())]
    
