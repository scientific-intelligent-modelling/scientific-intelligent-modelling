# tools/gplearn_wrapper/wrapper.py
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
        # 仅在需要时导入
        from dso import DeepSymbolicRegressor
        import warnings
        # 过滤掉特定的FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning, 
                                message="`BaseEstimator._validate_data` is deprecated")
        
        # 创建并训练模型
        self.model = DeepSymbolicRegressor(**self.params)
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
    