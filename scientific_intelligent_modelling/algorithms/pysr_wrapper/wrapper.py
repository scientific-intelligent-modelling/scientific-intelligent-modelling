# tools/pysr_wrapper/wrapper.py
import pickle
import base64
import numpy as np

from ..base_wrapper import BaseWrapper 

class PySRRegressor(BaseWrapper):
    def __init__(self, **kwargs):
        # 延迟导入，避免环境问题
        self.params = kwargs
        self.model = None
    
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
        print(self.model.sympy())
        return str(self.model.sympy())
    
    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        return self.model.equations_
  


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