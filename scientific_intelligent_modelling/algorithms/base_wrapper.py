from abc import ABC, abstractmethod

class BaseWrapper(ABC):
    """所有估计器的基类"""
    
    @abstractmethod
    def fit(self, X, y):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """使用模型进行预测"""
        pass

    @abstractmethod
    def get_optimal_equation(self):
        """获得最优方程"""
        pass

    @abstractmethod
    def get_total_equations(self):
        """获得所有方程"""
        pass






    @abstractmethod
    def to_dict(self):
        """将模型序列化为字典"""
        pass
    
    @abstractmethod
    def from_dict(self, state_dict):
        """从字典反序列化模型"""
        pass
        
    def __str__(self):
        """返回模型的字符串表示"""
        return f"{self.__class__.__name__}()"
