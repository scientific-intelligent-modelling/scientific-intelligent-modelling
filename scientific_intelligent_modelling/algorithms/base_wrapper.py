from abc import ABC, abstractmethod
import pickle
import base64

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

    def serialize(self):
        """将整个类实例序列化为字符串"""
        # 使用pickle序列化整个类实例
        instance_bytes = pickle.dumps(self)
        # 将字节转换为base64编码的字符串
        instance_b64 = base64.b64encode(instance_bytes).decode('utf-8')
        return instance_b64

    @classmethod
    def deserialize(cls, instance_b64):
        """从序列化的字符串反序列化为类实例"""
        # 将base64编码的字符串转换回字节
        instance_bytes = base64.b64decode(instance_b64)
        # 使用pickle从字节重建类实例
        instance = pickle.loads(instance_bytes)
        return instance
        
    def __str__(self):
        """返回模型的字符串表示"""
        return f"{self.__class__.__name__}()"
