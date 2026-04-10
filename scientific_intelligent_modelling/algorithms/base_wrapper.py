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

    def _infer_tool_name(self):
        """从模块路径推断工具名。"""
        module_name = getattr(self.__class__, "__module__", "") or ""
        marker = ".algorithms."
        if marker in module_name and "_wrapper" in module_name:
            try:
                suffix = module_name.split(marker, 1)[1]
                return suffix.split("_wrapper", 1)[0]
            except Exception:
                pass
        return self.__class__.__name__

    def export_canonical_symbolic_program(self):
        """导出统一符号工件的默认实现。

        Phase 1 只保证导出最小工件，不在这里做重度归一化。
        具体算法可在各自 wrapper 中覆写该方法，补更高保真度的信息。
        """
        from scientific_intelligent_modelling.benchmarks.artifact_schema import (
            build_canonical_symbolic_program,
        )

        raw_equation = self.get_optimal_equation()
        parameter_values = None
        if hasattr(self, "get_fitted_params"):
            try:
                parameter_values = self.get_fitted_params()
            except Exception:
                parameter_values = None
        return build_canonical_symbolic_program(
            tool_name=self._infer_tool_name(),
            raw_equation=raw_equation,
            parameter_values=parameter_values,
            normalization_mode="wrapper_raw",
        )

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
