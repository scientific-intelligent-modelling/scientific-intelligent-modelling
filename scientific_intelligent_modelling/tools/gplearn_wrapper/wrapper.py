# tools/gplearn_wrapper/wrapper.py
import pickle
import base64
import numpy as np

class GPLearnRegressor:
    def __init__(self, **kwargs):
        # 延迟导入，避免环境问题
        self.params = kwargs
        self.model = None
    
    def fit(self, X, y):
        # 仅在需要时导入
        from gplearn.genetic import SymbolicRegressor as GPLearnSR
        
        # 创建并训练模型
        self.model = GPLearnSR(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def to_dict(self):
        """将模型序列化为字典"""
        if self.model is None:
            return {'params': self.params}
        
        # 使用pickle序列化模型
        model_bytes = pickle.dumps(self.model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        return {
            'params': self.params,
            'model_b64': model_b64
        }
    
    def from_dict(self, state_dict):
        """从字典反序列化模型"""
        self.params = state_dict.get('params', {})
        
        if 'model_b64' in state_dict:
            model_bytes = base64.b64decode(state_dict['model_b64'])
            self.model = pickle.loads(model_bytes)
