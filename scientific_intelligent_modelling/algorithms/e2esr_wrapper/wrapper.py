import pickle
import base64
import numpy as np
import os
import sys
import requests
import sympy as sp
import torch

from ..base_wrapper import BaseWrapper 

class E2ESRRegressor(BaseWrapper):
    def __init__(self, 
                 model_path=None, 
                 model_url="https://dl.fbaipublicfiles.com/symbolicregression/model1.pt",
                 max_input_points=200, 
                 n_trees_to_refine=100, 
                 rescale=True,
                 **kwargs):
        """
        初始化E2ESR回归器
        
        参数：
        model_path: 模型文件路径，如果为None，则使用默认路径
        model_url: 模型下载URL，如果本地没有模型文件则从此URL下载
        max_input_points: 最大输入点数
        n_trees_to_refine: 要优化的树的数量
        rescale: 是否重新缩放数据
        """
        # 保存参数
        self.params = {
            'max_input_points': max_input_points,
            'n_trees_to_refine': n_trees_to_refine,
            'rescale': rescale,
            **kwargs
        }
        self.model = None
        self.regressor = None
        self.best_tree = None
        
        # 获取e2esr模块的路径，添加到系统路径中
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.e2esr_path = os.path.join(current_dir, "e2esr")
        sys.path.append(self.e2esr_path)
        
        # 如果提供了模型路径，则使用该路径，否则使用默认路径
        if model_path is None:
            # 默认模型路径修改为当前包装器目录下的model.pt文件
            model_path = os.path.join(current_dir, "model.pt")
        
        self.model_path = model_path
        self.model_url = model_url
        
        # 立即加载模型
        self._load_model()
    
    def _load_model(self):
        """加载预训练模型，如果本地不存在则从URL下载"""
        # 如果模型已经加载，直接返回
        if self.model is not None:
            return
            
        try:
            # 检查模型文件是否存在，不存在则从URL下载
            if not os.path.isfile(self.model_path):
                print(f"从 {self.model_url} 下载模型...")
                r = requests.get(self.model_url, allow_redirects=True)
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    f.write(r.content)
                print(f"模型已保存到 {self.model_path}")
            
            # 加载模型
            if not torch.cuda.is_available():
                self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
            else:
                self.model = torch.load(self.model_path)
                self.model = self.model.cuda()
            
            print(f"模型已成功加载！设备: {self.model.device if hasattr(self.model, 'device') else 'cpu'}")
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            # 如果加载失败，模型将在fit时创建
            self.model = None
    
    def fit(self, X, y):
        """
        训练模型，如果已经加载了预训练模型，则使用该模型
        否则创建一个新模型
        """
        try:
            # 导入E2ESR相关模块
            from symbolicregression.model import SymbolicTransformerRegressor
            
            # 如果模型尚未加载，报错
            if self.model is None:
                raise ValueError("模型未加载成功，请检查模型路径或网络连接")
            
            # 创建回归器
            self.regressor = SymbolicTransformerRegressor(
                model=self.model,
                **self.params
            )
            
            # 训练回归器
            self.regressor.fit(X, y)
            
            # 获取并保存最佳树
            self.best_tree = self.regressor.retrieve_tree(with_infos=True)
            
            return self
            
        except Exception as e:
            print(f"训练模型时出错: {str(e)}")
            raise
    
    def predict(self, X):
        """使用训练好的模型进行预测"""
        if self.regressor is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.regressor.predict(X)
    
    def get_optimal_equation(self):
        """返回模型拟合的最优数学方程"""
        if self.best_tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 如果已经计算过方程，直接返回缓存的结果
        if hasattr(self, '_cached_optimal_equation'):
            return self._cached_optimal_equation
            
        try:
            # 获取表达式并格式化
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            model_str = self.best_tree["relabed_predicted_tree"].infix()
            for op, replace_op in replace_ops.items():
                model_str = model_str.replace(op, replace_op)
            
            # 解析为sympy表达式并转为字符串
            expr = sp.parse_expr(model_str)
            self._cached_optimal_equation = str(expr)
            return self._cached_optimal_equation
        except Exception as e:
            print(f"获取方程时出错: {str(e)}")
            # 如果解析失败，返回原始的中缀表达式
            self._cached_optimal_equation = str(self.best_tree["relabed_predicted_tree"].infix())
            return self._cached_optimal_equation

    def get_total_equations(self):
        """获取模型学习到的所有符号方程"""
        if self.best_tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 如果已经计算过所有方程，直接返回缓存的结果
        if hasattr(self, '_cached_total_equations'):
            return self._cached_total_equations
            
        # E2ESR可能没有提供获取多个方程的方法，这里返回最优方程作为单元素列表
        self._cached_total_equations = [self.get_optimal_equation()]
        return self._cached_total_equations


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    # 生成示例数据
    X = np.random.randn(100, 2)
    y = np.cos(2*np.pi*X[:, 0]) + X[:, 1]**2

    # 创建模型，指定预训练模型路径
    model = E2ESRRegressor()
    
    # 训练模型
    model.fit(X, y)

    # 获取最优方程
    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")

    # 进行预测
    y_pred = model.predict(X)
    print(f"预测值前5个: {y_pred[:5]}")