# srkit/regressor.py
import os
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path

from .config_manager import config_manager
from .environment_manager import environment_manager

class SymbolicRegressor:
    def __init__(self, tool_name, **kwargs):
        """
        初始化符号回归器
        
        参数:
            tool_name: 要使用的工具名称 (例如 'gplearn', 'pysr')
            **kwargs: 传递给实际工具的参数
        """
        self.tool_name = tool_name
        self.params = kwargs
        self.model_state = None
        
        # 使用config_manager获取环境名称
        self.env_name = config_manager.get_env_name_by_tool(tool_name)
        if not self.env_name:
            raise ValueError(f"未找到工具 '{tool_name}' 的环境配置")
        
        # 检查环境是否存在
        if not environment_manager.environment_exists(self.env_name):
            print(f"环境 '{self.env_name}' 不存在，正在创建...")
            success = environment_manager.create_conda_environment(self.env_name)
            if not success:
                raise RuntimeError(f"无法创建环境 '{self.env_name}'")
    
    def fit(self, X, y):
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
        
        返回:
            self: 支持链式调用
        """
        # 准备数据
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        
        # 创建命令
        command = {
            'action': 'fit',
            'data': {'X': X, 'y': y},
            'params': self.params,
            'tool_name': self.tool_name
        }
        
        # 执行命令并获取结果
        result = self._execute_subprocess(command)
        
        # 检查结果
        if 'error' in result:
            raise RuntimeError(f"训练失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 保存模型状态
        self.model_state = result.get('model_state', {})
        return self
    
    def predict(self, X):
        """
        使用模型进行预测
        
        参数:
            X: 特征矩阵
        
        返回:
            predictions: 预测结果
        """
        # 检查模型是否已训练
        if self.model_state is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 准备数据
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # 创建命令
        command = {
            'action': 'predict',
            'data': {'X': X},
            'model_state': self.model_state,
            'tool_name': self.tool_name
        }
        
        # 执行命令并获取结果
        result = self._execute_subprocess(command)
        
        # 检查结果
        if 'error' in result:
            raise RuntimeError(f"预测失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 返回预测结果
        predictions = result.get('predictions', [])
        return np.array(predictions)
    
    def _execute_subprocess(self, command):
        """执行子进程命令"""
        # 获取Python解释器路径
        python_path = environment_manager.get_environment_python(self.env_name)
        if not python_path:
            raise RuntimeError(f"无法获取环境 '{self.env_name}' 的Python路径")
        
        # 创建临时文件存储命令
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as cmd_file:
            cmd_path = cmd_file.name
            json.dump(command, cmd_file)
        
        # 创建临时文件存储结果
        result_path = cmd_path + '.result'
        
        # 构建子进程命令
        runner_script = os.path.join(os.path.dirname(__file__), 'subprocess_runner.py')
        
        try:
            # 执行子进程
            subprocess.run(
                [python_path, runner_script, '--input', cmd_path, '--output', result_path],
                check=True
            )
            
            # 读取结果
            with open(result_path, 'r') as f:
                result = json.load(f)
            
            # 清理临时文件
            os.unlink(cmd_path)
            os.unlink(result_path)
            
            return result
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"子进程执行失败: {e}")
        except Exception as e:
            raise RuntimeError(f"执行命令时发生错误: {e}")
