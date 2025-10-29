# srkit/regressor.py
import os
import json
import tempfile
import subprocess
import numpy as np

from .config_manager import config_manager
from .conda_env_manager import env_manager

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
        self.serialized_model = None
        
        # 使用config_manager获取环境名称
        self.env_name = config_manager.get_env_name_by_tool(tool_name)
        if not self.env_name:
            raise ValueError(f"未找到工具 '{tool_name}' 的环境配置")
        
        # 检查环境是否存在
        if not env_manager.check_environment(self.env_name):
            print(f"环境 '{self.env_name}' 不存在，正在创建...")
            success = env_manager.create_environment(self.env_name)
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
        self.serialized_model = result.get('serialized_model', {})
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
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 准备数据
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # 创建命令
        command = {
            'action': 'predict',
            'data': {'X': X},
            'serialized_model': self.serialized_model,
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
    
    def get_optimal_equation(self):
        """
        获取模型学习到的最优符号方程
        
        返回:
            equation: 符号方程的字符串表示
        """
        # 检查模型是否已训练
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 创建命令
        command = {
            'action': 'get_optimal_equation',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # 执行命令并获取结果
        result = self._execute_subprocess(command)
        
        # 检查结果
        if 'error' in result:
            raise RuntimeError(f"获取方程失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 返回方程
        return result.get('equation', '')
    

    def get_total_equations(self, n=None):
        """
        获取模型学习到的所有符号方程
        
        返回:
            equations: 符号方程的字符串表示列表
        """
        # 检查模型是否已训练
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 创建命令
        command = {
            'action': 'get_total_equations',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # 执行命令并获取结果
        result = self._execute_subprocess(command)
        
        # 检查结果
        if 'error' in result:
            raise RuntimeError(f"获取所有方程失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 返回方程列表
        return result.get('equations', [])


    def __str__(self):
        """
        返回模型的字符串表示
        
        返回:
            model_str: 模型的字符串表示
        """
        # 基础信息
        model_str = f"SymbolicRegressor(tool='{self.tool_name}'"
        
        # 添加主要参数
        for key, value in self.params.items():
            if key in ['population_size', 'generations', 'n_components', 'binary_operators', 
                    'unary_operators', 'parsimony_coefficient', 'max_samples', 'random_state']:
                model_str += f", {key}={value}"
        
        model_str += ")"
        
        # 如果模型已训练，添加方程
        if self.serialized_model is not None:
            try:
                equation = self.get_equation()
                model_str += f"\n最佳方程: {equation}"
            except Exception as e:
                model_str += f"\n模型已训练，但无法获取方程: {str(e)}"
        else:
            model_str += "\n模型尚未训练"
        
        return model_str


    def _execute_subprocess(self, command):
        """执行子进程命令"""
        # 获取Python解释器路径
        python_path = env_manager.get_env_python(self.env_name)
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
            # 执行子进程（无缓冲），并实时转发其 stdout/stderr 到当前进程
            env = os.environ.copy()
            env.setdefault('PYTHONUNBUFFERED', '1')
            proc = subprocess.Popen(
                [python_path, '-u', runner_script, '--input', cmd_path, '--output', result_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env
            )
            # 实时并发读取 stdout/stderr，避免管道阻塞
            assert proc.stdout is not None and proc.stderr is not None
            try:
                import selectors
                sel = selectors.DefaultSelector()
                sel.register(proc.stdout, selectors.EVENT_READ)
                sel.register(proc.stderr, selectors.EVENT_READ)
                while True:
                    if proc.poll() is not None and not sel.get_map():
                        break
                    events = sel.select(timeout=0.1)
                    if not events and proc.poll() is not None:
                        break
                    for key, _ in events:
                        line = key.fileobj.readline()
                        if line:
                            print(line, end='')
                        else:
                            # EOF: 取消注册
                            sel.unregister(key.fileobj)
                ret = proc.wait()
            except Exception:
                # 兜底：逐流读取
                for line in proc.stdout:
                    print(line, end='')
                for line in proc.stderr:
                    print(line, end='')
                ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, proc.args)

            # 读取结果
            with open(result_path, 'r') as f:
                result = json.load(f)

            # 清理临时文件
            os.unlink(cmd_path)
            os.unlink(result_path)

            return result
        except subprocess.CalledProcessError as e:
            # 读取结果
            with open(result_path, 'r') as f:
                result = json.load(f)
            raise RuntimeError(f"子进程执行失败: {e}\n{result.get('traceback', '')}")
        except Exception as e:
            # 读取结果
            with open(result_path, 'r') as f:
                result = json.load(f)
            raise RuntimeError(f"执行命令时发生错误: {e}\n{result.get('traceback', '')}")
