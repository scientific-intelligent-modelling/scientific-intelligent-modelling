"""增强的子进程管理，提供依赖隔离功能"""

import os
import sys
import json
import pickle
import subprocess
import tempfile
import logging
from pathlib import Path

from .config_manager import config_manager
from .cuda_conda_manager import cuda_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_subprocess")

class EnhancedSubprocess:
    """增强的子进程类，支持在隔离环境中运行代码"""
    
    def __init__(self, cuda_version=None, env_name=None):
        """
        初始化子进程管理器
        
        Args:
            cuda_version (str, optional): CUDA版本，如"11.1"
            env_name (str, optional): 环境名称，如果指定则优先使用
        """
        if env_name:
            self.env_name = env_name
        elif cuda_version:
            self.env_name = config_manager.get_cuda_env_name(cuda_version)
        else:
            # 使用base环境
            self.env_name = config_manager.get_cuda_env_name("base")
        
        self.python_executable = cuda_manager.get_environment_python(self.env_name)
        if not self.python_executable:
            logger.warning(f"未找到环境{self.env_name}的Python，尝试创建环境")
            cuda_manager.create_cuda_environment(cuda_version or "base")
            self.python_executable = cuda_manager.get_environment_python(self.env_name)
        
        self.timeout = config_manager.get_config("toolbox_config").get("subprocess_timeout", 3600)
    
    def run_code(self, code, **kwargs):
        """
        在隔离环境中运行Python代码
        
        Args:
            code (str): 要执行的Python代码
            **kwargs: 传递给代码的参数
        
        Returns:
            返回代码的执行结果
        """
        if not self.python_executable:
            raise RuntimeError(f"找不到环境{self.env_name}的Python可执行文件")
        
        # 创建临时文件存储代码和参数
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as code_file:
            code_file.write(code)
            code_file_path = code_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as args_file:
            pickle.dump(kwargs, args_file)
            args_file_path = args_file.name
        
        output_file_path = args_file_path + ".out"
        
        # 构建要执行的代码包装器，加载参数并保存结果
        wrapper_code = f"""
import sys
import pickle
import traceback

try:
    # 加载参数
    with open("{args_file_path}", "rb") as f:
        kwargs = pickle.load(f)
    
    # 执行主代码
    exec_globals = {{'__name__': '__main__'}}
    with open("{code_file_path}", "r") as f:
        code = f.read()
    exec(code, exec_globals)
    
    # 如果代码定义了main函数，执行它
    if 'main' in exec_globals:
        result = exec_globals['main'](**kwargs)
    else:
        result = None
    
    # 保存结果
    with open("{output_file_path}", "wb") as f:
        pickle.dump({{'status': 'success', 'result': result}}, f)

except Exception as e:
    error_info = {{
        'error': str(e),
        'traceback': traceback.format_exc()
    }}
    with open("{output_file_path}", "wb") as f:
        pickle.dump({{'status': 'error', 'error_info': error_info}}, f)
    sys.exit(1)
"""
        
        # 将包装器写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as wrapper_file:
            wrapper_file.write(wrapper_code)
            wrapper_file_path = wrapper_file.name
        
        try:
            # 执行包装器代码
            process = subprocess.run(
                [self.python_executable, wrapper_file_path],
                timeout=self.timeout,
                stderr=subprocess.PIPE
            )
            
            # 加载结果
            if os.path.exists(output_file_path):
                with open(output_file_path, "rb") as f:
                    result = pickle.load(f)
                
                if result['status'] == 'error':
                    logger.error(f"子进程执行错误: {result['error_info']['error']}")
                    logger.debug(f"错误详情: {result['error_info']['traceback']}")
                    raise RuntimeError(result['error_info']['error'])
                
                return result['result']
            else:
                stderr = process.stderr.decode('utf-8') if process.stderr else "未知错误"
                logger.error(f"子进程执行失败，未生成输出文件: {stderr}")
                raise RuntimeError(f"子进程执行失败: {stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"子进程执行超时 (>{self.timeout}秒)")
            raise TimeoutError(f"子进程执行超时，超过{self.timeout}秒")
        
        finally:
            # 清理临时文件
            for file_path in [code_file_path, args_file_path, output_file_path, wrapper_file_path]:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")


class ToolProxy:
    """工具代理类，用于透明地在隔离环境中调用工具函数"""
    
    def __init__(self, tool_name, cuda_version=None):
        """
        初始化工具代理
        
        Args:
            tool_name (str): 工具名称
            cuda_version (str, optional): CUDA版本
        """
        self.tool_name = tool_name
        
        if cuda_version is None:
            self.cuda_version = config_manager.get_tool_cuda_version(tool_name)
        else:
            self.cuda_version = cuda_version
            
        self.subprocess_runner = EnhancedSubprocess(cuda_version=self.cuda_version)
        
    def __getattr__(self, name):
        """拦截属性访问，返回可在隔离环境中执行的函数"""
        
        def wrapped_method(*args, **kwargs):
            """在隔离环境中执行工具方法"""
            # 构建要执行的代码
            code = f"""
from scientific_intelligent_modelling.{self.tool_name} import {name}

def main(*args, **kwargs):
    # 解包嵌套的参数结构
    args_list = kwargs.get('args', ())
    kwargs_dict = kwargs.get('kwargs', {{}})
    
    # 正确地将参数传递给实际函数
    return {name}(*args_list, **kwargs_dict)
"""
            return self.subprocess_runner.run_code(code, args=args, kwargs=kwargs)
        
        return wrapped_method
