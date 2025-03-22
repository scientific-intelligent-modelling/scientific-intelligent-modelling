"""CUDA Conda环境管理器，负责创建和管理CUDA环境"""

import os
import subprocess
import sys
import logging
from pathlib import Path

from .config_manager import config_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cuda_conda_manager")

class CudaCondaManager:
    """管理CUDA Conda环境"""
    
    def __init__(self):
        self.cuda_config = config_manager.get_config("cuda_config")
        self.conda_base_path = self._get_conda_base_path()
    
    def _get_conda_base_path(self):
        """获取conda安装路径"""
        try:
            # 使用conda info命令获取conda的安装信息
            result = subprocess.run(
                ["conda", "info", "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            import json
            conda_info = json.loads(result.stdout)
            return conda_info['conda_prefix']
        except Exception as e:
            logger.error(f"获取conda路径失败: {e}")
            return None
    
    def environment_exists(self, env_name):
        """检查指定的conda环境是否存在"""
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            import json
            env_list = json.loads(result.stdout)
            env_names = [os.path.basename(env) for env in env_list['envs']]
            return env_name in env_names
        except Exception as e:
            logger.error(f"检查环境失败: {e}")
            return False
    
    def create_cuda_environment(self, cuda_version):
        """创建指定CUDA版本的conda环境"""
        cuda_versions = self.cuda_config.get("cuda_versions", {})
        
        if cuda_version not in cuda_versions:
            logger.error(f"不支持的CUDA版本: {cuda_version}")
            return False
        
        env_config = cuda_versions[cuda_version]
        env_name = env_config.get("conda_env_name")
        
        if self.environment_exists(env_name):
            logger.info(f"环境{env_name}已存在")
            return True
        
        # 构建创建环境的命令
        channels = " ".join([f"-c {channel}" for channel in env_config.get("channels", [])])
        packages = " ".join(env_config.get("packages", []))
        python_version = env_config.get("python_version", "3.8")
        
        cmd = f"conda create -n {env_name} python={python_version} {packages} {channels} -y"
        
        try:
            logger.info(f"正在创建环境: {env_name}")
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"环境{env_name}创建成功")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"创建环境失败: {e}")
            return False
    
    def get_environment_python(self, env_name):
        """获取指定环境的Python可执行文件路径"""
        if not self.conda_base_path:
            return None
        
        if os.name == 'nt':  # Windows
            python_path = os.path.join(self.conda_base_path, "envs", env_name, "python.exe")
        else:  # Linux/MacOS
            python_path = os.path.join(self.conda_base_path, "envs", env_name, "bin", "python")
        
        if os.path.exists(python_path):
            return python_path
        return None

# 创建全局CUDA管理器实例
cuda_manager = CudaCondaManager()
