"""配置管理器，负责加载和访问配置"""

import os
import json
from pathlib import Path

class ConfigManager:
    """管理工具箱的配置"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            # 如果未指定配置目录，使用包内的config目录
            self.config_dir = Path(__file__).parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """加载所有配置文件"""
        for config_file in self.config_dir.glob("*.json"):
            config_name = config_file.stem
            with open(config_file, 'r') as f:
                self.configs[config_name] = json.load(f)
        print(self.configs)
    
    def get_config(self, config_name):
        """获取指定配置"""
        return self.configs.get(config_name, {})
    
    def get_tool_cuda_version(self, tool_name):
        """获取工具对应的CUDA版本"""
        toolbox_config = self.get_config("toolbox_config")
        tool_mapping = toolbox_config.get("tool_mapping", {})
        
        if tool_name in tool_mapping:
            return tool_mapping[tool_name].get("cuda")
        return toolbox_config.get("default_cuda")
    
    def get_cuda_env_name(self, cuda_version):
        """获取CUDA版本对应的conda环境名称"""
        cuda_config = self.get_config("cuda_config")
        cuda_versions = cuda_config.get("cuda_versions", {})
        
        if cuda_version in cuda_versions:
            return cuda_versions[cuda_version].get("conda_env_name")
        return None

# 创建全局配置管理器实例
config_manager = ConfigManager()
