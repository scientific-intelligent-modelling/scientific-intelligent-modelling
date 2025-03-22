from scientific_intelligent_modelling.config_manager import config_manager
from scientific_intelligent_modelling.cuda_conda_manager import cuda_manager

print("配置测试:")
print(f"默认CUDA版本: {config_manager.get_config('toolbox_config').get('default_cuda')}")
print(f"torch_1_8_tool的CUDA版本: {config_manager.get_tool_cuda_version('torch_1_8_tool')}")

print("\nCUDA管理器测试:")
print(f"conda基础路径: {cuda_manager.conda_base_path}")

cuda_manager.create_cuda_environment("11.8")
print(f"环境{config_manager.get_cuda_env_name('11.8')}是否存在: {cuda_manager.environment_exists(config_manager.get_cuda_env_name('11.8'))}")
# # 检查环境
# cuda_version = "11.1"
# env_name = config_manager.get_cuda_env_name(cuda_version)
# print(f"CUDA {cuda_version}对应的环境名称: {env_name}")
# print(f"环境{env_name}是否存在: {cuda_manager.environment_exists(env_name)}")
