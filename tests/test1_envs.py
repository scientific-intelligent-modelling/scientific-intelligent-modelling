from scientific_intelligent_modelling.config_manager import config_manager
from scientific_intelligent_modelling.environment_manager import environment_manager

print(f"sim_cuda_11_8的配置: {config_manager.get_env_config('sim_cuda_11_8')}")

print("\nCUDA管理器测试:")
print(f"conda基础路径: {environment_manager.conda_base_path}")

environment_manager.create_conda_environment("sim_cuda_11_8")

tool_name = "sklearn_tool"
env_name = config_manager.get_env_name_by_tool(tool_name)
print(f"{tool_name}对应的环境名称: {env_name}")
print(f"环境{env_name}是否存在: {environment_manager.environment_exists(env_name)}")

# cuda_version = "11.1"
# env_name = config_manager.get_cuda_env_name(cuda_version)
# print(f"CUDA {cuda_version}对应的环境名称: {env_name}")
# print(f"环境{env_name}是否存在: {cuda_manager.environment_exists(env_name)}")
