"""Conda环境管理器，负责创建和删除环境"""

import subprocess
import os
import sys
import json
import logging
from pathlib import Path
from .config_manager import config_manager


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("conda_env_manager")

class EnvManager:
    """管理Conda环境的创建和删除"""
    
    def __init__(self):
        self.config_manager = config_manager
        self.env_config = self.config_manager.get_config("envs_config")
        self.conda_base_path = self._get_conda_base_path()

    def _get_conda_base_path(self):
        """获取conda安装路径"""
        try:
            # 使用conda info命令获取conda的安装信息
            returncode, stdout, stderr = self.run_command(
                ["conda", "info", "--json"], show_output=False
            )
            import json
            conda_info = json.loads(stdout)
            # ANSI 转义序列
            GREEN = "\033[92m"  # 绿色
            RESET = "\033[0m"  # 重置颜色
            print(f"{GREEN}conda_prefix: {conda_info['conda_prefix']}{RESET}")
            return conda_info['conda_prefix']
        except Exception as e:
            logger.error(f"获取conda路径失败: {e}")
            return None
        
    def check_conda_installed(self):
        """检查是否安装Conda并可在PATH中使用"""
        try:
            returncode, stdout, stderr = self.run_command(["conda", "--version"])
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("错误: Conda未安装或不在PATH中。")
            return False
    
    def list_environments(self):
        """打印可用环境列表"""
        env_list = self.env_config.get("env_list", {})
        if not env_list:
            print("配置中未找到环境。")
            return
        
        print("\n**可用环境:**")
        for i, (env_name, env_details) in enumerate(env_list.items(), 1):
            python_version = env_details.get("python_version", "未指定")
            comments = env_details.get("comments", "")
            
            print(f"{i}. {env_name} (Python {python_version})")
            if comments:
                print(f"   备注: {comments}")
            
            # # 打印包信息
            # packages = env_details.get("packages", [])
            # if packages:
            #     print(f"   包: {', '.join(packages)}")
            
            # # 打印渠道信息
            # channels = env_details.get("channels", [])
            # if channels:
            #     print(f"   渠道: {', '.join(channels)}")
                
            # 打印安装后命令
            # post_commands = env_details.get("post_install_commands", [])
            # if post_commands:
            #     print(f"   安装后命令:")
            #     for cmd in post_commands:
            #         print(f"     - {cmd}")
            
            print()  # 空行提高可读性
    
    def get_env_path(self, env_name):
        """获取Conda环境的路径"""
        try:
            returncode, stdout, stderr = self.run_command(
                ["conda", "info", "--envs", "--json"], show_output=False
            )
            env_data = json.loads(stdout)
            envs = env_data.get("envs", [])
            # ANSI 转义序列
            GREEN = "\033[92m"  # 绿色
            RESET = "\033[0m"  # 重置颜色
            print(f"{GREEN}envs: {envs}{RESET}")
            # 寻找指定名称的环境
            for env_path in envs:
                if os.path.basename(env_path) == env_name:
                    return env_path
                
            return None
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"获取环境路径时出错: {e}")
            return None
    def get_env_python(self, conda_env_name):
        """获取指定环境的Python可执行文件路径"""
        if not self.conda_base_path:
            return None
        
        if os.name == 'nt':  # Windows
            python_path = os.path.join(self.conda_base_path, "envs", conda_env_name, "python.exe")
        else:  # Linux/MacOS
            python_path = os.path.join(self.conda_base_path, "envs", conda_env_name, "bin", "python")
        
        if os.path.exists(python_path):
            return python_path
        return None

    def get_post_commands_marker_path(self, env_name):
        """获取记录已执行后处理命令的标记文件路径"""
        env_path = self.get_env_path(env_name)
        if not env_path:
            return None
        return os.path.join(env_path, ".post_commands_executed")
    
    def record_post_command_execution(self, env_name, command):
        """记录后处理命令已成功执行"""
        marker_path = self.get_post_commands_marker_path(env_name)
        if not marker_path:
            return False
            
        executed_commands = set()
        
        # 如果标记文件存在，读取已执行的命令
        if os.path.exists(marker_path):
            with open(marker_path, 'r') as f:
                executed_commands = set(line.strip() for line in f)
        
        # 添加新执行的命令
        executed_commands.add(command)
        
        # 写入更新后的命令列表
        with open(marker_path, 'w') as f:
            for cmd in executed_commands:
                f.write(f"{cmd}\n")
        
        return True
    
    def get_executed_post_commands(self, env_name):
        """获取已执行的后处理命令列表"""
        marker_path = self.get_post_commands_marker_path(env_name)
        if not marker_path or not os.path.exists(marker_path):
            return set()
            
        with open(marker_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    def check_environment(self, env_name):
        """
        检查指定的环境是否存在并满足条件。
        
        返回:
            - (True, None): 环境存在且满足所有条件
            - (False, "reason"): 环境不存在或不满足条件，reason 是原因说明
        """
        # 获取环境配置
        env_config = self.config_manager.get_env_config(env_name)
        if not env_config:
            return False, f"配置中未找到环境 '{env_name}'"
        
        # 检查环境是否存在
        existing_envs = self.get_existing_environments()
        if env_name not in existing_envs:
            return False, f"环境 '{env_name}' 不存在"
        
        # 检查环境中是否安装了所需的conda包
        conda_packages = env_config.get("conda_packages", [])
        if conda_packages:
            try:
                returncode, stdout, stderr = self.run_command(
                    ["conda", "list", "--name", env_name]
                )
                
                installed_packages = []
                for line in stdout.splitlines():
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            pkg_name = parts[0]
                            installed_packages.append(pkg_name)
                
                # 检查所需的conda包是否都已安装
                missing_packages = []
                for package in conda_packages:
                    # 处理包名可能包含版本号的情况
                    package_name = package.split('=')[0] if '=' in package else package
                    if package_name not in installed_packages:
                        missing_packages.append(package_name)
                
                if missing_packages:
                    return False, f"环境 '{env_name}' 缺少必需的conda包: {', '.join(missing_packages)}"
                        
            except subprocess.CalledProcessError as e:
                return False, f"检查环境 '{env_name}' 的conda包时出错: {e}"
        
        # 检查环境中是否安装了所需的pip包
        pip_packages = env_config.get("pip_packages", [])
        if pip_packages:
            try:
                returncode, stdout, stderr = self.run_command(
                    ["conda", "run", "-n", env_name, "uv", "pip", "list"]
                )
                
                installed_pip_packages = []
                for line in stdout.splitlines():
                    if line and not line.startswith('Package') and not line.startswith('-'):
                        parts = line.split()
                        if len(parts) >= 1:
                            pkg_name = parts[0].lower()  # pip包名通常是小写的
                            installed_pip_packages.append(pkg_name)
                
                # 检查所需的pip包是否都已安装
                missing_pip_packages = []
                for package in pip_packages:
                    # 处理包名可能包含版本号的情况
                    package_name = package.split('=')[0] if '=' in package else package
                    package_name = package_name.lower()  # 转换为小写以进行比较
                    if package_name not in installed_pip_packages:
                        missing_pip_packages.append(package_name)
                
                if missing_pip_packages:
                    return False, f"环境 '{env_name}' 缺少必需的pip包: {', '.join(missing_pip_packages)}"
                        
            except subprocess.CalledProcessError as e:
                return False, f"检查环境 '{env_name}' 的pip包时出错: {e}"
        
        # 检查Python版本是否匹配
        required_python = env_config.get("python_version")
        if required_python:
            try:
                returncode, stdout, stderr = self.run_command(
                    ["conda", "run", "-n", env_name, "python", "--version"]
                )
                python_version = stdout.strip()
                # Python版本输出通常是"Python X.Y.Z"格式
                if required_python not in python_version:
                    return False, f"环境 '{env_name}' Python版本不匹配，需要 {required_python}，实际是 {python_version}"
            except subprocess.CalledProcessError as e:
                return False, f"检查环境 '{env_name}' 的Python版本时出错: {e}"
        
        # 检查后处理命令是否已执行
        post_commands = env_config.get("post_install_commands", [])
        if post_commands:
            executed_commands = self.get_executed_post_commands(env_name)
            missing_commands = [cmd for cmd in post_commands if cmd not in executed_commands]
            
            if missing_commands:
                return False, f"环境 '{env_name}' 缺少必需的后处理命令: {', '.join(missing_commands)}"
        
        return True, None

    def check_all_environments(self):
        """
        检查所有配置的环境的状态。
        每检查一个环境就输出一个，并且在全部检查完后进行汇总。
        
        返回:
            - configured_envs: 已经配置好的环境列表
            - unconfigured_envs: 未配置好的环境列表，每个元素是 (env_name, reason) 的元组
        """
        env_list = self.env_config.get("env_list", {})
        configured_envs = []
        unconfigured_envs = []
        
        print("开始检查所有环境...")
        for env_name in env_list:
            exists, reason = self.check_environment(env_name)
            if exists:
                configured_envs.append(env_name)
                print(f"环境 '{env_name}' 配置正确。")
            else:
                unconfigured_envs.append((env_name, reason))
                print(f"环境 '{env_name}' 配置失败: {reason}")
        
        # 汇总结果
        print("\n检查完成，汇总结果:")
        print(f"配置正确的环境: {len(configured_envs)} 个")
        if configured_envs:
            print("  - " + "\n  - ".join(configured_envs))
        
        print(f"配置失败的环境: {len(unconfigured_envs)} 个")
        if unconfigured_envs:
            for env_name, reason in unconfigured_envs:
                print(f"  - {env_name}: {reason}")
        
        return configured_envs, unconfigured_envs


    def create_environment(self, env_name):
        """根据配置创建Conda环境，如果环境已存在且满足条件则跳过创建"""
        # 先检查环境是否已存在且满足条件
        exists, reason = self.check_environment(env_name)
        if exists:
            print(f"环境 '{env_name}' 已存在且满足所有条件，跳过创建。")
            return True
        
        # 如果环境存在但不满足条件，可以选择先删除再创建，或者提醒用户
        if reason and "不存在" not in reason:
            print(f"警告: {reason}")
            choice = input(f"环境 '{env_name}' 存在但不满足要求，是否重新创建? (y/n): ").strip().lower()
            if choice != 'y':
                print(f"跳过创建环境 '{env_name}'")
                return False
            
            # 先删除现有环境
            self.delete_environment(env_name)
        
        # 以下是创建环境的逻辑
        env_details = self.config_manager.get_env_config(env_name)
        if not env_details:
            print(f"错误: 配置中未找到环境'{env_name}'。")
            return False

        python_version = env_details.get("python_version", "3.10")
        conda_packages = env_details.get("conda_packages", [])
        pip_packages = env_details.get("pip_packages", [])
        channels = env_details.get("channels", [])
        post_commands = env_details.get("post_install_commands", [])

        # 构建conda create命令
        cmd = ["conda", "create", "-y", "-n", env_name, f"python={python_version}"]

        # 添加指定的conda包
        cmd.extend(conda_packages)

        # 添加指定的渠道
        for channel in channels:
            cmd.extend(["-c", channel])
        
        print(f"创建环境'{env_name}'...")
        try:
            # 执行conda create命令
            returncode, stdout, stderr = self.run_command(cmd)
            
            # 安装pip包
            if pip_packages:
                print(f"为'{env_name}'安装pip包...")
                all_pip_packages_installed = True
                
                for package in pip_packages:
                    print(f"正在安装pip包: {package}...")
                    pip_cmd = ["conda", "run", "-n", env_name, "uv", "pip", "install", package]
                    
                    try:
                        # 使用直接输出到终端的方式运行命令
                        returncode, stdout, stderr = self.run_command(
                            pip_cmd
                        )
                        print(f"pip包 {package} 安装完成。")
                    except subprocess.CalledProcessError as e:
                        print(f"安装pip包 {package} 时出错: {e}")
                        all_pip_packages_installed = False
                        
                        # 可以选择是否继续安装其他包
                        if input(f"安装 {package} 失败。是否继续安装其他包? (y/n): ").lower() != 'y':
                            return False
                
                if not all_pip_packages_installed:
                    print("警告: 某些pip包安装失败")
                    if input("是否继续环境创建过程? (y/n): ").lower() != 'y':
                        return False
                else:
                    print("所有pip包安装完成。")

            
            # 执行安装后命令
            if post_commands:
                print(f"为'{env_name}'执行安装后命令...")
                all_commands_succeeded = True
                
                for command in post_commands:
                    print(f"\n==== 开始执行后处理命令: {command} ====")
                    try:
                        # 将命令按照引号和空格正确拆分
                        import shlex
                        command_parts = shlex.split(command)
                        
                        # 使用conda run在新环境中执行命令，不捕获输出而是直接显示
                        full_command = ["conda", "run", "-n", env_name] + command_parts
                        print(f"完整命令: {' '.join(full_command)}")
                        
                        # 执行命令，直接将输出显示到终端
                        returncode, stdout, stderr = self.run_command(
                            full_command
                        )
                        
                        print(f"==== 后处理命令执行成功 ====")
                        
                        # 记录成功执行的命令
                        self.record_post_command_execution(env_name, command)
                        
                    except subprocess.CalledProcessError as e:
                        all_commands_succeeded = False
                        print(f"==== 后处理命令执行失败 ====")
                        print(f"错误信息: {e}")
                        
                        # 可以选择是否继续执行剩余命令
                        if input("命令执行失败。是否继续执行下一个命令? (y/n): ").lower() != 'y':
                            break
                
                if all_commands_succeeded:
                    print(f"\n所有后处理命令执行成功！")
                else:
                    print(f"\n警告: 某些后处理命令执行失败")
            return True
        except subprocess.CalledProcessError as e:
            print(f"创建环境'{env_name}'时出错: {e}")
            return False

    
    def delete_environment(self, env_name):
        """删除Conda环境"""
        # 删除环境前先记录标记文件路径，以便我们可以在删除环境后删除该文件
        marker_path = self.get_post_commands_marker_path(env_name)
        
        print(f"删除环境'{env_name}'...")
        try:
            returncode, stdout, stderr = self.run_command(["conda", "env", "remove", "-y", "-n", env_name])
            print(f"环境'{env_name}'删除成功!")
            
            # 尝试删除标记文件（如果环境已被删除，此操作可能会失败，但这没关系）
            if marker_path and os.path.exists(marker_path):
                try:
                    os.remove(marker_path)
                except (OSError, IOError):
                    pass
                    
            return True
        except subprocess.CalledProcessError as e:
            print(f"删除环境'{env_name}'时出错: {e}")
            return False
    
    def get_existing_environments(self):
        """获取现有Conda环境列表"""
        returncode, stdout, stderr = self.run_command(["conda", "env", "list"])
        existing_envs = []
        for line in stdout.splitlines():
            if line and not line.startswith('#'):
                env_name = line.split()[0]
                if env_name != "base":  # 排除base环境
                    existing_envs.append(env_name)
        return existing_envs
    
    def run_command(self, command, show_output=True):
        # ANSI 转义序列
        GRAY = "\033[90m"  # 亮灰色
        RESET = "\033[0m"  # 重置颜色
        RED = "\033[91m"  # 红色
        LIGHT_GREEN = "\033[92m"  # 亮绿色
        Green = "\033[32m"  # 绿色
        YELLOW = "\033[93m"
        BLUE = "\033[94m"  # 蓝色
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"  # 青色
        WHITE = "\033[97m"
        # 启动子进程
        process = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,# 将输出作为文本处理
                                bufsize=0  # 关键参数：设置为0禁用缓冲
                                )  

        # 存储所有输出
        stdout_all = []
        stderr_all = []

        if show_output:
                
            # 实时读取输出
            for line in process.stdout:
                stdout_all.append(line)  # 收集标准输出
                print(f"{CYAN}{line}{RESET}", end='')  # 打印标准输出，颜色为灰色

        # 等待进程结束
        stdout, stderr = process.communicate()  # 捕获剩余输出
        if stdout:
            stdout_all.append(stdout)  # 收集最终的标准输出
            if show_output:
                print(f"{CYAN}{stdout}{RESET}", end='')
        if stderr:
            stderr_all.append(stderr)  # 收集标准错误
            if show_output:
                print(f"{BLUE}{stderr}{RESET}", end='')

        return process.returncode, ''.join(stdout_all), ''.join(stderr_all)  # 返回返回码及所有输出
    
    def run_cli(self):
        """运行命令行界面"""
        if not self.check_conda_installed():
            print("请安装Conda或确保它已正确配置在您的PATH中。")
            return
        
        env_list = self.env_config.get("env_list", {})
        
        if not env_list:
            print("配置中未找到环境。退出。")
            return
        
        while True:
            print("\n" + "="*50)
            print("**Conda环境管理器**")
            print("="*50)
            print("1. 创建环境")
            print("2. 删除环境")
            print("3. 检查环境状态")
            print("4. 退出")
            
            choice = input("\n请输入您的选择 (1-4): ").strip()
            
            if choice == "1":
                self.list_environments()
                
                env_names = list(env_list.keys())
                env_input = input("\n请输入要创建的环境的编号或名称 (输入'all'创建所有环境): ").strip()
                
                if env_input.lower() == 'all':
                    print("\n创建所有环境...")
                    for env_name in env_names:
                        self.create_environment(env_name)
                else:
                    try:
                        # 检查输入是否为数字
                        env_idx = int(env_input) - 1
                        if 0 <= env_idx < len(env_names):
                            env_name = env_names[env_idx]
                            self.create_environment(env_name)
                        else:
                            print("无效的环境编号。")
                    except ValueError:
                        # 输入不是数字，尝试作为环境名称
                        if env_input in env_names:
                            self.create_environment(env_input)
                        else:
                            print(f"配置中未找到环境'{env_input}'。")
            
            elif choice == "2":
                existing_envs = self.get_existing_environments()
                
                if not existing_envs:
                    print("没有可删除的conda环境 (不包括base)。")
                    continue
                
                print("\n**现有conda环境:**")
                for i, env_name in enumerate(existing_envs, 1):
                    print(f"{i}. {env_name}")
                
                env_input = input("\n请输入要删除的环境的编号或名称 (输入'all'删除所有环境): ").strip()
                
                if env_input.lower() == 'all':
                    confirm = input("您确定要删除所有环境吗？此操作无法撤销! (y/n): ").strip().lower()
                    if confirm == 'y':
                        print("\n删除所有环境...")
                        for env_name in existing_envs:
                            self.delete_environment(env_name)
                    else:
                        print("操作已取消。")
                else:
                    try:
                        # 检查输入是否为数字
                        env_idx = int(env_input) - 1
                        if 0 <= env_idx < len(existing_envs):
                            env_name = existing_envs[env_idx]
                            self.delete_environment(env_name)
                        else:
                            print("无效的环境编号。")
                    except ValueError:
                        # 输入不是数字，尝试作为环境名称
                        if env_input in existing_envs:
                            self.delete_environment(env_input)
                        else:
                            print(f"未找到环境'{env_input}'。")
            
            elif choice == "3":
                configured_envs, unconfigured_envs = self.check_all_environments()
            
            elif choice == "4":
                print("退出Conda环境管理器。再见!")
                break
            
            else:
                print("无效选择。请输入1到4之间的数字。")


env_manager = EnvManager()

def main():
    """主函数"""
    # 导入ConfigManager
    env_manager.run_cli()

if __name__ == "__main__":
    main()
