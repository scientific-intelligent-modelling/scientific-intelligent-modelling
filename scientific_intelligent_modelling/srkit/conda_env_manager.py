"""Conda环境管理器，负责创建和删除环境"""

import subprocess
import os
import sys
import json
from pathlib import Path
from .config_manager import config_manager

class EnvManager:
    """管理Conda环境的创建和删除"""
    
    def __init__(self):
        self.config_manager = config_manager
        self.env_config = self.config_manager.get_config("envs_config")
    
    def check_conda_installed(self):
        """检查是否安装Conda并可在PATH中使用"""
        try:
            subprocess.run(["conda", "--version"], capture_output=True, check=True)
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
            
            # 打印包信息
            packages = env_details.get("packages", [])
            if packages:
                print(f"   包: {', '.join(packages)}")
            
            # 打印渠道信息
            channels = env_details.get("channels", [])
            if channels:
                print(f"   渠道: {', '.join(channels)}")
                
            # 打印安装后命令
            post_commands = env_details.get("post_install_commands", [])
            if post_commands:
                print(f"   安装后命令:")
                for cmd in post_commands:
                    print(f"     - {cmd}")
            
            print()  # 空行提高可读性
    
    def get_env_path(self, env_name):
        """获取Conda环境的路径"""
        try:
            result = subprocess.run(
                ["conda", "info", "--envs", "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            env_data = json.loads(result.stdout)
            envs = env_data.get("envs", [])
            
            # 寻找指定名称的环境
            for env_path in envs:
                if os.path.basename(env_path) == env_name:
                    return env_path
                
            return None
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"获取环境路径时出错: {e}")
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
        
        # 检查环境中是否安装了所需的包
        required_packages = env_config.get("packages", [])
        
        if required_packages:
            # 获取环境中已安装的包
            try:
                result = subprocess.run(
                    ["conda", "list", "--name", env_name], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                
                installed_packages = []
                for line in result.stdout.splitlines():
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            pkg_name = parts[0]
                            installed_packages.append(pkg_name)
                
                # 检查所需的包是否都已安装
                missing_packages = []
                for package in required_packages:
                    # 处理包名可能包含版本号的情况
                    package_name = package.split('=')[0] if '=' in package else package
                    if package_name not in installed_packages:
                        missing_packages.append(package_name)
                
                if missing_packages:
                    return False, f"环境 '{env_name}' 缺少必需的包: {', '.join(missing_packages)}"
                        
            except subprocess.CalledProcessError as e:
                return False, f"检查环境 '{env_name}' 的包时出错: {e}"
        
        # 检查Python版本是否匹配
        required_python = env_config.get("python_version")
        if required_python:
            try:
                result = subprocess.run(
                    ["conda", "run", "-n", env_name, "python", "--version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                python_version = result.stdout.strip()
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
        
        返回:
            - configured_envs: 已经配置好的环境列表
            - unconfigured_envs: 未配置好的环境列表，每个元素是 (env_name, reason) 的元组
        """
        env_list = self.env_config.get("env_list", {})
        configured_envs = []
        unconfigured_envs = []
        
        for env_name in env_list:
            exists, reason = self.check_environment(env_name)
            if exists:
                configured_envs.append(env_name)
            else:
                unconfigured_envs.append((env_name, reason))
        
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
        packages = env_details.get("packages", [])
        channels = env_details.get("channels", [])
        post_commands = env_details.get("post_install_commands", [])
    
        # 构建conda create命令
        cmd = ["conda", "create", "-y", "-n", env_name, f"python={python_version}"]
        
        # 添加指定的渠道
        for channel in channels:
            cmd.extend(["-c", channel])
        
        # 添加指定的包
        cmd.extend(packages)
        
        print(f"创建环境'{env_name}'...")
        try:
            # 执行conda create命令
            subprocess.run(cmd, check=True)
            
            # 执行安装后命令
            if post_commands:
                print(f"为'{env_name}'执行安装后命令...")
                all_commands_succeeded = True
                
                for command in post_commands:
                    print(f"执行: {command}")
                    try:
                        # 使用conda run在新环境中执行命令，并捕获输出
                        result = subprocess.run(
                            ["conda", "run", "-n", env_name] + command.split(), 
                            check=True, 
                            capture_output=True, 
                            text=True
                        )
                        print(f"命令执行成功！输出:\n{result.stdout}")
                        
                        # 记录成功执行的命令
                        self.record_post_command_execution(env_name, command)
                        
                    except subprocess.CalledProcessError as e:
                        all_commands_succeeded = False
                        print(f"命令 '{command}' 执行失败: {e}")
                        print(f"错误输出: {e.stderr}")
                        
                        # 可以选择是否继续执行剩余命令
                        if input("是否继续执行剩余命令? (y/n): ").lower() != 'y':
                            break
                
                if all_commands_succeeded:
                    print(f"所有后处理命令执行成功！")
                else:
                    print(f"警告: 某些后处理命令执行失败")
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
            subprocess.run(["conda", "env", "remove", "-y", "-n", env_name], check=True)
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
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        existing_envs = []
        for line in result.stdout.splitlines():
            if line and not line.startswith('#'):
                env_name = line.split()[0]
                if env_name != "base":  # 排除base环境
                    existing_envs.append(env_name)
        return existing_envs
    
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
                
                print("\n**环境状态检查结果:**")
                print("\n已配置好的环境:")
                if configured_envs:
                    for i, env_name in enumerate(configured_envs, 1):
                        print(f"{i}. {env_name}")
                else:
                    print("  [无]")
                
                print("\n未配置好的环境:")
                if unconfigured_envs:
                    for i, (env_name, reason) in enumerate(unconfigured_envs, 1):
                        print(f"{i}. {env_name} - 原因: {reason}")
                else:
                    print("  [无]")
            
            elif choice == "4":
                print("退出Conda环境管理器。再见!")
                break
            
            else:
                print("无效选择。请输入1到4之间的数字。")

def main():
    """主函数"""
    # 导入ConfigManager
    env_manager = EnvManager()
    env_manager.run_cli()

if __name__ == "__main__":
    main()
