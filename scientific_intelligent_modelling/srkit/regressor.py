# srkit/regressor.py
import os
import re
import json
import tempfile
import subprocess
import time
from datetime import datetime, timezone
import numpy as np

from .config_manager import config_manager
from .conda_env_manager import env_manager

class SymbolicRegressor:
    def __init__(self, tool_name, problem_name: str | None = None, experiments_dir: str | None = None, seed: int = 1314, **kwargs):
        """
        初始化符号回归器
        
        参数:
            tool_name: 要使用的工具名称 (例如 'gplearn', 'pysr')
            problem_name: 问题/数据集名称，用于实验命名与目录组织
            experiments_dir: 实验目录根路径（默认在当前工作目录下的 './experiments'）
            seed: 随机种子（默认 1314），也用于实验目录命名
            **kwargs: 传递给实际工具的参数
        """
        self.tool_name = tool_name
        self.params = kwargs
        self.serialized_model = None
        
        # 基本实验信息
        self.problem_name = problem_name or "problem"
        self.seed = int(seed) if seed is not None else 1314
        # 默认 experiments 根目录：相对于调用者当前工作目录
        self.experiments_root = experiments_dir or os.path.join(os.getcwd(), "experiments")

        # 创建实验目录：{problem}_{tool}_seed{seed}_YYYYMMDD-HHMMSS
        def _slugify(text: str) -> str:
            try:
                return re.sub(r"[^A-Za-z0-9_\-]+", "-", str(text)).strip("-") or "item"
            except Exception:
                return "item"

        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_name = f"{_slugify(self.problem_name)}_{_slugify(self.tool_name)}_seed{self.seed}_{ts}"
        try:
            os.makedirs(self.experiments_root, exist_ok=True)
            self.experiment_dir = os.path.join(self.experiments_root, exp_name)
            os.makedirs(self.experiment_dir, exist_ok=True)
        except Exception:
            # 若目录创建失败，回退到临时目录，但不影响后续运行
            tmp_root = tempfile.gettempdir()
            self.experiment_dir = os.path.join(tmp_root, exp_name)
            os.makedirs(self.experiment_dir, exist_ok=True)

        # 写入最小元信息（manifest.json）：created 状态 + 基本配置
        try:
            self._write_initial_manifest()
        except Exception:
            # 元信息写入失败不影响主流程
            pass
        
        # 使用config_manager获取环境名称
        self.env_name = config_manager.get_env_name_by_tool(tool_name)
        if not self.env_name:
            raise ValueError(f"未找到工具 '{tool_name}' 的环境配置")
        
        # 快速路径：仅在无法定位 Python 可执行文件时，才进行较重的环境检查/创建
        # 这样可以避免每次实例化都调用昂贵的 conda 检查（如 pip freeze、python --version 等）
        python_path = env_manager.get_env_python(self.env_name)
        if not python_path:
            # 无法直接定位到 python，退回到完整检查/创建逻辑
            exists, reason = env_manager.check_environment(self.env_name)
            if not exists:
                print(f"环境 '{self.env_name}' 不存在或未就绪，正在创建...")
                success = env_manager.create_environment(self.env_name)
                if not success:
                    raise RuntimeError(f"无法创建环境 '{self.env_name}'：{reason}")
    
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
        
        # 标记实验进入 running
        try:
            self._update_manifest(status="running")
        except Exception:
            pass

        # 执行命令并获取结果
        try:
            result = self._execute_subprocess(command)
        except Exception:
            # 失败状态落盘后再抛出
            try:
                self._update_manifest(status="failed")
            except Exception:
                pass
            raise
        
        # 检查结果
        if 'error' in result:
            try:
                self._update_manifest(status="failed")
            except Exception:
                pass
            raise RuntimeError(f"训练失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 保存模型状态
        self.serialized_model = result.get('serialized_model', {})
        # 成功状态
        try:
            self._update_manifest(status="success")
        except Exception:
            pass
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
        
        for key, value in self.params.items():
            model_str += f", {key}={value}"
        model_str += ")"

        # 若已训练，尝试通过子进程获取最佳方程与参数
        if self.serialized_model is not None:
            try:
                equation = self.get_optimal_equation()
                if equation:
                    model_str += f"\n最佳方程:\n{equation}"
            except Exception as e:
                model_str += f"\n模型已训练，但无法获取方程: {str(e)}"
            try:
                params = self.get_fitted_params()
                if params is not None:
                    model_str += f"\n最佳参数: {params}"
            except Exception:
                pass
        else:
            model_str += "\n模型尚未训练"
        return model_str

    def get_fitted_params(self):
        """获取最佳方程的训练期拟合参数（若算法支持）。"""
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        command = {
            'action': 'get_fitted_params',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"获取参数失败: {result['message']}\n{result.get('traceback', '')}")
        return result.get('params')

    def get_total_equations_with_params(self, n=None):
        """获取所有（或Top-N）候选的方程与参数（若算法支持）。"""
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        command = {
            'action': 'get_total_equations_with_params',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name,
        }
        if n is not None:
            command['n'] = int(n)
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"获取方程与参数失败: {result['message']}\n{result.get('traceback', '')}")
        return result.get('items', [])


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

    # ========= 实验清单（manifest）最小实现 =========
    def _sanitize_config(self) -> dict:
        """脱敏/规整后的配置，用于写入 manifest 的 config 字段。"""
        hidden = {"api_key", "apikey", "token", "password", "secret"}
        cfg = {k: v for k, v in (self.params or {}).items() if str(k).lower() not in hidden}
        cfg.setdefault("tool_name", self.tool_name)
        cfg.setdefault("problem_name", self.problem_name)
        cfg.setdefault("seed", self.seed)
        return cfg

    def _manifest_path(self) -> str:
        return os.path.join(self.experiment_dir, "manifest.json")

    def _write_initial_manifest(self):
        now_local = datetime.now().astimezone()
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        manifest = {
            "experiment_id": os.path.basename(self.experiment_dir),
            "problem_name": self.problem_name,
            "algorithm": self.tool_name,
            "seed": self.seed,
            "created_at_local": now_local.isoformat(),
            "created_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "status": "created",
            # 迭代次数占位，暂不写入具体数值
            "iterations": None,
            "config": self._sanitize_config(),
        }
        path = self._manifest_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _update_manifest(self, **fields):
        path = self._manifest_path()
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data.update(fields or {})
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update_iterations(self, iterations):
        """更新迭代次数占位接口（当前不主动调用）。"""
        try:
            it = int(iterations)
        except Exception:
            return
        try:
            self._update_manifest(iterations=it)
        except Exception:
            pass
