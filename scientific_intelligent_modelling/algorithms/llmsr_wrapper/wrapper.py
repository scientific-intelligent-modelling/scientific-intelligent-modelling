# algorithms/llmsr_wrapper/wrapper.py
import os
import json
import re
import glob
import inspect
import numpy as np
import pandas as pd
import tempfile
import sys
from ..base_wrapper import BaseWrapper

# 引入自定义 LLM Client 工厂
from scientific_intelligent_modelling.srkit.llm import ClientFactory, parse_provider_model

class LLMSRRegressor(BaseWrapper):
    def __init__(self, **kwargs):
        """
        初始化LLMSR回归器
        
        参数:
        use_api (bool): 是否使用API模型 (默认: False)
        api_model (str): API模型名称 (默认: "gpt-3.5-turbo")
        # spec_path (str): 提示规范文件路径
        log_path (str): 日志目录 (默认: "./logs/llmsr_output")
        # problem_name (str): 问题名称 (默认: "example_problem")
        temperature (float): 采样温度 (默认: 0.7)
        api_key (str): API密钥 (默认: None)
        # api_params (dict): 额外的API参数 (默认: None)
        api_base (str): 自定义API基地址 (默认: None)
        # debug (bool): 是否启用调试输出 (默认: False)
        samples_per_prompt (int): 每个提示生成的样本数 (默认: 5)
        max_samples (int): 生成的最大样本数 (默认: 10000)
        """
        # 保存参数
        self.params = kwargs
        self.model = None
        self.best_equation = None
        self.all_equations = []
        # 拟合后缓存：方程函数与最优参数
        self._equation_func = None
        self._equation_argcount = None  # 不含 params 的输入位参数个数
        self._best_params = None

    def serialize(self):
        """仅序列化必要状态，避免pickle函数对象。"""
        state = {
            'params': self.params,
            'best_equation': self.best_equation,
            'all_equations': self.all_equations,
            '_equation_argcount': self._equation_argcount,
            '_best_params': self._best_params.tolist() if isinstance(self._best_params, np.ndarray) else self._best_params,
        }
        return json.dumps(state)

    @classmethod
    def deserialize(cls, payload: str):
        """从JSON状态重建实例，并恢复可调用方程（不重复拟合）。"""
        obj = json.loads(payload)
        inst = cls(**obj.get('params', {}))
        inst.model = True
        inst.best_equation = obj.get('best_equation')
        inst.all_equations = obj.get('all_equations', [])
        inst._equation_argcount = obj.get('_equation_argcount')
        best_params = obj.get('_best_params')
        inst._best_params = np.array(best_params) if best_params is not None else None
        # 仅编译函数，不进行再次拟合
        if inst.best_equation:
            inst._equation_func = inst._compile_equation(inst.best_equation)
        return inst
        
        # 设置默认值
        # self.params.setdefault('use_api', False)
        # self.params.setdefault('api_model', 'gpt-3.5-turbo')
        # self.params.setdefault('log_path', './logs/llmsr_output')
        # self.params.setdefault('problem_name', 'example_problem')
        # self.params.setdefault('temperature', 0.7)
        # self.params.setdefault('debug', False)
        # self.params.setdefault('samples_per_prompt', 5)
        # self.params.setdefault('max_samples', 10000)

    def fit(self, X, y):
        """
        训练模型
        
        参数:
        X: 输入特征
        y: 目标变量
        
        返回:
        self: 训练后的模型实例
        """
        # 确保存在临时目录来存储数据
        temp_dir = tempfile.mkdtemp(prefix='llmsr_')
        data_path = os.path.join(temp_dir, 'data', self.params['problem_name'])
        os.makedirs(data_path, exist_ok=True)
        
        # 将数据转换为CSV格式
        data = np.column_stack((X, y.reshape(-1, 1)))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        
        # 导入LLMSR模块
        original_cwd = os.getcwd()
        original_path = list(sys.path)
        
        try:
            # 将LLMSR模块所在目录添加到sys.path
            llmsr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llmsr')
            if llmsr_dir not in sys.path:
                sys.path.insert(0, llmsr_dir)
                
            # 切换到llmsr目录以确保相对导入正常工作
            os.chdir(llmsr_dir)
            
            # 现在导入模块
            from llmsr import pipeline, config, sampler, evaluator

            # 捕获外层参数用于 APILLM 闭包
            outer_params = dict(self.params)

            # 定义仅使用 API 的自定义 LLM（不依赖 litellm，不走本地推理）
            class APILLM(sampler.LLM):
                def __init__(self, samples_per_prompt: int) -> None:
                    super().__init__(samples_per_prompt)
                    self._instruction_prompt = (
                        "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
                        "Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n"
                    )
                    # 从外层参数构建 API 上下文（避免修改冻结的 Config）
                    self._api_ctx = {
                        'api_model': outer_params.get('api_model'),
                        'api_key': outer_params.get('api_key'),
                        'api_base': outer_params.get('api_base'),
                        'temperature': outer_params.get('temperature'),
                        'api_params': outer_params.get('api_params') if isinstance(outer_params.get('api_params'), dict) else api_params,
                    }

                def draw_samples(self, prompt: str, cfg: config.Config):
                    # 强制 API 路径
                    full_prompt = '\n'.join([self._instruction_prompt, prompt])
                    # 解析 provider/model，确定默认环境变量
                    api_model = self._api_ctx.get('api_model') or cfg.api_model
                    provider, _model = parse_provider_model(api_model)

                    # 构造 ClientFactory 配置
                    api_key = getattr(cfg, 'api_key', None)
                    if not api_key:
                        if provider == 'deepseek':
                            api_key = os.getenv('DEEPSEEK_API_KEY', '')
                        elif provider in ('siliconflow', 'silicon-flow', 'sflow'):
                            api_key = os.getenv('SILICONFLOW_API_KEY', '')
                        elif provider == 'ollama':
                            api_key = ''

                    client = ClientFactory.from_config({
                        'model': api_model,
                        'api_key': api_key,
                        'base_url': self._api_ctx.get('api_base')
                    })

                    # 透传温度等生成参数
                    api_temperature = self._api_ctx.get('temperature')
                    extra_params = self._api_ctx.get('api_params')
                    if api_temperature is not None:
                        client.kwargs['temperature'] = api_temperature
                    if isinstance(extra_params, dict):
                        client.kwargs.update(extra_params)

                    messages = [{"role": "user", "content": full_prompt}]
                    verbose = outer_params.get('debug', True)
                    if verbose:
                        print(f"[LLMSR][Sampler] provider={provider}, model={api_model} | samples_per_prompt={self._samples_per_prompt}", flush=True)

                    all_samples = []
                    for i in range(self._samples_per_prompt):
                        while True:
                            try:
                                if verbose:
                                    print(f"[LLMSR][Sampler] requesting sample {i+1}/{self._samples_per_prompt} ...", flush=True)
                                resp = client.chat(messages)
                                content = resp.get('content', '') or ''
                                # 提取函数体（去除 def ... 行）
                                content = sampler._extract_body(content, cfg)
                                all_samples.append(content)
                                if verbose:
                                    tk = resp.get('tokens', {})
                                    print(f"[LLMSR][Sampler] got sample {i+1}/{self._samples_per_prompt} | tokens={tk}", flush=True)
                                break
                            except Exception as e:
                                if verbose:
                                    print(f"[LLMSR][Sampler] API 采样出错，重试中: {e}", flush=True)
                                import time as _t
                                _t.sleep(1)
                                continue

                    return all_samples
            
            # 构建命令行参数
            # 如果spec_path不是绝对路径，转换为绝对路径
            spec_path = self.params.get('spec_path')
            if spec_path and not os.path.isabs(spec_path):
                spec_path = os.path.abspath(spec_path)
                
            # 不再设置 litellm 的环境变量，这里仅记录 API Key/Base 以便下游使用
            
            # 解析API参数（传递到 cfg 上，由 APILLM 使用）
            api_params = {}
            if 'api_params' in self.params and self.params['api_params']:
                if isinstance(self.params['api_params'], str):
                    api_params = json.loads(self.params['api_params'])
                else:
                    api_params = self.params['api_params']
            
            # 加载提示规范
            with open(spec_path, encoding="utf-8") as f:
                specification = f.read()
            
            # 准备数据
            data_dict = {'inputs': X, 'outputs': y}
            dataset = {'data': data_dict}
            
            # 配置类
            class_config = config.ClassConfig(
                llm_class=APILLM,
                sandbox_class=evaluator.LocalSandbox
            )
            
            # 创建配置对象
            # 强制使用 API 方式，并透传 samples_per_prompt
            config_obj = config.Config(
                use_api=True,
                api_model=self.params.get('api_model', 'deepseek/deepseek-chat'),
                samples_per_prompt=self.params.get('samples_per_prompt', 4)
            )

            # 额外参数交由 APILLM._api_ctx 管理（Config 为冻结类，禁止动态赋值）
            
            # 运行主流程
            _ = pipeline.main(
                specification=specification,
                inputs=dataset,
                config=config_obj,
                max_sample_nums=self.params.get('max_samples', 10000),
                class_config=class_config,
                log_dir=self.params.get('log_path', './logs/llmsr_output'),
                samples_per_prompt=self.params.get('samples_per_prompt', 5),
            )
            
            # 保存结果（pipeline.main 无返回值）
            self.model = True

            # 从日志中提取最优方程字符串与所有候选
            log_dir = self.params.get('log_path', './logs/llmsr_output')
            samples_dir = os.path.join(log_dir, 'samples')
            best_func_str, all_func_strs = self._load_equations_from_logs(samples_dir)
            if best_func_str is None:
                raise RuntimeError(f"未在日志目录中找到任何候选方程: {samples_dir}")

            self.best_equation = best_func_str
            self.all_equations = all_func_strs

            # 基于训练数据对最优方程进行参数再拟合，并缓存可调用函数与最优参数
            self._prepare_predictor(best_func_str, X, y)
                
        except Exception as e:
            print(f"LLMSR训练过程中出现错误: {str(e)}")
            raise
        finally:
            # 恢复原始工作目录和Python路径
            os.chdir(original_cwd)
            sys.path = original_path
            
            # 清理临时文件
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
        return self
    
    def predict(self, X):
        """使用模型进行预测"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        if self._equation_func is None or self._best_params is None:
            raise ValueError("未找到可调用的方程或最优参数，请先调用fit完成训练与参数拟合")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self._equation_argcount is None:
            raise ValueError("内部方程元数据缺失")
        if X.shape[1] < self._equation_argcount:
            raise ValueError(f"特征维度不足：需要 {self._equation_argcount} 列，实际 {X.shape[1]}")

        cols = [X[:, i] for i in range(self._equation_argcount)]
        y_pred = self._equation_func(*cols, self._best_params)
        return np.asarray(y_pred)
    
    def get_optimal_equation(self):
        """获得最优方程"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回最优方程字符串
        if self.best_equation:
            return str(self.best_equation)
        
        # 如果没有明确的最优方程，但有结果对象，尝试从中提取
        if self.model and hasattr(self.model, 'best_formula'):
            return str(self.model.best_formula)
        
        # 如果没有可用的方程
        return "未找到可用的方程"
    
    def get_total_equations(self):
        """获得所有方程"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回所有方程列表
        if self.all_equations:
            return self.all_equations
        
        # 如果没有明确的方程列表，但有结果对象，尝试从中提取
        if self.model and hasattr(self.model, 'formulas'):
            return self.model.formulas
        
        # 如果没有可用的方程列表
        return []

    # ===================== 内部工具方法 =====================
    def _load_equations_from_logs(self, samples_dir: str):
        """读取日志目录，返回(最佳函数字符串, 全部函数字符串列表)。"""
        if not os.path.isdir(samples_dir):
            return None, []
        files = sorted(glob.glob(os.path.join(samples_dir, 'samples_*.json')))
        best = None
        best_score = -float('inf')
        all_funcs = []
        for fp in files:
            try:
                with open(fp, 'r') as f:
                    obj = json.load(f)
                func = obj.get('function')
                score = obj.get('score')
                if func:
                    all_funcs.append(func)
                if isinstance(score, (int, float)) and func:
                    if score > best_score:
                        best_score = score
                        best = func
            except Exception:
                continue
        return best, all_funcs

    def _compile_equation(self, function_str: str):
        """将函数字符串编译为可调用的 equation 函数。"""
        safe_globals = {"np": np}
        local_ns = {}
        try:
            exec(function_str, safe_globals, local_ns)
        except Exception as e:
            raise RuntimeError(f"编译方程函数失败: {e}\n{function_str}")
        equation = local_ns.get('equation') or safe_globals.get('equation')
        if equation is None:
            raise RuntimeError("未在函数字符串中找到 'equation' 定义")
        return equation

    def _prepare_predictor(self, function_str: str, X: np.ndarray, y: np.ndarray):
        """编译函数字符串并在训练集上再次拟合最优参数。"""
        equation = self._compile_equation(function_str)

        # 推断位参：除最后一个 params 外，其余均为输入特征
        sig = inspect.signature(equation)
        param_names = [p.name for p in sig.parameters.values()]
        if len(param_names) < 1:
            raise RuntimeError("equation 函数参数异常")
        # 认为最后一个是 params
        input_argcount = len(param_names) - 1
        self._equation_argcount = input_argcount

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] < input_argcount:
            raise ValueError(f"特征维度不足：需要 {input_argcount} 列，实际 {X.shape[1]}")

        cols = [X[:, i] for i in range(input_argcount)]

        # 估计参数维度：扫描 params[k]
        max_idx = -1
        for m in re.finditer(r"params\s*\[\s*(\d+)\s*\]", function_str):
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
        param_dim = max(1, max_idx + 1) if max_idx >= 0 else 10  # 默认10

        # 使用 SciPy 最小化 MSE 进行拟合
        try:
            from scipy.optimize import minimize

            def loss(p):
                try:
                    pred = equation(*cols, p)
                    pred = np.asarray(pred).reshape(-1)
                    if pred.shape[0] != y.shape[0]:
                        return 1e12
                    return float(np.mean((pred - y) ** 2))
                except Exception:
                    return 1e12

            p0 = np.ones(param_dim, dtype=float)
            # 与 spec 中保持一致的 BFGS 方法，提升可重复性
            res = minimize(loss, p0, method='BFGS')
            best_params = res.x if res.success else p0
        except Exception:
            # 回退：简单最小二乘（仅适用于线性组合情形），否则退回全1
            try:
                # 线性假设：pred = A @ p 近似
                # 无法通用解析，保持回退为全1
                best_params = np.ones(param_dim, dtype=float)
            except Exception:
                best_params = np.ones(param_dim, dtype=float)

        self._equation_func = equation
        self._best_params = best_params

if __name__ == "__main__":
    print("请使用 tests/test_new_arch.py 进行集成测试，并确保设置好对应的 API Key 环境变量。")
