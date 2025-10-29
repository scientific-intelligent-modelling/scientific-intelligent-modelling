# algorithms/llmsr_wrapper/wrapper.py
import os
import json
import re
import glob
import inspect
import time
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
        self._equation_function_str = None  # 原始方程函数定义字符串
        self._sympy_expr_symbolic = None    # 符号表达式（包含参数符号）
        self._sympy_expr_fitted = None      # 替换了最优参数后的表达式
        self._last_log_dir = None           # 本次运行使用的日志目录
        self._specification_str = None      # 本次使用的规范文本（用于解析参数设置）

    def serialize(self):
        """仅序列化必要状态，避免pickle函数对象。"""
        state = {
            'params': self.params,
            'best_equation': self.best_equation,
            'all_equations': self.all_equations,
            'equation_function_str': self._equation_function_str,
            '_equation_argcount': self._equation_argcount,
            '_best_params': self._best_params.tolist() if isinstance(self._best_params, np.ndarray) else self._best_params,
            # 仅作为展示备份（可选）
            'expr_symbolic': str(self._sympy_expr_symbolic) if self._sympy_expr_symbolic is not None else None,
            'expr_fitted': str(self._sympy_expr_fitted) if self._sympy_expr_fitted is not None else None,
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
        inst._equation_function_str = obj.get('equation_function_str')
        best_params = obj.get('_best_params')
        inst._best_params = np.array(best_params) if best_params is not None else None
        # 根据函数定义字符串编译方程（而不是对可读表达式编译）
        if inst._equation_function_str:
            try:
                inst._equation_func = inst._compile_equation(inst._equation_function_str)
            except Exception:
                inst._equation_func = None
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
        # 自动推导 problem_name（如未提供）
        if not self.params.get('problem_name'):
            self.params['problem_name'] = self._derive_problem_name(self.params.get('background'))

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
            
            # 规范来源：有 spec_path 用文件，否则自动生成
            spec_path = self.params.get('spec_path')
            specification = None
            if spec_path:
                if not os.path.isabs(spec_path):
                    spec_path = os.path.abspath(spec_path)
                with open(spec_path, encoding="utf-8") as f:
                    specification = f.read()
            else:
                specification = self._build_dynamic_spec_from_background(X, y, self.params.get('background'))
                # 写到临时目录便于复现
                auto_spec_dir = os.path.join(temp_dir, 'auto_spec')
                os.makedirs(auto_spec_dir, exist_ok=True)
                spec_path = os.path.join(auto_spec_dir, 'generated_spec.txt')
                with open(spec_path, 'w', encoding='utf-8') as f:
                    f.write(specification)
            self._specification_str = specification

            # 不再设置 litellm 的环境变量，这里仅记录 API Key/Base 以便下游使用
            
            # 解析API参数（传递到 cfg 上，由 APILLM 使用）
            api_params = {}
            if 'api_params' in self.params and self.params['api_params']:
                if isinstance(self.params['api_params'], str):
                    api_params = json.loads(self.params['api_params'])
                else:
                    api_params = self.params['api_params']
            
            # 此时 specification 已就绪
                self._specification_str = specification
            
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
            
            # 构造本次唯一日志目录，避免与历史混淆
            base_log_dir = self.params.get('log_path', './logs/llmsr_output')
            run_log_dir = os.path.join(base_log_dir, f"run_{int(time.time())}")
            os.makedirs(run_log_dir, exist_ok=True)
            self._last_log_dir = run_log_dir

            # 记录开始时间，用于筛选本次生成的样本
            start_ts = time.time()

            # 运行主流程
            _ = pipeline.main(
                specification=specification,
                inputs=dataset,
                config=config_obj,
                max_sample_nums=self.params.get('max_samples', 10000),
                class_config=class_config,
                log_dir=run_log_dir,
                samples_per_prompt=self.params.get('samples_per_prompt', 5),
            )
            
            # 保存结果（pipeline.main 无返回值）
            self.model = True

            # 从日志中提取最优方程字符串与所有候选
            samples_dir = os.path.join(run_log_dir, 'samples')
            best_func_str, all_func_strs, best_params_captured = self._load_equations_from_logs(samples_dir, since_ts=start_ts, with_params=True)
            if best_func_str is None:
                raise RuntimeError(f"未在日志目录中找到任何候选方程: {samples_dir}")

            self._equation_function_str = best_func_str
            self.all_equations = all_func_strs

            # 构建符号表达式（包含参数符号），用于更直观展示
            try:
                expr_symbolic = self._build_symbolic_expression(best_func_str)
                self._sympy_expr_symbolic = expr_symbolic
            except Exception:
                self._sympy_expr_symbolic = None

            # 若评估阶段已捕获参数，直接使用；否则再拟合
            if best_params_captured is not None:
                # 编译方程，设置参数
                equation = self._compile_equation(best_func_str)
                sig = inspect.signature(equation)
                self._equation_argcount = len(sig.parameters) - 1
                self._equation_func = equation
                self._best_params = np.array(best_params_captured, dtype=float)
            else:
                # 基于训练数据对最优方程进行参数再拟合
                self._prepare_predictor(best_func_str, X, y, spec_text=self._specification_str, run_log_dir=run_log_dir)

            # 生成带最优参数的可读表达式
            try:
                if self._sympy_expr_symbolic is not None and self._best_params is not None:
                    import sympy as sp
                    param_syms = self._get_param_symbols(self._estimate_param_dim_from_function(best_func_str))
                    subs_map = {param_syms[i]: float(self._best_params[i]) for i in range(min(len(param_syms), len(self._best_params)))}
                    self._sympy_expr_fitted = sp.simplify(self._sympy_expr_symbolic.subs(subs_map))
            except Exception:
                self._sympy_expr_fitted = None

            # 默认把 best_equation 暴露为更直观的表达式字符串
            self.best_equation = self._format_expr_for_output()
                
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
        
        # 懒编译：若反序列化后未编译方程，但有函数定义与参数，执行一次编译
        if self._equation_func is None and self._equation_function_str:
            try:
                self._equation_func = self._compile_equation(self._equation_function_str)
            except Exception as e:
                raise ValueError(f"方程编译失败: {e}")

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
        
        # 优先返回带参数的可读数学表达式
        if self.best_equation:
            return str(self.best_equation)
        
        # 如果没有明确的最优方程，但有结果对象，尝试从中提取
        if self._equation_function_str:
            return self._equation_function_str
        
        # 如果没有可用的方程
        return "未找到可用的方程"
    
    def get_total_equations(self):
        """获得所有方程"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回所有方程列表（函数定义字符串列表）
        if self.all_equations:
            return self.all_equations
        
        # 如果没有明确的方程列表，但有结果对象，尝试从中提取
        if self.model and hasattr(self.model, 'formulas'):
            return self.model.formulas
        
        # 如果没有可用的方程列表
        return []

    # ===================== 内部工具方法 =====================
    def _load_equations_from_logs(self, samples_dir: str, since_ts: float | None = None, with_params: bool = False):
        """读取日志目录，返回(最佳函数字符串, 全部函数字符串列表)。
        可选按时间戳筛选仅本次运行生成的样本。
        """
        if not os.path.isdir(samples_dir):
            return None, []
        files = sorted(glob.glob(os.path.join(samples_dir, 'samples_*.json')))
        if since_ts is not None:
            files = [fp for fp in files if os.path.getmtime(fp) >= since_ts - 1.0]
        best = None
        best_score = -float('inf')
        all_funcs = []
        best_params = None
        for fp in files:
            try:
                with open(fp, 'r') as f:
                    obj = json.load(f)
                func = obj.get('function')
                score = obj.get('score')
                params = obj.get('fitted_params') if with_params else None
                if func:
                    all_funcs.append(func)
                if isinstance(score, (int, float)) and func:
                    if score > best_score:
                        best_score = score
                        best = func
                        best_params = params
            except Exception:
                continue
        return best, all_funcs, best_params

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

    def _prepare_predictor(self, function_str: str, X: np.ndarray, y: np.ndarray, spec_text: str | None = None, run_log_dir: str | None = None):
        """编译函数字符串并在训练集上再次拟合最优参数，同时保存拟合产物。"""
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

        # 估计参数维度与初始值：优先从规范文本解析 MAX_NPARAMS / PRAMS_INIT
        init_p = None
        param_dim = None
        if spec_text:
            try:
                spec_dim, spec_init = self._parse_spec_param_init(spec_text)
                if spec_dim:
                    param_dim = spec_dim
                if spec_init is not None:
                    init_p = np.array(spec_init, dtype=float)
            except Exception:
                pass
        if param_dim is None:
            param_dim = self._estimate_param_dim_from_function(function_str)

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

            p0 = init_p if init_p is not None and len(init_p) == param_dim else np.ones(param_dim, dtype=float)
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

    def _parse_spec_param_init(self, spec_text: str):
        """从规范文本中解析 MAX_NPARAMS 和 PRAMS_INIT/PARAMS_INIT（若存在）。"""
        # MAX_NPARAMS = 10
        dim = None
        m = re.search(r"MAX_NPARAMS\s*=\s*(\d+)", spec_text)
        if m:
            dim = int(m.group(1))
        # PRAMS_INIT / PARAMS_INIT = [1.0, 1.0, ...]
        init = None
        m2 = re.search(r"PRAMS?\s*_INIT\s*=\s*\[([^\]]*)\]", spec_text)
        if m2:
            try:
                arr = '[' + m2.group(1) + ']'
                init = json.loads(arr.replace("'", '"'))
            except Exception:
                # 回退：用逗号切分再转 float
                try:
                    init = [float(x) for x in m2.group(1).split(',') if x.strip()]
                except Exception:
                    init = None
        return dim, init

    def get_fitted_params(self):
        """返回当前最优参数的字典映射（p0->val, ...）。"""
        if self._best_params is None:
            return None
        param_syms = self._get_param_symbols(len(self._best_params))
        return {str(param_syms[i]): float(self._best_params[i]) for i in range(len(self._best_params))}

    def _estimate_param_dim_from_function(self, function_str: str) -> int:
        max_idx = -1
        for m in re.finditer(r"params\s*\[\s*(\d+)\s*\]", function_str):
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
        return max(1, max_idx + 1) if max_idx >= 0 else 10

    def _get_param_symbols(self, dim: int):
        import sympy as sp
        return sp.symbols(' '.join([f'p{i}' for i in range(dim)]))

    def _build_symbolic_expression(self, function_str: str):
        """将 equation 函数体转换为 SymPy 表达式（包含参数符号）。"""
        import ast
        import sympy as sp

        # 解析 AST，定位 equation 函数
        tree = ast.parse(function_str)
        func_node = None
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == 'equation':
                func_node = n
                break
        if func_node is None:
            raise RuntimeError('未找到 equation 定义')

        # 获取输入变量名与 params 名
        arg_names = [a.arg for a in func_node.args.args]
        if len(arg_names) < 1:
            raise RuntimeError('equation 参数异常')
        input_arg_names = arg_names[:-1]
        param_dim = self._estimate_param_dim_from_function(function_str)
        param_syms = self._get_param_symbols(param_dim)
        var_syms = {name: sp.symbols(name) for name in input_arg_names}

        # 将 numpy 常见函数映射到 sympy（可扩展）
        def _log10(x):
            return sp.log(x, 10)
        def _log2(x):
            return sp.log(x, 2)
        def _clip(x, a, b):
            return sp.Min(sp.Max(x, a), b)
        np_to_sp = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'exp': sp.exp,
            'log': sp.log,
            'log10': _log10,
            'log2': _log2,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'power': sp.Pow,
            'maximum': sp.Max,
            'minimum': sp.Min,
            'clip': _clip,
        }

        # 简单的语句求值环境（变量名 -> sympy 表达式）
        local_defs: dict[str, sp.Expr] = {}

        def to_sp(node):
            # 处理常量属性，如 np.pi / math.pi / np.e / math.e
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                base = node.value.id
                attr = node.attr
                if base in ('np', 'numpy', 'math'):
                    if attr.lower() == 'pi':
                        return sp.pi
                    if attr.lower() in ('e', 'euler'): 
                        return sp.E
                # 其他属性暂不支持，返回符号占位，避免丢失
                return sp.symbols(f"{base}_{attr}")
            if isinstance(node, ast.BinOp):
                left = to_sp(node.left)
                right = to_sp(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                return None
            if isinstance(node, ast.UnaryOp):
                operand = to_sp(node.operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.Not):
                    return sp.Not(operand)
            if isinstance(node, ast.Name):
                if node.id in local_defs:
                    return local_defs[node.id]
                if node.id in var_syms:
                    return var_syms[node.id]
                # 忽略 np 名称本身
                if node.id == 'np':
                    return None
                # 其他未知名称，按符号处理
                return sp.symbols(node.id)
            if isinstance(node, ast.Subscript):
                # 仅处理 params[i]
                if isinstance(node.value, ast.Name) and node.value.id == 'params':
                    idx = None
                    if isinstance(node.slice, ast.Constant):
                        idx = int(node.slice.value)
                    elif hasattr(ast, 'Index') and isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Constant):
                        idx = int(node.slice.value.value)
                    if idx is not None and 0 <= idx < len(param_syms):
                        return param_syms[idx]
                return None
            if isinstance(node, ast.Call):
                # np.xxx(...) 或直接 xxx(...)
                func = node.func
                func_name = None
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'np':
                    func_name = func.attr
                elif isinstance(func, ast.Name):
                    func_name = func.id
                if func_name in np_to_sp:
                    args = [to_sp(a) for a in node.args]
                    return np_to_sp[func_name](*args)
                # 未映射函数，尝试作为符号函数
                args = [to_sp(a) for a in node.args]
                return sp.Function(func_name)(*args)
            if isinstance(node, ast.Constant):
                return sp.Float(node.value) if isinstance(node.value, float) else sp.Integer(node.value)
            if isinstance(node, ast.Compare):
                # 控制流不处理
                return None
            return None

        # 顺序处理赋值，最后处理 return
        ret_expr = None
        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                val = to_sp(stmt.value)
                if val is not None:
                    local_defs[stmt.targets[0].id] = val
            elif isinstance(stmt, ast.Return):
                ret_expr = to_sp(stmt.value)

        if ret_expr is None:
            raise RuntimeError('无法从函数中提取返回表达式')
        # 强制内联：多轮替换所有已定义的中间变量
        prev = None
        cur = ret_expr
        for _ in range(10):  # 最多迭代10次避免死循环
            prev = cur
            for k, v in list(local_defs.items()):
                cur = cur.xreplace({sp.Symbol(k): v})
            if sp.simplify(cur - prev) == 0:
                break
        return sp.simplify(cur)

    def _format_expr_for_output(self):
        """返回最直观的表达式字符串：优先替换最优参数，否则返回符号表达式，否则返回原函数。"""
        try:
            import sympy as sp
            if self._sympy_expr_fitted is not None:
                return sp.sstr(sp.simplify(sp.expand(self._sympy_expr_fitted)))
            if self._sympy_expr_symbolic is not None:
                return sp.sstr(sp.simplify(sp.expand(self._sympy_expr_symbolic)))
            if self._equation_function_str is not None:
                return self._equation_function_str
        except Exception:
            pass
        return None

    # ============== 背景驱动的动态规范生成 ==============
    def _derive_problem_name(self, background):
        base = 'auto_problem'
        if isinstance(background, dict):
            title = background.get('title') or background.get('name')
            if title:
                base = str(title)
        elif isinstance(background, str) and background.strip():
            base = background.strip().split('\n')[0][:32]
        slug = re.sub(r'[^\w\-]+', '_', base.strip())
        slug = re.sub(r'_+', '_', slug).strip('_').lower()
        return slug or 'auto_problem'

    def _build_dynamic_spec_from_background(self, X, y, background):
        n_features = int(X.shape[1]) if hasattr(X, 'shape') else int(len(X[0]))
        var_names, var_descs = self._extract_variable_info(background, n_features)
        max_params = None
        param_init = None
        notes_text = ''
        if isinstance(background, dict):
            max_params = background.get('max_params')
            param_init = background.get('param_init')
            notes_text = background.get('notes') or background.get('domain') or ''
        elif isinstance(background, str):
            notes_text = background
        max_params = int(max_params) if isinstance(max_params, int) and max_params > 0 else 10
        if not (isinstance(param_init, list) and len(param_init) == max_params):
            param_init = [1.0] * max_params

        # Header docstring
        header_lines = [
            '"""',
            'Auto-generated spec for LLMSR.',
            f"Problem: {self.params.get('problem_name', 'auto')}",
            'Background:',
        ]
        if notes_text:
            header_lines += [str(notes_text)]
        header_lines += ['Variables:']
        for name, desc in zip(var_names, var_descs):
            header_lines += [f'  - {name}: {desc}']
        header_lines += ['"""']
        header = '\n'.join(header_lines)

        # Inputs extraction code
        extract_lines = []
        for idx, nm in enumerate(var_names):
            extract_lines.append(f"    {nm} = inputs[:, {idx}]")
        extract_block = '\n'.join(extract_lines)

        # Baseline equation terms
        term_count = min(n_features, max_params - 1) if max_params > 1 else 0
        terms = [f"params[{i}] * {var_names[i]}" for i in range(term_count)]
        bias_idx = term_count if max_params > term_count else max(0, max_params - 1)
        if max_params > 0:
            terms.append(f"params[{bias_idx}]")
        body_return = ' + '.join(terms) if terms else '0.0'

        spec = (
            f"{header}\n\n"
            "import numpy as np\n\n"
            f"MAX_NPARAMS = {max_params}\n"
            f"PRAMS_INIT = {param_init}\n\n"
            "@evaluate.run\n"
            "def evaluate(data: dict) -> float:\n"
            "    \"\"\" Evaluate the equation on data observations. \"\"\"\n"
            "    inputs, outputs = data['inputs'], data['outputs']\n"
            f"{extract_block}\n"
            "    from scipy.optimize import minimize\n"
            "    def loss(params):\n"
            f"        y_pred = equation({', '.join(var_names)}, params)\n"
            "        return np.mean((y_pred - outputs) ** 2)\n"
            "    result = minimize(lambda p: loss(p), PRAMS_INIT, method='BFGS')\n"
            "    val = float(result.fun)\n"
            "    if np.isnan(val) or np.isinf(val):\n"
            "        return None\n"
            "    return -val\n\n"
            "@equation.evolve\n"
            f"def equation({', '.join(var_names)}, params: np.ndarray) -> np.ndarray:\n"
            "    \"\"\" Mathematical function to be discovered.\n\n"
            "    Hints:\n"
            "      - Use physically meaningful combinations if applicable.\n"
            "      - Keep it simple and smooth to avoid overfitting.\n"
            "    \"\"\"\n"
            f"    return {body_return}\n"
        )
        return spec

    def _extract_variable_info(self, background, n_features):
        names = []
        descs = []
        if isinstance(background, dict) and isinstance(background.get('variables'), list):
            for item in background['variables']:
                if isinstance(item, dict):
                    nm = item.get('name'); ds = item.get('desc') or ''
                elif isinstance(item, str):
                    if ':' in item or '：' in item:
                        nm, ds = re.split(r'[:：]', item, maxsplit=1)
                    else:
                        nm, ds = item, ''
                else:
                    continue
                nm = re.sub(r'[^A-Za-z0-9_]', '_', str(nm).strip())
                if nm:
                    names.append(nm)
                    descs.append(str(ds).strip())
            if len(names) >= n_features:
                return names[:n_features], descs[:n_features]
        if isinstance(background, str):
            for line in background.splitlines():
                m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:：]\s*(.+)$', line.strip())
                if m:
                    names.append(m.group(1)); descs.append(m.group(2))
            if len(names) >= n_features:
                return names[:n_features], descs[:n_features]
        names = [f'x{i}' for i in range(n_features)]
        descs = ['feature'] * n_features
        return names, descs

    # ============== 公共接口：返回表达式与参数细节 ==============
    def get_equation_details(self):
        """
        返回一个包含多种表示的字典：
        - input_vars: 输入变量名列表（按顺序）
        - param_symbols: 参数符号名列表 ['p0','p1',...]
        - params: {p0: val, ...}（若已拟合）
        - expression_symbolic: 未替换参数的 SymPy 字符串
        - expression_fitted: 已替换参数的 SymPy 字符串
        - latex_symbolic: LaTeX 形式（未替换参数）
        - latex_fitted: LaTeX 形式（已替换参数）
        """
        import sympy as sp

        # 确保符号表达式准备好
        if self._sympy_expr_symbolic is None and self._equation_function_str:
            try:
                self._sympy_expr_symbolic = self._build_symbolic_expression(self._equation_function_str)
            except Exception:
                self._sympy_expr_symbolic = None

        expr_symbolic = self._sympy_expr_symbolic
        expr_fitted = None
        params_map = None

        # 构造参数符号与输入变量名
        param_dim = None
        input_vars = []
        try:
            # 从函数签名恢复输入变量名
            import ast
            tree = ast.parse(self._equation_function_str)
            func_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'equation')
            arg_names = [a.arg for a in func_node.args.args]
            input_vars = arg_names[:-1] if len(arg_names) >= 1 else []
        except Exception:
            input_vars = []

        try:
            param_dim = self._estimate_param_dim_from_function(self._equation_function_str)
        except Exception:
            param_dim = len(self._best_params) if self._best_params is not None else 0

        param_syms = self._get_param_symbols(param_dim) if param_dim else []
        if self._best_params is not None and len(param_syms) >= len(self._best_params):
            params_map = {str(param_syms[i]): float(self._best_params[i]) for i in range(len(self._best_params))}

        if expr_symbolic is not None and params_map is not None:
            try:
                subs_map = {param_syms[i]: float(self._best_params[i]) for i in range(len(self._best_params))}
                expr_fitted = sp.simplify(expr_symbolic.subs(subs_map))
            except Exception:
                expr_fitted = None

        details = {
            'input_vars': input_vars,
            'param_symbols': [str(s) for s in param_syms],
            'params': params_map,
            'expression_symbolic': sp.sstr(expr_symbolic) if expr_symbolic is not None else None,
            'expression_fitted': sp.sstr(expr_fitted) if expr_fitted is not None else None,
            'latex_symbolic': sp.latex(expr_symbolic) if expr_symbolic is not None else None,
            'latex_fitted': sp.latex(expr_fitted) if expr_fitted is not None else None,
        }
        return details

if __name__ == "__main__":
    print("请使用 tests/test_new_arch.py 进行集成测试，并确保设置好对应的 API Key 环境变量。")
