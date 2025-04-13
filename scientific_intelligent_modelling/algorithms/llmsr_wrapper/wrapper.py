# algorithms/llmsr_wrapper/wrapper.py
import os
import json
import pickle
import base64
import numpy as np
import pandas as pd
import tempfile
import sys
from ..base_wrapper import BaseWrapper

class LLMSRRegressor(BaseWrapper):
    def __init__(self, **kwargs):
        """
        初始化LLMSR回归器
        
        参数:
        use_api (bool): 是否使用API模型 (默认: False)
        api_model (str): API模型名称 (默认: "gpt-3.5-turbo")
        spec_path (str): 提示规范文件路径
        log_path (str): 日志目录 (默认: "./logs/llmsr_output")
        problem_name (str): 问题名称 (默认: "example_problem")
        temperature (float): 采样温度 (默认: 0.7)
        api_key (str): API密钥 (默认: None)
        api_params (dict): 额外的API参数 (默认: None)
        api_base (str): 自定义API基地址 (默认: None)
        debug (bool): 是否启用调试输出 (默认: False)
        samples_per_prompt (int): 每个提示生成的样本数 (默认: 5)
        max_samples (int): 生成的最大样本数 (默认: 10000)
        """
        # 保存参数
        self.params = kwargs
        self.model = None
        self.best_equation = None
        self.all_equations = []
        
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
            
            # 构建命令行参数
            # 如果spec_path不是绝对路径，转换为绝对路径
            spec_path = self.params.get('spec_path')
            if spec_path and not os.path.isabs(spec_path):
                spec_path = os.path.abspath(spec_path)
                
            # 设置API密钥（如果提供）
            if 'api_key' in self.params and self.params['api_key']:
                api_key = self.params['api_key']
                api_model = self.params['api_model']
                
                if api_model.startswith('gpt-') or 'openai' in api_model:
                    os.environ["OPENAI_API_KEY"] = api_key
                elif 'anthropic' in api_model or 'claude' in api_model:
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                elif 'google' in api_model or 'gemini' in api_model:
                    os.environ["GOOGLE_API_KEY"] = api_key
                elif 'mistral' in api_model:
                    os.environ["MISTRAL_API_KEY"] = api_key
                elif 'deepseek' in api_model:
                    os.environ["DEEPSEEK_API_KEY"] = api_key
                else:
                    os.environ["OPENAI_API_KEY"] = api_key
            
            # 设置API基地址（如果提供）
            if 'api_base' in self.params and self.params['api_base']:
                api_base = self.params['api_base']
                api_model = self.params['api_model']
                
                if 'deepseek' in api_model:
                    os.environ["DEEPSEEK_API_BASE"] = api_base
                else:
                    os.environ["OPENAI_API_BASE"] = api_base
            
            # 解析API参数
            api_params = {}
            if 'api_params' in self.params and self.params['api_params']:
                if isinstance(self.params['api_params'], str):
                    api_params = json.loads(self.params['api_params'])
                else:
                    api_params = self.params['api_params']
            
            # 设置调试模式
            if self.params.get('debug', False):
                import litellm
                litellm.verbose = True
            
            # 加载提示规范
            with open(spec_path, encoding="utf-8") as f:
                specification = f.read()
            
            # 准备数据
            data_dict = {'inputs': X, 'outputs': y}
            dataset = {'data': data_dict}
            
            # 配置类
            class_config = config.ClassConfig(
                llm_class=sampler.UnifiedLLM,
                sandbox_class=evaluator.LocalSandbox
            )
            
            # 创建配置对象
            config_obj = config.Config(
                use_api=self.params.get('use_api', False),
                api_model=self.params.get('api_model', 'gpt-3.5-turbo')
            )
            
            # 运行主流程
            results = pipeline.main(
                specification=specification,
                inputs=dataset,
                config=config_obj,
                max_sample_nums=self.params.get('max_samples', 10000),
                class_config=class_config,
                log_dir=self.params.get('log_path', './logs/llmsr_output'),
                samples_per_prompt=self.params.get('samples_per_prompt', 5),
            )
            
            # 保存结果
            self.model = results
            
            # 提取最优方程和所有方程
            if results and hasattr(results, 'best_formula'):
                self.best_equation = results.best_formula
                
            if results and hasattr(results, 'formulas'):
                self.all_equations = results.formulas
                
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
        
        try:
            # 如果模型有predict方法，使用它
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            # 否则，尝试使用最佳方程进行预测
            elif self.best_equation:
                # 根据方程进行预测
                # 注: 这部分可能需要根据LLMSR的具体实现进行调整
                import sympy as sp
                from sympy.parsing.sympy_parser import parse_expr
                
                # 解析最佳方程字符串为sympy表达式
                expr = parse_expr(self.best_equation.replace("^", "**"))
                
                # 创建lambda函数用于评估
                var_names = [f"x{i}" for i in range(X.shape[1])]
                symbols = sp.symbols(var_names)
                f = sp.lambdify(symbols, expr, "numpy")
                
                # 将X传给lambda函数
                if X.shape[1] == 1:  # 如果只有一个特征
                    return f(X[:, 0])
                else:
                    return f(*[X[:, i] for i in range(X.shape[1])])
            else:
                raise ValueError("模型没有可用的预测功能或最佳方程")
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            raise
    
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

if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 生成示例数据
    X = np.random.rand(100, 2)
    y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.1*np.random.randn(100)
    
    # 创建并训练模型
    model = LLMSRRegressor(
        use_api=True,
        api_model="deepseek/deepseek-chat",
        api_key="sk-3cef2f9d83a44fcda7d932bc2384112c",
        spec_path="./specs/specification_oscillator1_numpy.txt",  # 使用实际存在的规范文件
        log_path="./logs/example_deepseek",
        problem_name="oscillator1",  # 使用实际存在的问题名称
        samples_per_prompt=5,
        max_samples=10000,
        debug=True
    )
    
    # 训练模型
    model.fit(X, y)
    
    # 获取最优方程
    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")
    
    # 获取所有方程
    total_equations = model.get_total_equations()
    print(f"所有方程: {total_equations}")
    
    # 使用模型进行预测
    predictions = model.predict(X[:5])
    print(f"预测结果: {predictions}")