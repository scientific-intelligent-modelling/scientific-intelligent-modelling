# algorithms/tpsr_wrapper/wrapper.py
import pickle
import base64
import numpy as np
import os
import sys
import tempfile
import time
import torch
from ..base_wrapper import BaseWrapper

class TPSRRegressor(BaseWrapper):
    def __init__(self, **kwargs):
        # 延迟导入，避免环境问题
        self.params = kwargs
        self.model = None
        self.best_tree = None
        self.all_trees = []

        # 设置默认参数
        self.params.setdefault('max_input_points', 10000)
        self.params.setdefault('max_number_bags', -1)
        self.params.setdefault('stop_refinement_after', 1)
        self.params.setdefault('n_trees_to_refine', 1)
        self.params.setdefault('rescale', True)
        self.params.setdefault('beam_size', 10)
        self.params.setdefault('beam_type', 'sampling')
        self.params.setdefault('backbone_model', 'e2e')  # 'e2e' 或 'nesymres'
        self.params.setdefault('no_seq_cache', False)
        self.params.setdefault('no_prefix_cache', False)
        self.params.setdefault('width', 3)  # Top-k in TPSR's expansion step
        self.params.setdefault('num_beams', 1)  # Beam size in TPSR's evaluation
        self.params.setdefault('rollout', 3)  # Number of rollouts in TPSR
        self.params.setdefault('horizon', 200)  # Horizon of lookahead planning
        
    def fit(self, X, y):
        """
        训练TPSR模型
        
        参数:
        X: 输入特征，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,) 或 (n_samples, 1)
        
        返回:
        self: 训练后的模型实例
        """
        # 确保y是二维的
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # 导入必要的模块
        try:
            # 将TPSR模块所在目录添加到sys.path
            tpsr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpsr')
            if tpsr_dir not in sys.path:
                sys.path.insert(0, tpsr_dir)
                
            # 切换到tpsr目录以确保相对导入正常工作
            original_cwd = os.getcwd()
            os.chdir(tpsr_dir)
            
            # 导入必要的模块
            import symbolicregression
            from symbolicregression.envs import build_env
            from symbolicregression.model import build_modules
            from symbolicregression.trainer import Trainer
            from symbolicregression.e2e_model import Transformer, pred_for_sample, refine_for_sample
            from symbolicregression.model.model_wrapper import ModelWrapper
            from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor
            from parsers import get_parser
            from dyna_gym.agents.uct import UCT
            from rl_env import RLEnv
            from default_pi import E2EHeuristic
            
            # 创建临时目录来保存输出
            temp_dir = tempfile.mkdtemp(prefix='tpsr_')
            
            # 解析命令行参数
            parser = get_parser()
            args = parser.parse_args([])
            
            # 设置参数
            args.backbone_model = self.params.get('backbone_model', 'e2e')
            args.beam_size = self.params.get('beam_size', 10)
            args.beam_type = self.params.get('beam_type', 'sampling')
            args.no_seq_cache = self.params.get('no_seq_cache', False)
            args.no_prefix_cache = self.params.get('no_prefix_cache', False)
            args.width = self.params.get('width', 3)
            args.num_beams = self.params.get('num_beams', 1)
            args.rollout = self.params.get('rollout', 3)
            args.horizon = self.params.get('horizon', 200)
            args.debug = self.params.get('debug', False)
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 构建环境和模型
            equation_env = build_env(args)[0]
            
            # 准备数据
            data = np.column_stack((X, y))
            samples = {'x_to_fit': 0, 'y_to_fit': 0, 'x_to_pred': 0, 'y_to_pred': 0}
            samples['x_to_fit'] = [X]
            samples['y_to_fit'] = [y]
            samples['x_to_pred'] = [X]
            samples['y_to_pred'] = [y]
            
            # 创建和训练模型
            if args.backbone_model == 'e2e':
                model = Transformer(params=args, env=equation_env, samples=samples)
                model.to(args.device)
                
                # 创建RL环境
                rl_env = RLEnv(
                    params=args,
                    samples=samples,
                    equation_env=equation_env,
                    model=model
                )
                
                # 创建TPSR规划器
                dp = E2EHeuristic(
                    equation_env=equation_env,
                    rl_env=rl_env,
                    model=model,
                    k=args.width,
                    num_beams=args.num_beams,
                    horizon=args.horizon,
                    device=args.device,
                    use_seq_cache=not args.no_seq_cache,
                    use_prefix_cache=not args.no_prefix_cache,
                    length_penalty=args.beam_length_penalty if hasattr(args, 'beam_length_penalty') else 1.0,
                    train_value_mode=args.train_value if hasattr(args, 'train_value') else False,
                    debug=args.debug
                )
                
                # 创建UCT代理
                agent = UCT(
                    action_space=[],
                    gamma=1.0,
                    ucb_constant=1.0,
                    horizon=args.horizon,
                    rollouts=args.rollout,
                    dp=dp,
                    width=args.width,
                    reuse_tree=True,
                    alg=args.uct_alg if hasattr(args, 'uct_alg') else 'uct',
                    ucb_base=args.ucb_base if hasattr(args, 'ucb_base') else 4
                )
                
                # 运行TPSR搜索
                done = False
                s = rl_env.state
                for t in range(args.horizon):
                    if len(s) >= args.horizon or done:
                        break
                    
                    act = agent.act(rl_env, done)
                    s, r, done, _ = rl_env.step(act)
                    update_root(agent, act, s)
                    dp.update_cache(s)
                
                # 获取结果
                mcts_str = equation_env.detokenize(s)
                
                # 创建SymbolicTransformerRegressor对象
                embedder = model.embedder
                encoder = model.encoder
                decoder = model.decoder
                
                # 使用ModelWrapper和SymbolicTransformerRegressor来获取方程
                mw = ModelWrapper(
                    env=equation_env,
                    embedder=embedder,
                    encoder=encoder,
                    decoder=decoder,
                    beam_size=args.beam_size,
                    beam_type=args.beam_type,
                    max_generated_output_len=args.horizon
                )
                
                # 训练回归器
                regressor = SymbolicTransformerRegressor(
                    model=mw,
                    max_input_points=self.params.get('max_input_points', 10000),
                    max_number_bags=self.params.get('max_number_bags', -1),
                    stop_refinement_after=self.params.get('stop_refinement_after', 1),
                    n_trees_to_refine=self.params.get('n_trees_to_refine', 1),
                    rescale=self.params.get('rescale', True)
                )
                
                # 应用精调
                regressor.fit(X, y)
                
                # 储存模型
                self.model = regressor
                
                # 获取最优方程和所有方程
                best_tree = regressor.retrieve_tree(refinement_type="BFGS", dataset_idx=0)
                self.best_tree = best_tree
                
                all_refinement_types = regressor.retrieve_refinements_types()
                self.all_trees = []
                for refinement_type in all_refinement_types:
                    tree = regressor.retrieve_tree(refinement_type=refinement_type, dataset_idx=0)
                    if tree is not None:
                        self.all_trees.append(tree)
            
            else:
                # NesymRes后端
                # 这部分可以在未来需要时实现
                raise NotImplementedError("NesymRes后端暂未实现")
            
        except Exception as e:
            print(f"TPSR训练过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 恢复原始工作目录和Python路径
            os.chdir(original_cwd)
            if tpsr_dir in sys.path:
                sys.path.remove(tpsr_dir)
            
            # 清理临时目录
            try:
                import shutil
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
                return self.model.predict(X, refinement_type="BFGS")
            else:
                raise ValueError("模型没有可用的预测功能")
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            raise
    
    def get_optimal_equation(self):
        """获取模型学习到的最优符号方程"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        if self.best_tree is not None:
            # 生成方程字符串表示
            equation_str = str(self.best_tree)
            
            # 替换常见的操作符使其更易读
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            for op, replace_op in replace_ops.items():
                equation_str = equation_str.replace(op, replace_op)
                
            return equation_str
        else:
            return "未找到可用的方程"
    
    def get_total_equations(self):
        """获取模型学习到的所有符号方程"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        equations = []
        if self.all_trees:
            # 生成每个方程的字符串表示
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            for tree in self.all_trees:
                if tree is not None:
                    equation_str = str(tree)
                    for op, replace_op in replace_ops.items():
                        equation_str = equation_str.replace(op, replace_op)
                    equations.append(equation_str)
                    
        return equations

# 导出工具
__all__ = ["TPSRRegressor"]

if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 生成示例数据
    X = np.random.rand(100, 1) * 10 - 5  # 范围 [-5, 5]
    y = X[:, 0]**2 * np.sin(X[:, 0])  # 二次函数乘以正弦函数
    
    # 创建并训练模型
    model = TPSRRegressor(
        backbone_model='e2e',
        beam_size=10,
        beam_type='sampling',
        width=3,
        rollout=3
    )
    model.fit(X, y)
    
    # 获取最优方程
    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")
    
    # 获取所有方程
    equations = model.get_total_equations()
    print(f"所有方程: {equations}")
    
    # 使用模型进行预测
    predictions = model.predict(X[:5])
    print(f"预测结果: {predictions}")