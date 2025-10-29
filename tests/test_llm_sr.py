# 测试：使用背景描述直接运行 LLMSR（无需手写 specs）
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def main():
    # 1) 构造示例数据：t、A 两个自变量
    n = 200
    rng = np.random.RandomState(0)
    t = np.linspace(0, 3, n)
    A = np.sin(t) + 0.1 * rng.randn(n)
    # 真实关系（用于快速验证）：y = 0.8*A + 1.0*t - 0.15 + 噪声
    y = 0.8 * A + 1.0 * t - 0.15 + 0.05 * rng.randn(n)
    X = np.c_[t, A]

    # 2) 背景描述（无需提供 spec_path / problem_name）
    background = """
这是一个关于化学动力学的建模问题
t: 时间（秒）
A: 浓度（mol/L）
常见结构可能包含一阶/二阶项、指数衰减、饱和项等
"""

    # 3) 构造与训练模型
    # 注意：需设置 DEEPSEEK_API_KEY 或改用 siliconflow 的 api_model + SILICONFLOW_API_KEY
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[警告] 未检测到 DEEPSEEK_API_KEY，若报鉴权错误请先设置该环境变量。")

    model = SymbolicRegressor(
        'llmsr',
        use_api=True,
        api_model='deepseek/deepseek-chat',
        background=background,
        log_path='./logs/llmsr_bg_demo',
        samples_per_prompt=2,
        max_samples=6,
    )

    print("训练 llmsr 模型...")
    model.fit(X, y)

    # 4) 输出结果（最优函数定义或表达式 + MSE）
    best = model.get_optimal_equation()
    print("llmsr 方程(骨架或表达式):", best)

    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    print("llmsr MSE:", mse)
    

if __name__ == '__main__':
    main()

