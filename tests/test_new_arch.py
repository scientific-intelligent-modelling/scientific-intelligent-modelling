# tests/compare_methods.py
#########
import scientific_intelligent_modelling
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
import os

# scientific_intelligent_modelling.environment_manager.


# 生成示例数据
X = np.random.rand(100, 2)
y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.1*np.random.randn(100)

# # 这里改成在相同目录下的key.txt文件
# with open('./key.txt', 'r') as f:
#     api_key = f.read().strip()

# print(api_key)
# # 使用不同工具进行对比
api_key = os.getenv("DEEPSEEK_API_KEY")
models = {
    # 'gplearn': SymbolicRegressor('gplearn', population_size=1000, generations=20),
    # 'pysr': SymbolicRegressor('pysr', niterations=5),
    # 'operon': SymbolicRegressor('operon'),
    'llmsr': SymbolicRegressor(
        'llmsr',
        use_api=True,
        api_model="deepseek/deepseek-chat",
        # api_model="deepseek/deepseek-reasoner",
        api_key=api_key,
        spec_path="./specs/specification_crk0_numpy.txt",  # 使用实际存在的规范文件
        log_path="./logs/example_deepseek1",
        problem_name="crk0",  # 使用实际存在的问题名称
        samples_per_prompt=5,
        max_samples=5
    ),
    # 'dso': SymbolicRegressor('dso'),
    # "e2esr": SymbolicRegressor(
    #     'e2esr',
    #     # 指定预训练模型路径，如果模型文件不存在会自动创建新模型
    #     model_path="./scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/model1",
    #     # 添加其他E2ESR参数
    #     max_input_points=200,
    #     n_trees_to_refine=100,
    #     rescale=True
    # )
}


for name, model in models.items():
    model.fit(X, y)
    result = model.get_optimal_equation()

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    print(f"{name} MSE: {mse}")




print("\n最佳模型:", min(results, key=results.get))



results = {}
for name, model in models.items():
    print(f"训练 {name} 模型...")
    model.fit(X, y)
    result = model.get_optimal_equation()
    print(f"{name} 方程: {result}")
    print(f"{name} 方程: {model.get_optimal_equation()}")

    # 获取所有方程
    equations = model.get_total_equations()
    print(f"{name} 所有方程: {equations}")
    print(f"{name} 方程: {model.get_total_equations()}")
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    results[name] = mse
    print(f"{name} MSE: {mse}")

print("\n最佳模型:", min(results, key=results.get))