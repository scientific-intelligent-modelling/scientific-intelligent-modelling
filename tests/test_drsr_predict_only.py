"""
测试已训练模型的 predict 功能
直接加载之前训练的结果进行预测
"""
import numpy as np
from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor

# 生成测试数据：5个特征
np.random.seed(42)
n_samples = 10
X_test = np.random.randn(n_samples, 5)

print("=" * 60)
print("测试 DRSR 预测功能")
print("=" * 60)

# 创建一个模拟的已训练模型
regressor = DRSRRegressor()
regressor._n_features = 5

# 使用之前生成的最优方程（去掉注释部分）
equation_body = """    total = params[5]  # bias term
    total = total + params[0] * col0 + params[1] * col1 + params[2] * col2 + params[3] * col3 + params[4] * col4
    return total
"""

# 使用之前训练得到的参数
best_params = np.array([0.2792713557195924, 1.9469485309271193, -14.915400698399862, 
                        1.3740382961205886, 1.844812182609371, 1.4888771944638328, 
                        -0.10443367085381672, 0.1057861781426559, 0.18539344775878708, 
                        -0.8382933473345695])

regressor._equation_body = equation_body
regressor._best_params = best_params
regressor._equation_func = regressor._compile_equation(equation_body, 5)
regressor.model_ready = True

print(f"\n特征数量: {regressor._n_features}")
print(f"参数数量: {len(regressor._best_params)}")
print(f"\n方程:\n{regressor.get_optimal_equation()}")

print("\n" + "=" * 60)
print("测试预测...")
print("=" * 60)

try:
    predictions = regressor.predict(X_test)
    print(f"✅ 预测成功！")
    print(f"输入形状: {X_test.shape}")
    print(f"输出形状: {predictions.shape}")
    print(f"预测值前5个: {predictions[:5]}")
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
except Exception as e:
    print(f"❌ 预测失败: {e}")
    import traceback
    traceback.print_exc()
