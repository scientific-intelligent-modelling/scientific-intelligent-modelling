"""
测试方程清理功能
"""
from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor

# 模拟 LLM 生成的带有测试代码的方程体
dirty_body = """    total = params[0]  # bias term
    total = total + params[1] * col0
    total = total + params[2] * col1
    total = total + params[3] * col2
    total = total + params[4] * col3
    total = total + params[5] * col4
    return total

# Testing the equation function with random data
col0 = np.random.rand(100)
col1 = np.random.rand(100)
col2 = np.random.rand(100)
col3 = np.random.rand(100)
col4 = np.random.rand(100)

predictions_v0 = equation_v0(col0, col1, col2, col3, col4, params)
predictions_v1 = equation_v1(col0, col1, col2, col3, col4, params)

print(predictions_v0)
print(predictions_v1)"""

print("原始方程体:")
print("=" * 60)
print(dirty_body)
print("=" * 60)

cleaned = DRSRRegressor._clean_equation_body(dirty_body)

print("\n清理后的方程体:")
print("=" * 60)
print(cleaned)
print("=" * 60)

# 测试能否编译
import numpy as np
try:
    regressor = DRSRRegressor()
    regressor._n_features = 5
    func = regressor._compile_equation(dirty_body, 5)
    
    # 测试调用
    test_data = np.random.randn(10, 5)
    params = np.ones(10)
    result = func(*test_data.T, params)
    
    print(f"\n✅ 编译和执行成功！")
    print(f"输出形状: {result.shape}")
    print(f"输出样例: {result[:3]}")
except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()
