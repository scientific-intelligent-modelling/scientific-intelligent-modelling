import numpy as np
import pytest
from pysr import PySRRegressor

def test_pysr_basic_regression():
    # 生成测试数据
    np.random.seed(42)  # 确保可重复性
    X = 2 * np.random.randn(100, 5)
    y = 2.5 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

    # 创建并配置模型
    model = PySRRegressor(
        model_selection="best",
        niterations=20,  # 为了测试速度，减少迭代次数
        binary_operators=["+", "*"],
        unary_operators=["cos", "exp", "sin"],
        population_size=20,
        timeout_in_seconds=30
    )

    # 训练模型
    model.fit(X, y)

    # 基本功能测试
    assert model is not None
    assert hasattr(model, 'predict')
    
    # 预测测试
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    
    # 检查模型是否有输出方程
    equations = model.equations_
    assert len(equations) > 0
    
    # 检查模型评分
    scores = model.score
    print(scores)

def test_pysr_input_validation():
    model = PySRRegressor(niterations=5)
    
    # 测试输入验证
    with pytest.raises(ValueError):
        # 测试空输入
        model.fit(np.array([]), np.array([]))
    
    with pytest.raises(ValueError):
        # 测试维度不匹配的输入
        X_invalid = np.random.randn(10, 2)
        y_invalid = np.random.randn(5)
        model.fit(X_invalid, y_invalid)

def test_pysr_custom_operators():
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = 1 / X[:, 0] + np.sin(X[:, 1])
    
    model = PySRRegressor(
        niterations=10,
        binary_operators=["+"],
        unary_operators=[
            "sin",
            "inv(x) = 1/x"
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x}
    )
    
    model.fit(X, y)
    assert model is not None
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape 

test_pysr_basic_regression()