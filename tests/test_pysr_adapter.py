import numpy as np
import pytest
from scientific_intelligent_modelling.adapters.pysr_adapter import PySRRegressor

def test_pysr_adapter_basic():
    """测试PySR适配器的基本功能"""
    # 生成测试数据
    np.random.seed(42)
    X = 2 * np.random.randn(100, 3)  # 100个样本，3个特征
    y = 2.5 * np.cos(X[:, 0]) + X[:, 1]**2  # 一个简单的目标函数

    # 创建并配置模型
    model = PySRRegressor(
        model_selection="best",
        niterations=10,  # 为了测试速度，使用较少的迭代次数
        binary_operators=["+", "*"],
        unary_operators=["cos"],
        population_size=20,
        timeout_in_seconds=30,
        verbosity=0  # 减少输出
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
    assert hasattr(model, 'equations_')
    assert model.equations_ is not None

def test_pysr_adapter_input_validation():
    """测试PySR适配器的输入验证"""
    model = PySRRegressor(niterations=5)
    
    # 测试空输入
    with pytest.raises(Exception):
        model.fit(np.array([]), np.array([]))
    
    # 测试维度不匹配的输入
    with pytest.raises(Exception):
        X_invalid = np.random.randn(10, 2)
        y_invalid = np.random.randn(5)
        model.fit(X_invalid, y_invalid)

def test_pysr_adapter_custom_operators():
    """测试PySR适配器的自定义操作符功能"""
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
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        verbosity=0
    )
    
    model.fit(X, y)
    assert model is not None
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape 