# test.py
import scientific_intelligent_modelling as sim
import numpy as np
from sklearn.metrics import mean_absolute_error
import pytest

def test_symbolic_regressor():
    # 生成测试数据
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (100, 2))
    y = X[:, 0]**2 + X[:, 1]**2  # 目标函数: f(x) = x1^2 + x2^2
    
    # 初始化符号回归器
    est = sim.SymbolicRegressor(
        population_size=100,
        generations=10,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt'),
        metric='mean absolute error',
        random_state=0
    )
    
    # 训练模型
    est.fit(X, y)
    
    # 预测
    y_pred = est.predict(X)
    
    # 验证预测结果
    mae = mean_absolute_error(y, y_pred)
    assert mae < 1.0, f"预测误差过大: {mae}"

def test_symbolic_classifier():
    # 生成测试数据
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (100, 2))
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)  # 圆形分类问题
    
    # 初始化符号分类器
    est = sim.SymbolicClassifier(
        population_size=100,
        generations=10,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt'),
        random_state=0
    )
    
    # 训练模型
    est.fit(X, y)
    
    # 预测
    y_pred = est.predict(X)
    y_proba = est.predict_proba(X)
    
    # 验证预测结果
    accuracy = np.mean(y == y_pred)
    assert accuracy > 0.7, f"分类准确率过低: {accuracy}"
    assert y_proba.shape == (100, 2), f"概率预测形状错误: {y_proba.shape}"
    assert np.allclose(np.sum(y_proba, axis=1), 1), "概率和不为1"

def test_invalid_parameters():
    # 测试无效参数
    with pytest.raises(ValueError):
        sim.SymbolicRegressor(population_size=-1)
    
    with pytest.raises(ValueError):
        sim.SymbolicRegressor(generations=0)
    
    with pytest.raises(ValueError):
        sim.SymbolicRegressor(metric='invalid_metric')

