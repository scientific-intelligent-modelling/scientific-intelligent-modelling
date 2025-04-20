#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试E2ESR包装器
"""

import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor

def test_e2esr():
    """测试E2ESR包装器的功能"""
    print("正在测试E2ESR包装器...")
    
    # 生成一些测试数据
    np.random.seed(42)  # 设置随机种子以获得可重复的结果
    
    # 生成两种不同的测试数据集
    
    # 1. 简单函数: f(x) = cos(2πx)
    X1 = np.random.uniform(-3, 3, (100, 1))
    y1 = np.cos(2 * np.pi * X1[:, 0]) + 0.1 * np.random.randn(100)
    
    # 2. 多变量函数: f(x1, x2) = cos(2πx1) + x2^2
    X2 = np.random.uniform(-3, 3, (100, 2))
    y2 = np.cos(2 * np.pi * X2[:, 0]) + X2[:, 1]**2 + 0.1 * np.random.randn(100)
    
    # 创建模型实例 - 修正后的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_dir = os.path.join(current_dir, "../scientific_intelligent_modelling/algorithms/e2esr_wrapper")
    model_path = os.path.join(wrapper_dir, "model.pt")
    
    # 使用各种参数配置测试模型
    configurations = [
        {
            "name": "默认配置",
            "params": {
                "model_path": model_path
            }
        },
        {
            "name": "自定义配置1",
            "params": {
                "model_path": model_path,
                "max_input_points": 200,
                "n_trees_to_refine": 100,
                "rescale": True
            }
        },
        {
            "name": "自定义配置2",
            "params": {
                "model_path": model_path,
                "max_input_points": 300,
                "n_trees_to_refine": 200,
                "rescale": True
            }
        }
    ]
    
    # 测试每种配置
    for config in configurations:
        print(f"\n测试配置: {config['name']}")
        
        # 创建模型
        model = SymbolicRegressor('e2esr', **config["params"])
        
        # 测试单变量
        print("\n测试单变量数据集:")
        model.fit(X1, y1)
        
        # 获取最优方程
        best_eq = model.get_optimal_equation()
        print(f"最优方程: {best_eq}")
        
        # 预测并计算指标
        y_pred = model.predict(X1)
        mse = mean_squared_error(y1, y_pred)
        r2 = r2_score(y1, y_pred)
        print(f"MSE: {mse:.6f}, R²: {r2:.6f}")
        
        # 获取所有方程
        all_eqs = model.get_total_equations()
        print(f"生成的方程数: {len(all_eqs)}")
        print(f"所有方程: {all_eqs}")
        
        # 测试多变量
        print("\n测试多变量数据集:")
        model.fit(X2, y2)
        
        # 获取最优方程
        best_eq = model.get_optimal_equation()
        print(f"最优方程: {best_eq}")
        
        # 预测并计算指标
        y_pred = model.predict(X2)
        mse = mean_squared_error(y2, y_pred)
        r2 = r2_score(y2, y_pred)
        print(f"MSE: {mse:.6f}, R²: {r2:.6f}")

if __name__ == "__main__":
    test_e2esr()