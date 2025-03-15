"""Testing with example.csv dataset for symbolic regression."""

import numpy as np
import pandas as pd
import os
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split

# import scientific_intelligent_modelling as sim
from scientific_intelligent_modelling import GplearnRegressor

if __name__ == '__main__':
    # 读取生成的数据集
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, 'datasets', 'example.csv')
    
    print(f"加载数据集: {dataset_path}")
    data = pd.read_csv(dataset_path)
    
    # 分离特征和目标变量
    X = data[['x1', 'x2', 'x3']].values
    y = data['y'].values
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"目标函数: y = sin(x1) + x2^2 - 0.5*x3 + 0.1*x1*x2*x3")
    
    # 配置并运行符号回归
    est_gp = GplearnRegressor(
        population_size=5000, 
        generations=20,
        stopping_criteria=0.01, 
        p_crossover=0.7,
        p_subtree_mutation=0.1, 
        p_hoist_mutation=0.05,
        p_point_mutation=0.1, 
        max_samples=0.9,
        parsimony_coefficient=0.01, 
        verbose=1,
        random_state=0
    )
    
    print("开始符号回归训练...")
    est_gp.fit(X_train, y_train)
    
    # 打印结果模型
    print("\n发现的模型:")
    print(est_gp)
    
    # 计算测试集得分
    test_score = est_gp.score(X_test, y_test)
    print(f"\n测试集 R^2 分数: {test_score:.4f}")