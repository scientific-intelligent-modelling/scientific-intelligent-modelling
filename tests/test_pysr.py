import numpy as np
import pandas as pd
import os
from scientific_intelligent_modelling import PySRRegressor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
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

    # 创建并配置模型 - 增加一些参数使其更适合我们的数据
    model = PySRRegressor(
        model_selection="best",
        niterations=40,  # 增加迭代次数以获得更好的结果
        binary_operators=["+", "-", "*", "/"],  # 添加减法和除法
        unary_operators=["sin", "cos", "exp"],  # 保留三角函数和指数函数
        population_size=100,  # 增加种群大小
        maxsize=20,  # 允许更复杂的表达式
        timeout_in_seconds=180,  # 增加超时时间
        procs=4,  # 并行处理
        # temp_equation_file=False,  # 不生成临时方程文件
        # delete_tempfiles=True,     # 删除所有临时文件
        # tempdir=None,              # 使用系统临时目录
        # verbosity=0                # 减少输出详细程度
        output_directory=None,  # 不保存任何输出文件
    )

    print("开始PySR符号回归训练...")
    # 训练模型
    model.fit(X_train, y_train)

    print("\n发现的模型:")
    # 打印所有找到的方程
    for i, eq in enumerate(model.equations_):
        print(f"方程 {i+1}: {eq}")
    
    # 使用最佳方程进行预测
    y_pred = model.predict(X_test)
    
    # 计算R^2分数
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print(f"\n测试集 R^2 分数: {r2:.4f}")
    
    # 如果PySRRegressor有内置评分方法，也可以使用它
    if hasattr(model, 'score') and callable(getattr(model, 'score', None)):
        test_score = model.score(X_test, y_test)
        print(f"模型内置评分: {test_score:.4f}")
