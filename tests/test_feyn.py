"""feyn/Qlattice 一键运行脚本（对齐 test11 逻辑，使用包装器）

逻辑：
- 使用 feyn.datasets.make_regression 生成 (train_df, test_df)
- 通过 SymbolicRegressor('feyn') 训练
- 打印所有候选方程（最多前 10 个），以及最优方程与 MSE
"""

import feyn
import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def main():
    # 1) 生成数据（与 test11.py 一致：500 样本，3 特征）
    train_df, test_df = feyn.datasets.make_regression(n_samples=500, n_features=3)

    feature_cols = [c for c in train_df.columns if c.startswith('x')]
    X = train_df[feature_cols].values
    y = train_df['y'].values

    # 2) 训练（与 test11.py 一致：n_epochs=20；kind='regression'）
    model = SymbolicRegressor('QLattice', kind='regression', n_epochs=20)
    print('训练 feyn 模型中...')
    model.fit(X, y)

    # 3) 打印候选方程（最多前 10 个）
    equations = model.get_total_equations(10)
    print('候选方程数量(截断至10):', len(equations))
    for i, eq in enumerate(equations, 1):
        print(f'Top{i}:', eq)

    # 4) 同时打印最优方程与训练/测试 MSE 以便对齐观察
    best = model.get_optimal_equation()
    print('最优方程:', best)

    y_pred_train = model.predict(X)
    mse_train = float(np.mean((y_pred_train - y) ** 2))
    print(f'训练集 MSE: {mse_train:.6f}')

    X_test = test_df[feature_cols].values
    y_test = test_df['y'].values
    y_pred_test = model.predict(X_test)
    mse_test = float(np.mean((y_pred_test - y_test) ** 2))
    print(f'测试集 MSE: {mse_test:.6f}')


if __name__ == '__main__':
    main()
