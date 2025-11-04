import numpy as np
import pandas as pd
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # 1) 读取 oscillator1 训练数据（x, v -> a）
    train_csv = 'scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/data/oscillator1/train.csv'
    df = pd.read_csv(train_csv).astype(float).to_numpy()
    X = df[:, :-1]
    y = df[:, -1]

    print("=" * 60)
    print("Oscillator1（背景驱动）测试")
    print("=" * 60)
    print(f"样本数: {X.shape[0]} | 特征数: {X.shape[1]}")

    # 2) 背景描述（无需 spec_path）：阻尼非线性振子，输入为位置 x 与速度 v，输出为加速度 a
    background = """
    这是一个关于阻尼非线性振子（含外驱动）的建模问题。
    自变量：x（位置），v（速度）；目标变量：a（加速度）。
    常见结构包含：线性刚度项（k*x）、线性阻尼项（c*v）、非线性刚度项（如 α*x^3）、
    以及可能的饱和/软硬化项、正弦/余弦等周期成分（若隐含驱动）。
    方程形式为 a = f(x, v; params)。
    """

    # 3) 构造与训练模型（使用背景自动生成 spec）
    model = SymbolicRegressor(
        'drsr',
        use_api=True,
        api_model='blt/gpt-3.5-turbo',  # 或 deepseek/siliconflow/ollama
        background=background,
        samples_per_prompt=4,
        max_samples=40,
        evaluate_timeout_seconds=12,
        problem_name='oscillator1_bg',
    )

    print("\n开始训练（背景→自动spec）...")
    model.fit(X, y)

    # 4) 输出最优方程与简单评估
    print("\n" + "=" * 60)
    print("最优方程:")
    print("=" * 60)
    print(model.get_optimal_equation())

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print("\n" + "=" * 60)
    print(f"训练集 MSE: {mse:.6f}  |  R²: {r2:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

