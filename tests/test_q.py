import feyn
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 加载数据 (以波士顿房价为例)
# 在现代 sklearn 中，波士顿房价数据已被移除，这里假设你有一个本地的 "boston.csv" 文件
# data = pd.read_csv("boston.csv")
# target = "PRICE"

# 为了可复现，我们使用 feyn 自带的示例数据
from feyn.datasets import make_regression
train_data, test_data = make_regression(n_samples=1000, n_features=5)
target = "y" # 目标变量的列名

# 2. 连接到 Qlattice (社区版需要联网)
# 这会自动连接到 Abzu 的云端免费服务器
ql = feyn.connect_qlattice()

# 3. 运行自动搜索
# auto_run 会自动处理训练、模型选择和排序
# n_epochs 控制搜索的迭代次数
models = ql.auto_run(
    data=train_data,
    output_name=target,
    kind='regression', # 也可以是 'classification'
    n_epochs=100
)

# 4. 查看最佳模型
# models 列表按性能排序，models[0] 是最佳模型
best_model = models[0]

# 5. 打印模型的数学公式
print(best_model.sympify(signif=3))

# 6. 可视化模型
best_model.plot(
    data=train_data,
    test_data=test_data,
    output_name=target
)

# 示例输出 (Sympify 结果可能如下所示):
# 3.14 * x1 + 0.5 * log(x2 + 1.0)