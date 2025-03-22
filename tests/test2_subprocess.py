from scientific_intelligent_modelling import sklearn_tool

# 测试基本功能
X = [[1, 2], [3, 4], [5, 6]]
y = [0, 1, 0]

# 训练模型
model = sklearn_tool.train_classifier(X, y, model_type='decision_tree111')
print("训练结果:", model)

# 预测
predictions = sklearn_tool.predict(model, X)
print("预测结果:", predictions)
