from scientific_intelligent_modelling import torch_tool

# 创建模型
model = torch_tool.create_model(input_size=10, hidden_size=32)
print("模型信息:", model)

# 测试前向传播
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
output = torch_tool.forward_pass(model, data)
print("输出:", output)
