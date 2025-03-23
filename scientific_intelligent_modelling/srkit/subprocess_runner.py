# srkit/subprocess_runner.py
import argparse
import json
import importlib
import sys
import traceback
import os

def main():
    """子进程执行入口点"""
    parser = argparse.ArgumentParser(description='符号回归子进程执行器')
    parser.add_argument('--input', required=True, help='输入命令文件路径')
    parser.add_argument('--output', required=True, help='输出结果文件路径')
    args = parser.parse_args()
    
    try:
        # 从文件读取命令
        with open(args.input, 'r') as f:
            command = json.load(f)
        
        # 获取工具名称
        tool_name = command.get('tool_name')
        if not tool_name:
            raise ValueError("命令中缺少工具名称")
        
        # 导入工具包装器
        wrapper_module = importlib.import_module(f"tools.{tool_name}_wrapper.wrapper")
        
        # 执行命令
        result = execute_command(wrapper_module, command)
        
        # 写入结果
        with open(args.output, 'w') as f:
            json.dump(result, f)
            
    except Exception as e:
        # 错误处理
        error_result = {
            'error': True,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        with open(args.output, 'w') as f:
            json.dump(error_result, f)
        sys.exit(1)

def execute_command(module, command):
    """根据命令类型和工具执行相应操作"""
    action = command['action']
    tool_name = command['tool_name']
    
    # 获取正确的类
    regressor_class = get_regressor_class(module, tool_name)
    
    if action == 'fit':
        return handle_fit(regressor_class, command)
    elif action == 'predict':
        return handle_predict(regressor_class, command)
    else:
        raise ValueError(f"未知操作: {action}")
    
def get_regressor_class(module, tool_name):
    """根据工具名获取对应的回归器类"""
    class_mapping = {
        'gplearn': 'GPLearnRegressor',
        'pysr': 'PySRRegressor',
        'srbench': 'SRBenchRegressor',
        # 其他工具映射
    }
    
    class_name = class_mapping.get(tool_name)
    if class_name and hasattr(module, class_name):
        return getattr(module, class_name)
    
    # 后备方案：尝试使用模块中的任何回归器类
    for attr_name in dir(module):
        if attr_name.endswith('Regressor'):
            return getattr(module, attr_name)
    
    raise ValueError(f"在模块 {module.__name__} 中未找到合适的回归器类")

def handle_fit(module, command):
    """处理fit操作"""
    import numpy as np
    
    # 提取数据和参数
    data = command['data']
    params = command.get('params', {})
    
    # 转换为numpy数组
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # 创建回归器并训练
    regressor = module.SymbolicRegressor(**params)
    regressor.fit(X, y)
    
    # 序列化模型状态
    model_state = {}
    if hasattr(regressor, 'to_dict'):
        model_state = regressor.to_dict()
    else:
        # 尝试简单序列化属性
        model_state = {
            'params': params,
            # 可以添加其他通用属性
        }
    
    return {
        'success': True,
        'model_state': model_state
    }

def handle_predict(module, command):
    """处理predict操作"""
    import numpy as np
    
    # 提取数据和模型状态
    data = command['data']
    model_state = command['model_state']
    
    # 转换为numpy数组
    X = np.array(data['X'])
    
    # 重建回归器并预测
    regressor = module.SymbolicRegressor()
    
    # 如果有from_dict方法，用它加载状态
    if hasattr(regressor, 'from_dict'):
        regressor.from_dict(model_state)
    
    # 执行预测
    predictions = regressor.predict(X)
    
    # 确保预测结果可序列化
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    
    return {
        'success': True,
        'predictions': predictions
    }

if __name__ == "__main__":
    main()
