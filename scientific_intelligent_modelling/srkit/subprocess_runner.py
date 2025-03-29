# srkit/subprocess_runner.py
import argparse
import json
import importlib
import sys
import traceback
import os
from config_manager import config_manager

print(f"当前工作目录: {os.getcwd()}")
print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")

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
        wrapper_module = importlib.import_module(f"scientific_intelligent_modelling.tools.{tool_name}_wrapper.wrapper")
        
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
    """根据命令类型执行相应操作"""
    action = command['action']
    tool_name = command['tool_name']
    
    # 获取正确的回归器类
    regressor_class = get_regressor_class(module, tool_name)
    
    if action == 'fit':
        return handle_fit(regressor_class, command)
    elif action == 'predict':
        return handle_predict(regressor_class, command)
    elif action == 'get_optimal_equation':
        return handle_get_optimal_equation(regressor_class, command)
    elif action == 'get_total_equations':
        return handle_get_total_equations(regressor_class, command)
    else:
        raise ValueError(f"未知操作: {action}")

def handle_get_optimal_equation(regressor_class, command):
    """处理get_optimal_equation操作，获取最优方程"""
    # 提取模型状态
    model_state = command['model_state']
    
    # 重建回归器
    regressor = regressor_class()
    
    # 如果有from_dict方法，用它加载状态
    if hasattr(regressor, 'from_dict'):
        regressor.from_dict(model_state)
    
    # 获取方程
    equation = None
    
    # 不同工具可能有不同的获取方程的方法，尝试几种常见方法
    if hasattr(regressor, 'get_optimal_equation'):
        equation = regressor.get_optimal_equation()
    elif hasattr(regressor, 'get_equation'):
        equation = regressor.get_equation()
    elif hasattr(regressor, 'get_model_string'):
        equation = regressor.get_model_string()
    elif hasattr(regressor, 'symbolic_model'):
        equation = str(regressor.symbolic_model)
    elif hasattr(regressor, 'best_'):
        equation = str(regressor.best_)
    elif hasattr(regressor, 'program_'):
        equation = str(regressor.program_)
    else:
        raise ValueError("无法从模型中提取最优方程表达式")
    
    return {
        'success': True,
        'equation': equation
    }

def handle_get_total_equations(regressor_class, command):
    """处理get_total_equations操作，获取所有方程"""
    # 提取模型状态
    model_state = command['model_state']
    
    # 重建回归器
    regressor = regressor_class()
    
    # 如果有from_dict方法，用它加载状态
    if hasattr(regressor, 'from_dict'):
        regressor.from_dict(model_state)
    
    # 获取所有方程
    equations = None
    
    # 尝试不同的方法获取所有方程
    if hasattr(regressor, 'get_total_equations'):
        equations = regressor.get_total_equations()
    elif hasattr(regressor, 'get_all_equations'):
        equations = regressor.get_all_equations()
    elif hasattr(regressor, 'hall_of_fame_'):
        equations = [str(program) for program in regressor.hall_of_fame_]
    elif hasattr(regressor, 'models_'):
        equations = [str(model) for model in regressor.models_]
    else:
        raise ValueError("无法从模型中提取所有方程表达式")
    
    return {
        'success': True,
        'equations': equations
    }

# 从配置管理器获取工具映射
def get_class_mapping_from_config():
    # 获取toolbox配置
    toolbox_config = config_manager.get_config("toolbox_config")
    tool_mapping = toolbox_config.get("tool_mapping", {})
    
    # 构建工具名到类名的映射
    class_mapping = {}
    for tool_name, tool_info in tool_mapping.items():
        class_mapping[tool_name] = tool_info.get("regressor")
    
    # 如果需要添加配置中没有的映射，可以在这里手动添加
    if "srbench" not in class_mapping:
        class_mapping["srbench"] = "SRBenchRegressor"
        
    return class_mapping

def get_regressor_class(module, tool_name):
    """根据工具名获取对应的回归器类"""
    # 工具名到类名的映射
    class_mapping = get_class_mapping_from_config()
    
    class_name = class_mapping.get(tool_name)
    if class_name and hasattr(module, class_name):
        return getattr(module, class_name)
    
    # # 后备方案：尝试查找任何以Regressor结尾的类
    # for attr_name in dir(module):
    #     if attr_name.endswith('Regressor'):
    #         return getattr(module, attr_name)
    
    raise ValueError(f"在模块 {module.__name__} 中未找到合适的回归器类")

def handle_fit(regressor_class, command):
    """处理fit操作"""
    import numpy as np
    
    # 提取数据和参数
    data = command['data']
    params = command.get('params', {})
    
    # 转换为numpy数组
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # 创建回归器并训练
    regressor = regressor_class(**params)  # 使用动态获取的类
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

def handle_predict(regressor_class, command):
    """处理predict操作"""
    import numpy as np
    
    # 提取数据和模型状态
    data = command['data']
    model_state = command['model_state']
    
    # 转换为numpy数组
    X = np.array(data['X'])
    
    # 重建回归器并预测
    regressor = regressor_class()  # 使用动态获取的类
    
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
