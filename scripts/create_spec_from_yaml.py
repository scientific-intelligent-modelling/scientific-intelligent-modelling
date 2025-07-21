
import yaml
import os
import argparse
from pathlib import Path

def create_spec_file_from_yaml(yaml_path: str, output_path: str, backend: str = 'numpy'):
    """
    Reads a metadata.yaml file and generates a corresponding llmsr specification file.

    Args:
        yaml_path (str): Path to the input metadata.yaml file.
        output_path (str): Path to write the output specification_*.txt file.
        backend (str): The computation backend to use, 'numpy' or 'torch'.
    """
    # 1. 读取并解析 YAML 文件
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    dataset_info = metadata.get('dataset', {})
    features = dataset_info.get('features', [])
    target = dataset_info.get('target', {})

    if not features or not target:
        raise ValueError(f"YAML file {yaml_path} is missing 'features' or 'target' section.")

    feature_names = [f['name'] for f in features]
    target_name = target.get('name', 'output')
    
    # 2. 根据提取的信息生成代码的各个部分
    
    # 文件顶部的描述
    description_docstring = f'''"""
Find the mathematical function skeleton that represents {target_name}, given data on {", ".join(feature_names)}.
"""'''

    # evaluate 函数中的输入切片
    # e.g., t, A = inputs[:,0], inputs[:,1]
    if len(feature_names) == 1:
        input_slicing = f'    {feature_names[0]} = inputs[:,0]'
    else:
        input_slicing = f'    {", ".join(feature_names)} = ' + ", ".join([f"inputs[:,{i}]" for i in range(len(feature_names))])


    # equation 函数的参数
    # e.g., t: np.ndarray, A: np.ndarray
    equation_args_str = ", ".join([f"{name}: np.ndarray" for name in feature_names])
    
    # equation 函数调用时的参数
    # e.g., equation(t, A, params)
    equation_call_args = ", ".join(feature_names)

    # equation 函数文档中的参数描述
    arg_docs = "\n".join([f"        {f['name']}: A numpy array representing observations of {f.get('description', f['name']).lower()}." for f in features])

    # 初始的 equation 函数体（所有特征的线性组合）
    initial_equation = " + ".join([f"params[{i}] * {name}" for i, name in enumerate(feature_names)]) + f" + params[{len(feature_names)}]"

    # 3. 使用模板组装成完整的代码
    # (目前只支持 numpy 后端)
    spec_template = f'''{description_docstring}

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
{input_slicing}
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation({equation_call_args}, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss

@equation.evolve
def equation({equation_args_str}, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for {target_name}

    Args:
{arg_docs}
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing {target_name} as the result of applying the mathematical function to the inputs.
    """
    return {initial_equation}
'''

    # 4. 写入到输出文件
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(spec_template)

    print(f"Successfully generated specification file at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a metadata.yaml file to an llmsr specification file.")
    parser.add_argument("yaml_file", type=str, help="Path to the input metadata.yaml file.")
    parser.add_argument("output_file", type=str, help="Path for the output specification .txt file.")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"], help="The backend to use for the spec file.")
    
    args = parser.parse_args()
    
    # 示例用法:
    # python scripts/create_spec_from_yaml.py tests/CRK0/metadata.yaml scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/specs/specification_crk0_numpy.txt
    
    create_spec_file_from_yaml(args.yaml_file, args.output_file, backend=args.backend) 