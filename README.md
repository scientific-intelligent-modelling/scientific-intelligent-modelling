# Scientific Intelligent Modelling (科学智能建模)

![Scientific Intelligent Modelling](https://raw.githubusercontent.com/scientific-intelligent-modelling/scientific-intelligent-modelling/refs/heads/main/cover.png)

一个全面的科学建模框架，为各种符号回归算法和科学计算工具提供统一的访问接口。该框架使用conda环境管理系统来隔离不同工具的依赖，通过子进程机制确保稳定运行。

## 项目概述

科学智能建模是一个Python框架，旨在简化将高级建模算法应用于科学数据的过程。该框架为多种符号回归库提供了一致的接口，包括：

- PySR (Python符号回归)
- GPlearn (用于符号回归的遗传编程)
- Operon (高性能符号回归)
- LLMSR (大语言模型进化符号回归)

这种统一的接口使研究人员和数据科学家能够轻松尝试不同的建模方法，而无需学习每个底层库的具体细节。

### 框架特点

- **统一接口**：通过一致的API访问不同的符号回归算法
- **环境隔离**：使用conda环境管理系统避免依赖冲突
- **子进程执行**：通过子进程机制保证主程序稳定性
- **配置灵活**：通过JSON配置文件轻松调整工具参数
- **可扩展性**：简单的插件架构便于添加新工具

## 安装指南

### 前提条件

- Python 3.8或更高版本
- Conda (必须，用于环境管理)

### 安装步骤

1. 克隆仓库：

   ```bash
   git clone https://github.com/yourusername/scientific-intelligent-modelling.git
   cd scientific-intelligent-modelling
   ```
2. 创建并激活conda环境：

   ```bash
   conda env create -f environment.yml
   conda activate sim
   ```
3. 以开发模式安装包：

   ```bash
   cd scientific-intelligent-modelling
   pip install -e .
   ```

### 配置环境

首次运行工具包时，需手动管理环境：

方法一：

```python
from scientific_intelligent_modelling.srkit.conda_env_manager import env_manager

# 检查所有环境状态
env_manager.check_all_environments()

# 创建特定环境
# env_manager.create_environment("test")

# 运行环境管理的命令行界面
env_manager.run_cli()
```

方法二：

```
python -m scientific_intelligent_modelling.srkit.conda_env_manager
```

二者实现效果一致

## 使用指南

### 基本示例

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

# 生成示例数据
X = np.random.rand(100, 2)
y = X[:, 0]**2 + X[:, 1] + 0.1*np.random.randn(100)

# 创建使用特定算法的回归器
regressor = SymbolicRegressor(tool_name="gplearn")

# 训练模型
regressor.fit(X, y)

# 进行预测
predictions = regressor.predict(X)

# 显示发现的最优方程
print(regressor.get_optimal_equation())

# 获取所有方程
equations = regressor.get_total_equations()
for eq in equations:
    print(eq)
```

### 算法选择与参数配置

您可以选择多种已实现的算法并传递特定参数：

```python
# 使用GPlearn，设置种群大小和代数
regressor = SymbolicRegressor(tool_name="gplearn", population_size=1000, generations=20)

# 使用Operon，设置迭代次数
regressor = SymbolicRegressor(tool_name="pyoperon", niterations=100)

# 使用PySR，设置公式复杂度和超参数
regressor = SymbolicRegressor(tool_name="pysr", maxsize=30, parsimony=0.001)
```

### 配置系统

该框架使用位于 `config`目录中的配置文件：

- `envs_config.json`：定义不同工具所需的conda环境配置

  - 包含Python版本、依赖包和安装后命令
  - 可根据需要添加新环境
- `toolbox_config.json`：定义算法映射和执行参数

  - 将工具名称映射到环境名称和具体的回归器类
  - 设置子进程超时和内存限制

配置示例：

```json
// toolbox_config.json
{
  "auto_env_creation": true,
  "subprocess_timeout": 3600,
  "memory_limit": 16000,
  "tool_mapping": {
    "gplearn": {"env": "test", "regressor": "GPLearnRegressor"},
    "pysr": {"env": "test", "regressor": "PySRRegressor"},
    "pyoperon": {"env": "test", "regressor": "OperonRegressor"}
  }
}
```

## 开发指南

### 项目结构

```
scientific_intelligent_modelling/
├── algorithms/               # 算法包装器
│   ├── base_wrapper.py       # 算法包装器的基类
│   ├── gplearn_wrapper/      # GPlearn实现
│   ├── operon_wrapper/       # Operon实现
│   └── pysr_wrapper/         # PySR实现
├── config/                   # 配置文件
│   ├── envs_config.json      # 环境配置
│   └── toolbox_config.json   # 算法参数
└── srkit/                    # 核心工具
    ├── conda_env_manager.py  # 管理conda环境
    ├── config_manager.py     # 处理配置加载
    ├── regressor.py          # 主回归器接口
    └── subprocess_runner.py  # 管理算法的子进程
```

### 工作原理

1. **统一接口层**：`SymbolicRegressor` 提供一致的API
2. **子进程执行**：通过 `subprocess_runner.py` 在隔离的环境中执行工具
3. **环境管理**：`conda_env_manager.py` 创建和维护工具所需的conda环境
4. **配置系统**：`config_manager.py` 读取和管理配置文件
5. **工具包装器**：各个算法包装器提供标准化接口

### 添加新工具

要向框架添加新算法：

1. 在 `scientific_intelligent_modelling/algorithms/`中创建一个名为 `your_algorithm_wrapper/`的新目录
2. 实现一个继承自 `base_wrapper.py`中 `BaseWrapper`类的 `wrapper.py`文件
3. 实现以下必需的方法：
   - `fit(X, y)`：训练模型
   - `predict(X)`：进行预测
   - `get_optimal_equation()`：返回最优符号表达式
   - `get_total_equations()`：返回所有获得的方程
4. 更新 `config/envs_config.json`添加工具所需的conda环境
5. 更新 `config/toolbox_config.json`添加工具映射配置

完整的包装器示例：

```python
from scientific_intelligent_modelling.algorithms.base_wrapper import BaseWrapper

class YourAlgorithmWrapper(BaseWrapper):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
  
    def fit(self, X, y):
        # 导入您的算法
        from your_package import YourModel
  
        # 创建并训练模型
        self.model = YourModel(**self.params)
        self.model.fit(X, y)
        return self
  
    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
  
    def get_optimal_equation(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        # 返回最优方程
        return str(self.model.best_equation)
  
    def get_total_equations(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        # 返回所有方程
        return [str(eq) for eq in self.model.equations]
```

然后在 `toolbox_config.json`中添加配置：

```json
"your_tool": {"env": "your_env", "regressor": "YourAlgorithmWrapper"}
```

## 技术实现细节

### conda环境管理系统

`srkit/conda_env_manager.py`提供了自动化conda环境管理的功能：

- **环境创建**：自动创建指定的conda环境，安装所需包
- **环境检查**：验证环境是否正确配置
- **命令行界面**：提供交互式界面管理环境
- **环境隔离**：确保不同工具的依赖不会冲突

### 子进程执行系统

`srkit/subprocess_runner.py`实现了子进程执行系统：

- **安全执行**：在隔离的环境中执行工具操作
- **动态加载**：根据工具名称动态导入相应的包装器
- **统一接口**：处理不同工具的fit、predict等通用操作
- **错误处理**：提供详细的错误信息和堆栈跟踪

### 序列化机制

BaseWrapper类提供了序列化和反序列化方法，确保模型状态可以在子进程之间传递：

- **serialize()**：将模型序列化为base64编码的字符串
- **deserialize()**：从序列化字符串重建模型

## 许可证

本项目根据LICENSE文件中指定的条款进行许可。

## 文档

有关更详细的信息，请参阅 `docs/`目录中的文档文件：

- `pysr.md`：PySR包装器的特定文档
- `readme.md`：附加说明和详细解释
