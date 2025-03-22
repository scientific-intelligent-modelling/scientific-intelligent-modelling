# scientific-intelligent-modelling

## 1.0.0版本的目录
toolbox/
├── pyproject.toml                    # 项目配置和依赖定义
├── README.md                         # 项目说明文档
│
├── toolbox/                          # 主包目录
│   ├── __init__.py                   # 主入口，提供工具导入接口
│   ├── enhanced_subprocess.py        # 依赖隔离管理器
│   ├── cuda_process_manager.py       # CUDA隔离管理器
│   ├── subprocess_runner.py          # 依赖隔离子进程运行器
│   ├── cuda_process_runner.py        # CUDA隔离子进程运行器
│   ├── dependency_analyzer.py        # 依赖分析工具
│   │
│   ├── torch_1_8_tool/               # 需要CUDA的工具包
│   │   ├── __init__.py               # 工具包入口
│   │   ├── model.py                  # 模型定义
│   │   ├── train.py                  # 训练功能
│   │   └── utils.py                  # 工具函数
│   │
│   └── sklearn_tool/                 # 不需要CUDA的工具包
│       ├── __init__.py               # 工具包入口
│       ├── classifier.py             # 分类器实现
│       ├── preprocessing.py          # 数据预处理
│       └── evaluation.py             # 评估函数
│
├── shared_deps/                      # 共享依赖目录
├── tools_env/                        # 工具特定依赖目录
│
├── config/                           # 配置文件目录
│   ├── toolbox_config.json           # 全局配置文件
│   ├── cuda_config.json              # CUDA环境配置
│   ├── dependencies_config.json      # 依赖关系配置
│   └── tools_config.json             # 工具特定配置
│
├── requirements/                     # 依赖文件目录
│   ├── requirements_torch_1_8.txt    # PyTorch 1.8工具的依赖文件
│   └── requirements_sklearn.txt      # scikit-learn工具的依赖文件
│
├── scripts/                          # 安装和管理脚本
│   ├── setup_environment.py          # 环境设置脚本
│   ├── install_tool.py               # 单个工具安装脚本
│   ├── update_dependencies.py        # 依赖更新脚本
│   └── clean_environments.py         # 环境清理脚本
│
└── install_tools.py                  # 工具环境安装脚本

## Installation

``` bash
conda create -n python310 python=3.10
```

在虚拟环境中安装cuda环境，cuda版本为11.8
``` bash
conda install -c nvidia cuda-toolkit=11.8 
```

执行完之后，使用
which nvcc
查看，nvcc所在的区域，若nvcc在虚拟路径内，如`/home/ziwen/anaconda3/envs/python310/bin/nvcc` 则为安装成功。

然后在虚拟环境中执行：
``` bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export CUDA_HOME=$(dirname $(dirname $(which nvcc)))' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
### 
先提前安装pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### 基本安装（没有 PyTorch）
pip install .

### Torch 安装
pip install ".[torch]"