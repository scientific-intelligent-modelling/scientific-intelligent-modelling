# scientific-intelligent-modelling

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