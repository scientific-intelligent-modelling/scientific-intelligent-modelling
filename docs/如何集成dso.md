# 如何集成DSO(deep-symbolic-optimization)环境

## 环境测试（与之前的环境融合，这里先忽略这一步）

测试dso环境是否与之前的环境冲突，若冲突了，则需要新建一个新的环境，再集成进去

这里先不进行环境的测试，直接新建一个名为“dso”的环境

## 环境测试（测试能否在虚拟环境中跑通）

要把工具集成进去，第一步，就是得在自己的环境里面能够成功运行。

1、主仓库进行clone：

```bash
git clone https://github.com/ziwenhahaha/scientific-intelligent-modelling.git --recursive
cd scientific-intelligent-modelling
```

2、把子仓库进行添加submodule

2.1 fork到仓库中
把https://github.com/dso-org/deep-symbolic-optimization 进行fork到对应的organization里面
注意owner设置为scientific-intellgent-modelling
![](./image/fork_dso.png)

```bash
mkdir scientific_intelligent_modelling/algorithms/dso_wrapper
git submodule add https://github.com/scientific-intelligent-modelling/deep-symbolic-optimization.git scientific_intelligent_modelling/algorithms/dso_wrapper/dso
```

3、创建conda子环境：dso

```bash
conda create -n dso  python=3.7
conda activate dso
pip install -e scientific_intelligent_modelling/algorithms/dso_wrapper/dso/dso
```

4、创建测试脚本：
``` python
from dso import DeepSymbolicRegressor
import numpy as np
# Generate some data
np.random.seed(0)
X = np.random.random((10, 2))
y = np.sin(X[:,0]) + X[:,1] ** 2

# Create the model
model = DeepSymbolicRegressor() # Alternatively, you can pass in your own config JSON path

# Fit the model
model.fit(X, y) # Should solve in ~10 seconds

# View the best expression
print(model.program_.pretty())

# Make predictions
model.predict(2 * X)

```

发现有报错，去check官方文档的issue，一般都有相应的解决方案。

这里的一个解决方案就是在 https://github.com/dso-org/deep-symbolic-optimization/issues/67  具体问题需要自行更改。
然后把子仓库对应的地方进行修改。

测试发现，无问题

5、因为这个仓库已经包含有sklearn like的api，所以直接进行封装wrapper即可。

由于过于简单，所以直接自己写了，没有用大模型辅助

6、在tests/test_new_arch.py里面进行测试
