# Examples 目标格式

当前仓库中，`examples/` 风格的数据集目录最少包含：

```text
<dataset_dir>/
├── train.csv
├── valid.csv
├── id_test.csv
├── ood_test.csv
├── metadata.yaml
└── formula.py        # 可选
```

## CSV 约束

- 四个 CSV 的列头必须一致。
- 必须至少包含：
  - 1 个目标列
  - 1 个或多个特征列
- 推荐列顺序：
  - 先特征列
  - 最后目标列
- 所有用于符号回归的列都应是数值型。

## metadata.yaml 最低要求

顶层需包含：

```yaml
dataset:
  name: <dataset_name>
  description: <dataset_description>
  splits:
    train:
      file: train.csv
      samples: <int>
    valid:
      file: valid.csv
      samples: <int>
    id_test:
      file: id_test.csv
      samples: <int>
    ood_test:
      file: ood_test.csv
      samples: <int>
  features:
    - name: <feature_1>
      type: continuous
      description: <text>
  target:
    name: <target_name>
    type: continuous
    description: <text>
```

## 当前代码真实依赖的字段

严格依赖：

- `dataset.target.name`
- `dataset.features[*].name`
- split 对应的 CSV 文件存在

会被现有逻辑消费的入口：

- `scientific_intelligent_modelling/pipelines/iterative_experiment.py`
- `check/run_phys_osc_task.py`

## formula.py

若存在，应提供一个可读、可执行的真值函数。

例如：

```python
import numpy as np

def y(x0, x1):
    return x0**2 + x1
```

建议：

- 函数名优先与 `dataset.target.name` 一致
- 若公式里会用 `sin/exp/log` 等数组运算，默认写 `import numpy as np`

如果真实公式未知，就不要伪造 `formula.py`。

## 公式一致性校验

若 metadata 声明了：

```yaml
dataset:
  ground_truth_formula:
    file: formula.py
```

则应对一个或多个 split 做公式代入验证：

- 读取特征列
- 把特征列按 metadata.features 顺序传给公式函数
- 比较公式输出和目标列
- 至少记录或检查：
  - `rmse`
  - `nmse`
  - `max_abs_err`
