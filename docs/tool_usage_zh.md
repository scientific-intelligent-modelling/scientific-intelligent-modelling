# 符号回归工具使用手册（简明版）

本文档用于快速上手本仓库的算法工具。示例以 `scientific_intelligent_modelling` 项目上下文为准。

---

## 1. 通用调用方式（推荐）

所有算法统一通过 `SymbolicRegressor` 调用：

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

# 示例数据
rng = np.random.RandomState(42)
X = rng.rand(30, 2)
y = 0.5 * X[:, 0] + 2.0 * X[:, 1] + 0.01 * rng.randn(30)

reg = SymbolicRegressor(
    "tpsr",          # 工具名：gplearn/pysr/pyoperon/llmsr/dso/tpsr/e2esr/QLattice/iMCTS/drsr
    problem_name="demo",
    seed=42,
    force_cpu=True,    # 建议先 CPU 验证
)

reg.fit(X, y)
print("best:", reg.get_optimal_equation())
print("top:", reg.get_total_equations()[:3])
print("pred:", reg.predict(X[:5]))
```

常用接口：
- `fit(X, y)`
- `get_optimal_equation()`
- `get_total_equations()`
- `predict(X)`

---

## 2. 快速闭环检查（check 脚本）

在仓库根目录执行：

```bash
python check/check_gplearn.py
python check/check_pysr.py
python check/check_pyoperon.py
python check/check_llmsr.py
python check/check_dso.py
python check/check_tpsr.py
python check/check_drsr.py
python check/check_qlattice.py
python check/check_imcts.py
python check/check_e2esr.py
```

---

## 3. TPSR 使用（e2e + nesymres）

`check/check_tpsr.py` 同时包含两条 backbone：`e2e` 和 `nesymres`。

### 3.1 首先准备权重

- e2e：`scientific_intelligent_modelling/algorithms/tpsr_wrapper/tpsr/symbolicregression/weights/model.pt`
- nesymres：`scientific_intelligent_modelling/algorithms/tpsr_wrapper/tpsr/nesymres/weights/10M.ckpt`

### 3.2 设置环境变量并运行

```bash
# 只跑 e2e
env TPSR_E2E_MODEL_PATH=.../scientific_intelligent_modelling/algorithms/tpsr_wrapper/tpsr/symbolicregression/weights/model.pt \
    python check/check_tpsr.py

# 同时跑两条路径（推荐）
TPSR_E2E_MODEL_PATH=.../scientific_intelligent_modelling/algorithms/tpsr_wrapper/tpsr/symbolicregression/weights/model.pt \
TPSR_NESYMRES_MODEL_PATH=.../scientific_intelligent_modelling/algorithms/tpsr_wrapper/tpsr/nesymres/weights/10M.ckpt \
python check/check_tpsr.py
```

成功闭环输出关键字：
- `[check_tpsr] e2e OK`
- `[check_tpsr] nesymres OK`

---

## 4. DSO 使用提示

- 直接跑 `python check/check_dso.py`
- 如需离线/在线行为，优先查看 `check/check_dso.py` 里的参数与环境变量说明
- 先以 `force_cpu=True` 进行快速验证，验证通过再放开 GPU/并发

---

## 5. 常见问题（先看这三条）

1) 报“未找到权重”
- TPSR：确认 `TPSR_E2E_MODEL_PATH` 或 `TPSR_NESYMRES_MODEL_PATH` 指向真实文件

2) 子进程报错（序列化/反序列化）
- 首先确认对应算法最近是否可在 check 脚本中独立通过

3) 运行太慢
- 降低数据规模（`horizon/rollout/beam_size`）或固定 `force_cpu=True`

---

## 6. 文件定位（快捷）

- 通用入口：`scientific_intelligent_modelling/srkit/regressor.py`
- 运行器（子进程）：`scientific_intelligent_modelling/srkit/subprocess_runner.py`
- 检查脚本目录：`check/`
- 工具包装器：`scientific_intelligent_modelling/algorithms/`
