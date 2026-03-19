# TPSR 使用文档

## 1. 两种主干

### e2e（Transformer）

```python
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
import numpy as np

reg = SymbolicRegressor(
    "tpsr",
    backbone_model="e2e",
    symbolicregression_model_path="/path/to/model.pt",
    beam_size=6,
    width=3,
    rollout=2,
    horizon=20,
    force_cpu=True,
)
```

### NeSymReS

```python
reg = SymbolicRegressor(
    "tpsr",
    backbone_model="nesymres",
    nesymres_model_path="/path/to/10M.ckpt",
    nesymres_cfg_path="scientific_intelligent_modelling/algorithms/tpsr_wrapper/tpsr/nesymres/jupyter/100M/config.yaml",
    nesymres_eq_setting_path="scientific_intelligent_modelling/algorithms/tpsr_wrapper/tpsr/nesymres/jupyter/100M/eq_setting.json",
    force_cpu=True,
)
```

## 2. 关键参数

- `backbone_model`: `e2e` | `nesymres`
- `beam_size`, `width`, `rollout`, `horizon`, `num_beams`
- `force_cpu`

## 3. 权重配置

- 默认会查找：
  - e2e：`.../symbolicregression/weights/model.pt`
  - NeSymReS：`.../nesymres/weights/10MCompleted.ckpt` 或 `10M.ckpt`
- 也可通过环境变量：
  - `TPSR_E2E_MODEL_PATH`
  - `TPSR_NESYMRES_MODEL_PATH`

## 4. 闭环脚本

```bash
python check/check_tpsr.py
```
