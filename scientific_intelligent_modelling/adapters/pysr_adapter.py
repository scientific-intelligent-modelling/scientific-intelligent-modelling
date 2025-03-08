# pysr_adapter.py
# 封装 PySR 的 PySRRegressor

from ..sublibs.PySR.pysr.sr import (
    PySRRegressor as _PySRRegressor
)

def PySRRegressor(*args, **kwargs):
    # 这里可以在调用 _PySRRegressor 之前或之后，插入任何自定义逻辑
    # 比如：print("PySRRegressor is called!")
    return _PySRRegressor(*args, **kwargs)

__all__ = ["PySRRegressor"] 