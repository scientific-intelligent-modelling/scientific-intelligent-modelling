# Filename: pysr_adapter.py
# 假设你要封装 pysr 里的 PySRRegressor

from pysr import PySRRegressor as _PySRRegressor

def PySRRegressor(*args, **kwargs):
    # 这里可以在调用 _SymbolicRegressor 之前或之后，插入任何自定义逻辑
    # 比如：print("SymbolicRegressor is called!")
    return _PySRRegressor(*args, **kwargs)

__all__ = ["PySRRegressor"]