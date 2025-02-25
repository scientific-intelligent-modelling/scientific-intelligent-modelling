# gplearn_adapter.py
# 假设你要封装 gplearn.genetic 里的 SymbolicRegressor / SymbolicTransformer

from ..sublibs.gplearn.gplearn.genetic import (
    SymbolicRegressor as _SymbolicRegressor,
    SymbolicTransformer as _SymbolicTransformer
)

# 你的适配器：对外仍然叫 SymbolicRegressor
def SymbolicRegressor(*args, **kwargs):
    # 这里可以在调用 _SymbolicRegressor 之前或之后，插入任何自定义逻辑
    # 比如：print("SymbolicRegressor is called!")
    return _SymbolicRegressor(*args, **kwargs)

def SymbolicTransformer(*args, **kwargs):
    # 同理，如果要修改初始化参数、加个 Hook 等，都可以在此处编写
    return _SymbolicTransformer(*args, **kwargs)

__all__ = ["SymbolicRegressor", "SymbolicTransformer"]
