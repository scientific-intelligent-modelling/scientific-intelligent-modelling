# gplearn_adapter.py
# 假设你要封装 gplearn.genetic 里的 SymbolicRegressor / SymbolicTransformer

from gplearn.genetic import (
    SymbolicRegressor as _SymbolicRegressor
)

# 你的适配器：对外仍然叫 SymbolicRegressor
def GplearnRegressor(*args, **kwargs):
    # 这里可以在调用 _SymbolicRegressor 之前或之后，插入任何自定义逻辑
    # 比如：print("SymbolicRegressor is called!")
    return _SymbolicRegressor(*args, **kwargs)

__all__ = ["GplearnRegressor"]
