from pyoperon.sklearn import SymbolicRegressor as _SymbolicRegressor

def OperonRegressor(*args, **kwargs):
    # 这里可以在调用 _SymbolicRegression 之前或之后，插入任何自定义逻辑
    # 比如：print("SymbolicRegression is called!")
    return _SymbolicRegressor(*args, **kwargs)

__all__ = ["OperonRegressor"]
