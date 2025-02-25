# scientific_intelligent_modelling/__init__.py

# 导出一些核心的东西（假设你自己写的在 core 里）
# from .core.main_function import main_func

# 再导出外部库的 “适配器” 中最常用的对象
from .adapters.gplearn_adapter import SymbolicRegressor, SymbolicTransformer

__all__ = [
    # "main_func",
    "SymbolicRegressor",
    "SymbolicTransformer",
]
