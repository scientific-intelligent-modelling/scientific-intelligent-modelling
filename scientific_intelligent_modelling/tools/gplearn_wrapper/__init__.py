# tools/gplearn_wrapper/__init__.py

# 从wrapper模块导入主要类
from .wrapper import GPLearnRegressor

# 可以提供一个别名，保持与统一接口一致的命名
SymbolicRegressor = GPLearnRegressor

# 导出版本信息
__version__ = "0.1.0"

# 导出所有应该在包级别可用的类和函数
__all__ = ['GPLearnRegressor', 'SymbolicRegressor']
