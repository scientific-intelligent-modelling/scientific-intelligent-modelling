"""QLattice 回归器包装器（别名）

说明：
- 为了让工具名与算法名一致，提供 QLattice 别名包装器
- 复用现有 feyn_wrapper 的实现，类名导出为 QLatticeRegressor
"""

from ..feyn_wrapper.wrapper import FeynRegressor as QLatticeRegressor

__all__ = ["QLatticeRegressor"]

