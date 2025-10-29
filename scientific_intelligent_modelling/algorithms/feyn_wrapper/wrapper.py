"""Feyn 包装器（兼容层）—— 引用 QLattice 原生实现

为保持历史兼容，提供 FeynRegressor 名称，内部复用 QLatticeRegressor。
"""

from ..QLattice_wrapper.wrapper import QLatticeRegressor as FeynRegressor

__all__ = ["FeynRegressor"]

