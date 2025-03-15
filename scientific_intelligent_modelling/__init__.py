# scientific_intelligent_modelling/__init__.py

__all__ = [
    "SymbolicRegressor"
    "PySRRegressor",
    "OperonRegressor"
]

def __getattr__(name):
    if name == "GplearnRegressor":
        from .adapters.gplearn_adapter import GplearnRegressor
        return GplearnRegressor
    elif name == "PySRRegressor":
        from .adapters.pysr_adapter import PySRRegressor
        return PySRRegressor
    elif name == "OperonRegressor":
        from .adapters.pyoperon_adapter import OperonRegressor
        return OperonRegressor
    raise AttributeError(f"module {__name__} has no attribute {name}")
