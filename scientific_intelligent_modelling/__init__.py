# scientific_intelligent_modelling/__init__.py

__all__ = [
    "SymbolicRegressor",
    "SymbolicTransformer",
    "PySRRegressor"
]

def __getattr__(name):
    if name == "SymbolicRegressor":
        from .adapters.gplearn_adapter import SymbolicRegressor
        return SymbolicRegressor
    elif name == "SymbolicTransformer":
        from .adapters.gplearn_adapter import SymbolicTransformer
        return SymbolicTransformer
    elif name == "PySRRegressor":
        from .adapters.pysr_adapter import PySRRegressor
        return PySRRegressor
    raise AttributeError(f"module {__name__} has no attribute {name}")
