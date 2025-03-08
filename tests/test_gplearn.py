from juliacall import Main as jl  # 或者：from julia import Julia; jl = Julia(...)

jl.seval('import Pkg')
jl.seval('Pkg.add("SymbolicRegression")')  # 安装
jl.seval('using SymbolicRegression')       # 再加载

print(1)


# test.py
import scientific_intelligent_modelling as sim

def test():
    pass

def __main__():
    test()
    print("test passed!")