import numpy as np


def a(x, v):
    return 0.8 * np.sin(x) - 0.5 * (v ** 3) - 0.2 * (x ** 3) - 0.5 * x * v - x * np.cos(x)
