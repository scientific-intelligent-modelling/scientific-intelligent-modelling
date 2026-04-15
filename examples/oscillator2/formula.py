import numpy as np


def a(t, x, v):
    return 0.3 * np.sin(t) - 0.5 * (v ** 3) - x * v - 5.0 * x * np.exp(0.5 * x)
