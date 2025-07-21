# ===== GROUND TRUTH REFERENCE =====
# NOTE: This ground truth serves as a reference answer for symbolic regression tasks
# This expression is NOT the original function used to generate the training data
# It is provided solely for evaluation and comparison of large model outputs
# DO NOT use this expression as the basis for data generation

def dA_dt(t, A):
    alpha = 0.18997742423620262
    alpha_z = 0.18997742423620262
    beta = 0.7497988950401423
    return -alpha * A**2 + alpha_z * A**2 / (beta * A**4 + 1)
