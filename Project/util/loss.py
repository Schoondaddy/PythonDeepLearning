import numpy as np

def MSE(outputs : np.array, targets : np.array) -> np.array:
    assert len(outputs) == len(targets)
    diff = outputs - targets
    return sum(diff * diff) / len(diff)