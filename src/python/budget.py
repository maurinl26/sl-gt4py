import numpy as np

def budget(psi: np.ndarray, psi_ref: np.ndarray):
    """Computes the budget ratio between the reference field
    and the actualized one.

    Args:
        psi (np.ndarray): field at t
        psi_ref (np.ndarray): field at reference time

    Returns:
        float: ratio of sums
    """
    
    budget_ratio = np.sum(psi) / np.sum(psi_ref)
    return budget_ratio
    