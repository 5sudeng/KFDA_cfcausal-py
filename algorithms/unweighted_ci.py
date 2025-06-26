import numpy as np

def unweighted_ci(X, C_L, C_R, gamma=0.1):
    """
    Construct unweighted conformal prediction intervals based on empirical quantiles.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The feature matrix (only used for determining n).
    C_L : array-like, shape (n_calibration,)
        Lower conformity scores (e.g., prediction - true value).
    C_R : array-like, shape (n_calibration,)
        Upper conformity scores (e.g., true value - prediction).
    gamma : float
        Miscoverage level (e.g., 0.1 for 90% intervals).

    Returns
    -------
    ci_lower : np.ndarray
        Lower interval width (same value repeated).
    ci_upper : np.ndarray
        Upper interval width (same value repeated).
    """
    n = len(C_L)
    k = int(np.floor((1 - gamma) * (n + 1))) - 1
    k = np.clip(k, 0, n - 1)

    q_L = np.sort(C_L)[k]
    q_R = np.sort(C_R)[k]

    return np.full(len(X), q_L), np.full(len(X), q_R)