import numpy as np
from utils.weight import get_weight
from models.propensity import fit_propensity, predict_propensity
from algorithms.weighted_cqr import weighted_split_cqr
from algorithms.unweighted_ci import unweighted_ci
from utils.data_utils import split_by_treatment


def estimate_ite_interval(Z, X_new, mode="ATE", alpha=0.1, quantile_model=None):
    """
    Estimate ITE intervals for new samples X_new given training data Z = [X, T, Y].

    Parameters:
        Z: ndarray of shape (n, d+2), containing [X, T, Y]
        X_new: ndarray of shape (m, d), new input samples
        mode: str, one of {"ATE", "ATT", "ATC", "general"}
        alpha: float, significance level (1 - confidence)
        quantile_model: a model that implements fit(X, Y) and predict(X, quantiles)

    Returns:
        Dictionary with keys "lower" and "upper", each an ndarray of shape (m,)
    """
    X = Z[:, :-2]
    T = Z[:, -2]
    Y = Z[:, -1]

    ite_bounds = {}

    intervals = {}
    for t in [0, 1]:
        X_t, Y_t = X[T == t], Y[T == t]

        # Propensity estimation (optional for some weighting modes)
        e_x = fit_propensity(X, T) if mode != "ATE" else None

        # Weighting
        w = get_weight(X_t, t, mode=mode, e_x=e_x)

        # Apply Algorithm 1: weighted CQR
        C_L, C_R = weighted_split_cqr(X_t, Y_t, w, quantile_model, alpha=alpha)

        # Apply Algorithm 2: unweighted correction
        interval = unweighted_ci(X_new, C_L, C_R, gamma=alpha)
        intervals[t] = interval

    lower = intervals[1]["lower"] - intervals[0]["upper"]
    upper = intervals[1]["upper"] - intervals[0]["lower"]
    return {"lower": lower, "upper": upper}