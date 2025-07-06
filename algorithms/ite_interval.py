import numpy as np
from utils.weight import get_weight
from models.propensity import fit_propensity, predict_propensity
from algorithms.weighted_cqr import weighted_split_cqr
from algorithms.unweighted_ci import unweighted_ci
from utils.data_utils import split_by_treatment


def estimate_ite_interval(Z, X_new, mode="ATE", alpha=0.1, quantile_model=None, q_lo=None, q_hi=None):
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
    if q_lo is None or q_hi is None:
        q_lo, q_hi = alpha / 2, 1 - alpha / 2

    X = Z[:, :-2]
    T = Z[:, -2].astype(int)
    Y = Z[:, -1]

    ite_bounds = {}

    e_model = fit_propensity(X, T)

    intervals = {}
    for t in [0, 1]:
        X_t, Y_t = X[T == t], Y[T == t]

        e_x = predict_propensity(X_t, e_model)

        # Ensure treatment is passed as array
        t_arr = np.full(len(X_t), t)
        w = get_weight(X_t, t_arr, mode=mode, e_x=e_x)

        # Apply Algorithm 1: weighted CQR
        quantile_model.fit(X_t, Y_t)
        low = quantile_model.predict(X_t, quantiles=[q_lo])[0]
        high = quantile_model.predict(X_t, quantiles=[q_hi])[0]
        quantile_preds = np.column_stack([low, high])
        C = weighted_split_cqr(
            Y_cal=Y_t,
            quantile_preds=quantile_preds,
            sample_weight=w,
            alpha=alpha
        )
        if isinstance(C, dict):
            C_L, C_R = C["C_L"], C["C_R"]
        elif isinstance(C, (tuple, list)) and len(C) == 2:
            C_L, C_R = C
        elif isinstance(C, (int, float, np.floating, np.integer)):
            mid = (low + high) / 2.0
            C_L = mid - C
            C_R = mid + C
        else:
            raise ValueError(f"Expected C to be a dictionary, 2-tuple, or scalar, but got type: {type(C)} with value: {C}")

        # Apply Algorithm 2: unweighted correction
        interval = unweighted_ci(X_new, C_L, C_R, gamma=alpha)
        lower, upper = interval
        intervals[t] = {"lower": lower, "upper": upper}

    lower = intervals[1]["lower"] - intervals[0]["upper"]
    upper = intervals[1]["upper"] - intervals[0]["lower"]
    return {"lower": lower, "upper": upper}