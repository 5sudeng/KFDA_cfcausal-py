import numpy as np
from utils.weighted_quantile import weighted_quantile

def weighted_split_cqr(
    Y_cal,
    quantile_preds,
    sample_weight,
    alpha: float = 0.1,
    test_weight=None,
):
    """
    Parameters
    ----------
    Y_cal : array-like, shape (n_cal,)
        True outcomes on the calibration split.
    quantile_preds : array-like, shape (n_cal, 2)
        Predicted lower / upper quantiles on the calibration split;
        column 0 = \hat q_alpha/2, column 1 = \hat q_{1‑alpha/2}.
    sample_weight : array-like, shape (n_cal,)
        Importance weights w_i assigned to each calibration point.
    alpha : float, default=0.1
        Target mis‑coverage level (e.g. 0.1 -> 90 % coverage).
    test_weight : scalar or array-like, shape (n_test,), optional
        Importance weight(s) w_x for the test point(s).  If None
        a single weight of 1.0 is assumed.

    Returns
    -------
    q_hat : ndarray, shape (n_test,)
        Conformal residuals C(x) for every test point.  These are the
        values that must be added to the predicted quantiles
        (lower − C, upper + C) to obtain valid prediction intervals.
    """
    Y_cal = np.asarray(Y_cal, dtype=float)
    quantile_preds = np.asarray(quantile_preds, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    if quantile_preds.ndim != 2 or quantile_preds.shape[1] != 2:
        raise ValueError("`quantile_preds` must have shape (n_cal, 2)")

    # conformity scores s_i = max{ q_lo − y, y − q_hi }
    q_lo, q_hi = quantile_preds[:, 0], quantile_preds[:, 1]
    conformity_scores = np.maximum(q_lo - Y_cal, Y_cal - q_hi)

    # ensure test_weight is a 1‑D array
    if test_weight is None:
        test_weight = np.array([1.0], dtype=float)
    else:
        test_weight = np.atleast_1d(test_weight).astype(float)

    q_hat_list = []
    for w_x in test_weight:
        # Probability mass p_i  
        p_i = sample_weight / (sample_weight.sum() + w_x)

        # (1 − alpha)‑quantile of conformity scores under p_i
        q_hat = weighted_quantile(
            conformity_scores,
            quantiles=1 - alpha,
            sample_weight=p_i,
        )
        q_hat_list.append(q_hat)

    return np.asarray(q_hat_list)