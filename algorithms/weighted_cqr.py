import numpy as np
from utils.weighted_quantile import weighted_quantile

def weighted_cqr(Y_cal, quantile_preds, sample_weight, alpha=0.1):
    """
    Perform Weighted Conformalized Quantile Regression (CQR).

    Parameters
    ----------
    Y_cal : array-like, shape (n_val,)
        True labels for the calibration set.
    quantile_preds : array-like, shape (n_val, 2)
        Predicted lower and upper quantiles for calibration set.
    sample_weight : array-like, shape (n_val,)
        Sample weights for conformal calibration.
    alpha : float
        Miscoverage level (e.g., 0.1 for 90% interval).

    Returns
    -------
    q_hat : float
        Calibrated quantile residual used for prediction interval adjustment.
    """
    q_lo = quantile_preds[:, 0]
    q_hi = quantile_preds[:, 1]

    # Conformity score: max(lower deviation, upper deviation)
    conformity_scores = np.maximum(q_lo - Y_cal, Y_cal - q_hi)

    # Weighted quantile of the conformity scores
    q_hat = weighted_quantile(conformity_scores, quantiles=1 - alpha, sample_weight=sample_weight)

    return q_hat