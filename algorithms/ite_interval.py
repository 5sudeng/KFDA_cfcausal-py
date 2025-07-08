import numpy as np
import pandas as pd
from utils.weight import get_weight
from models.propensity import fit_propensity, predict_propensity
from algorithms.weighted_cqr import weighted_split_cqr
from algorithms.unweighted_ci import unweighted_ci
from utils.data_utils import split_by_treatment


def estimate_ite_interval(Z, X_new, user_ids=None, mode="ATT", alpha=0.1, quantile_model=None, q_lo=None, q_hi=None, T=None, Y=None):
    """
    Estimate ITE intervals for treated units only, using CQR trained on control group.
    
    Parameters:
        Z: ndarray of shape (n, d+2), containing [X, T, Y]
        X_new: ndarray of shape (m, d), new treated input samples
        user_ids: optional, user ID mapping
        mode: str, one of {"ATT"} supported for now
        alpha: significance level (default 0.1 → 90% CI)
        quantile_model: model implementing fit(X, Y) and predict(X, quantiles)
        q_lo/q_hi: optional, lower/upper quantiles (defaults to alpha/2, 1-alpha/2)

    Returns:
        Dictionary with ITE bounds and counterfactual CI for T=0.
    """
    if q_lo is None or q_hi is None:
        q_lo, q_hi = alpha / 2, 1 - alpha / 2

    # Step 1: Decompose input
    if T is None or Y is None:
        X = Z[:, :-2]
        T = Z[:, -2].astype(int)
        Y = Z[:, -1]
    else:
        X = Z

    # Step 2: Split data based on mode
    if mode == "ATT":
        X_train, Y_train = X[T == 0], Y[T == 0]
        X_target, Y_target = X[T == 1], Y[T == 1]
    elif mode == "ATC":
        X_train, Y_train = X[T == 1], Y[T == 1]
        X_target, Y_target = X[T == 0], Y[T == 0]
    elif mode == "ATE":
        X_train, Y_train = X, Y
        X_target, Y_target = X, Y
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Step 3: Compute weights for training set
    e_model = fit_propensity(X, T)
    T_train = np.zeros_like(Y_train)  # Placeholder T values for weighting
    e_x = predict_propensity(X_train, e_model)
    w = get_weight(X_train, T_train, mode=mode, e_x=e_x)

    print("[DEBUG] weight w (first 10):", w[:10])
    print("[DEBUG] weight w (mean/std):", np.mean(w), np.std(w))

    # Step 4: Fit quantile model
    column_names = [f"x{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=column_names)
    quantile_model.fit(X_train_df, Y_train)

    # Step 5: Predict counterfactual quantiles for target group
    X_new_df = pd.DataFrame(X_target, columns=column_names)
    q_lo_pred = quantile_model.predict(X_new_df, quantiles=[q_lo]).ravel()
    q_hi_pred = quantile_model.predict(X_new_df, quantiles=[q_hi]).ravel()

    if user_ids is not None:
        if len(user_ids) != len(X):
            raise ValueError(f"user_ids length ({len(user_ids)}) does not match total data samples ({len(X)})")
        user_ids = user_ids[T == 1] if mode == "ATT" else (
                   user_ids[T == 0] if mode == "ATC" else user_ids)
        if len(user_ids) != len(X_target):
            raise ValueError(f"Filtered user_ids length ({len(user_ids)}) does not match target samples ({len(X_target)})")

    print("[DEBUG] q_lo_pred (first 5):", q_lo_pred[:5])
    print("[DEBUG] q_hi_pred (first 5):", q_hi_pred[:5])
    print("[DEBUG] Unique values in q_lo_pred:", np.unique(q_lo_pred[:10]))
    print("[DEBUG] Unique values in q_hi_pred:", np.unique(q_hi_pred[:10]))

    quantile_preds = np.column_stack([q_lo_pred, q_hi_pred])

    # Step 6: Construct CI for counterfactual outcome of target group
    quantile_preds_cal = quantile_model.predict(X_train_df, quantiles=[q_lo, q_hi])
    if isinstance(quantile_preds_cal, list):
        pred_lo_c, pred_hi_c = quantile_preds_cal[0].ravel(), quantile_preds_cal[1].ravel()
    else:
        pred_lo_c, pred_hi_c = quantile_preds_cal[:, 0].ravel(), quantile_preds_cal[:, 1].ravel()

    calibration_preds = np.column_stack([pred_lo_c, pred_hi_c])
    C = weighted_split_cqr(
        Y_cal=Y_train,
        quantile_preds=calibration_preds,
        sample_weight=w,
        alpha=alpha
    )

    if isinstance(C, dict):
        C_L, C_R = C["C_L"], C["C_R"]
    elif isinstance(C, (tuple, list)) and len(C) == 2:
        C_L, C_R = C
    else:
        mid = (q_lo_pred + q_hi_pred) / 2.0
        C_L = mid - C
        C_R = mid + C

    print("[DEBUG] pred_lo_c std:", np.std(pred_lo_c))
    print("[DEBUG] pred_hi_c std:", np.std(pred_hi_c))

    print("[DEBUG] C_L[:5]:", C_L[:5])
    print("[DEBUG] C_R[:5]:", C_R[:5])
    print("[DEBUG] C_L std:", np.std(C_L))
    print("[DEBUG] C_R std:", np.std(C_R))

    # Step 7: Construct CI for counterfactual outcome
    y0_cf_lower, y0_cf_upper = C_L, C_R

    # Step 8: Get factual Y for target group
    Y_target = Y_target  # already selected earlier

    assert len(X_target) == len(Y_target), "Target group X and Y must match"

    # Step 9: Compute ITE CI
    ite_lower = Y_target - y0_cf_upper
    ite_upper = Y_target - y0_cf_lower

    print("[DEBUG] y0_cf_lower (first 5):", y0_cf_lower[:5])
    print("[DEBUG] y0_cf_upper (first 5):", y0_cf_upper[:5])
    print("[DEBUG] ITE lower (first 5):", ite_lower[:5])
    print("[DEBUG] ITE upper (first 5):", ite_upper[:5])

    result = {
        "lower": ite_lower,
        "upper": ite_upper,
        "user_id": user_ids
    }

    if mode in ["ATT", "ATE"]:
        result["y0_cf_lower"] = y0_cf_lower
        result["y0_cf_upper"] = y0_cf_upper
    if mode in ["ATC", "ATE"]:
        result["y1_cf_lower"] = None
        result["y1_cf_upper"] = None

    return result