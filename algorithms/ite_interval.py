import numpy as np
import pandas as pd
from utils.weight import get_weight
from models.propensity import fit_propensity, predict_propensity
from algorithms.weighted_cqr import weighted_split_cqr
from algorithms.unweighted_ci import unweighted_ci
from utils.data_utils import split_by_treatment
from sklearn.model_selection import train_test_split


def estimate_ite_interval(Z, X_new, user_ids=None, mode="ATT", alpha=0.1, quantile_model=None, q_lo=None, q_hi=None, T=None, Y=None):
    """
    Estimate ITE intervals for treated units only, using CQR trained on control group.
    
    Parameters:
        Z: ndarray of shape (n, d+2), containing [X, T, Y]
        X_new: ndarray of shape (m, d), new treated input samples
        user_ids: optional, user ID mapping
        mode: str, one of {"ATT"} supported for now
        alpha: significance level (default 0.1 -> 90% CI)
        quantile_model: model implementing fit(X, Y) and predict(X, quantiles)
        q_lo/q_hi: optional, lower/upper quantiles (defaults to alpha/2, 1-alpha/2)

    Returns:
        Dictionary with ITE bounds and counterfactual CI for T=0.
    """
    if q_lo is None or q_hi is None:
        q_lo, q_hi = alpha / 2, 1 - alpha / 2

    # Step 0: Split into Z1 (train) and Z2 (estimation)
    if T is None or Y is None:
        # Assume Z = [X, T, Y]
        Z1, Z2 = train_test_split(Z, test_size=0.5, random_state=42)
        X1, T1, Y1 = Z1[:, :-2], Z1[:, -2].astype(int), Z1[:, -1]
        X2, T2, Y2 = Z2[:, :-2], Z2[:, -2].astype(int), Z2[:, -1]
    else:
        # If T and Y are passed separately, then assume Z = X
        Z_combined = np.column_stack([Z, T, Y])
        Z1, Z2 = train_test_split(Z_combined, test_size=0.5, random_state=42)
        X1, T1, Y1 = Z1[:, :-2], Z1[:, -2].astype(int), Z1[:, -1]
        X2, T2, Y2 = Z2[:, :-2], Z2[:, -2].astype(int), Z2[:, -1]

    # Z1 = training set for CQR / calibration (dksl used to fit quantiles and conformal interval)
    # Z2 = evaluation set where counterfactual inference and ITE estimation are applied

    # Step 1: Decompose input
    # Use X2, T2, Y2 for estimation
    X = X2
    T = T2
    Y = Y2

    # Step 2: From Z1 -> training set (control), from Z2 -> target treated samples (to evaluate ITE)
    if mode == "ATT":
        X_train, Y_train = X1[T1 == 0], Y1[T1 == 0]
        X_target, Y_target = X[T == 1], Y[T == 1]
    elif mode == "ATC":
        X_train, Y_train = X1[T1 == 1], Y1[T1 == 1]
        X_target, Y_target = X[T == 0], Y[T == 0]
    elif mode == "ATE":
        X_train, Y_train = X1, Y1
        X_target, Y_target = X, Y
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Step 3: Compute weights for training set
    e_model = fit_propensity(X1, T1)
    if mode == "ATT":
        T_train = np.zeros_like(Y_train)  # treated <- control
    elif mode == "ATC":
        T_train = np.ones_like(Y_train)   # control <- treated
    else:
        T_train = T1  # use true T1 if mode == ATE
    e_x1 = predict_propensity(X_train, e_model)
    w = get_weight(X_train, T_train, mode=mode, e_x=e_x1)

    print("[DEBUG] weight w (first 10):", w[:10])
    print("[DEBUG] weight w (mean/std):", np.mean(w), np.std(w))

    # Step 3.5: Split X_train/Y_train/w into calibration and training sets
    X_tr, X_cal, Y_tr, Y_cal, w_tr, w_cal = train_test_split(
        X_train, Y_train, w, test_size=0.5, random_state=42
    )

    # Step 4: Fit quantile model
    column_names = [f"x{i}" for i in range(X.shape[1])]
    X_tr_df = pd.DataFrame(X_tr, columns=column_names)
    quantile_model.fit(X_tr_df, Y_tr)

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

    quantile_preds = np.column_stack([q_lo_pred, q_hi_pred])

    # Step 5.5: Compute test-time sample weight w(x) for eta(x)
    e_x_target = predict_propensity(X_target, e_model)
    T_target_placeholder = np.zeros_like(Y_target)
    w_x = get_weight(X_target, T_target_placeholder, mode=mode, e_x=e_x_target)

    # Step 6: Use Z1 (training/calibration set) for weighted conformal inference
    X_cal_df = pd.DataFrame(X_cal, columns=column_names)
    quantile_preds_cal = quantile_model.predict(X_cal_df, quantiles=[q_lo, q_hi])
    if isinstance(quantile_preds_cal, list):
        pred_lo_c, pred_hi_c = quantile_preds_cal[0].ravel(), quantile_preds_cal[1].ravel()
    else:
        pred_lo_c, pred_hi_c = quantile_preds_cal[:, 0].ravel(), quantile_preds_cal[:, 1].ravel()

    calibration_preds = np.column_stack([pred_lo_c, pred_hi_c])
    C = weighted_split_cqr(
        Y_cal=Y_cal,
        quantile_preds=calibration_preds,
        sample_weight=w_cal,
        alpha=alpha,
        test_weight=w_x  # Pass test-time sample weights for proper eta(x) computation
    )

    if isinstance(C, dict):
        C_L, C_R = C["C_L"], C["C_R"]
    elif isinstance(C, (tuple, list)) and len(C) == 2:
        C_L, C_R = C
    else:
        mid = (q_lo_pred + q_hi_pred) / 2.0
        C_L = mid - C
        C_R = mid + C

    # Step 7: Construct CI for counterfactual outcome
    y0_cf_lower, y0_cf_upper = None, None
    y1_cf_lower, y1_cf_upper = None, None
    if mode == "ATT" or mode == "ATE":
        y0_cf_lower, y0_cf_upper = C_L, C_R
    if mode == "ATC" or mode == "ATE":
        y1_cf_lower, y1_cf_upper = C_L, C_R

    # Step 8: Get factual Y for target group
    Y_target = Y_target  # already selected earlier

    assert len(X_target) == len(Y_target), "Target group X and Y must match"

    # Step 9: Compute ITE CI
    if mode == "ATT":
        ite_lower = Y_target - y0_cf_upper
        ite_upper = Y_target - y0_cf_lower
    elif mode == "ATC":
        ite_lower = y1_cf_lower - Y_target
        ite_upper = y1_cf_upper - Y_target
    elif mode == "ATE":
        ite_lower = y1_cf_lower - y0_cf_upper
        ite_upper = y1_cf_upper - y0_cf_lower
    else:
        raise ValueError("Unsupported mode for ITE computation")

    result = {
        "lower": ite_lower,
        "upper": ite_upper,
        "user_id": user_ids
    }

    if mode == "ATT":
        result["y0_cf_lower"] = y0_cf_lower
        result["y0_cf_upper"] = y0_cf_upper
        result["y1_cf_lower"] = None
        result["y1_cf_upper"] = None
    elif mode == "ATC":
        result["y0_cf_lower"] = None
        result["y0_cf_upper"] = None
        result["y1_cf_lower"] = y1_cf_lower
        result["y1_cf_upper"] = y1_cf_upper
    elif mode == "ATE":
        result["y0_cf_lower"] = y0_cf_lower
        result["y0_cf_upper"] = y0_cf_upper
        result["y1_cf_lower"] = y1_cf_lower
        result["y1_cf_upper"] = y1_cf_upper

    result["target_mask"] = T == 0 if mode == "ATC" else (T == 1 if mode == "ATT" else np.ones_like(T, dtype=bool))

    return result