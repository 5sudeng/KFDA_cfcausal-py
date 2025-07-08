"""
python main.py --data data/250707_statin_ldl.csv --mode ATT --alpha 0.1 --quantile_model qgb
"""

import argparse
import numpy as np
import pandas as pd
import inspect

from sklearn.preprocessing import OneHotEncoder

from utils.data_utils import load_dataset, split_train_calibration
from models.quantile import QuantileGradientBoosting
from models.propensity import fit_propensity, predict_propensity
from utils.weight import get_weight
from algorithms.weighted_cqr import weighted_split_cqr
from algorithms.unweighted_ci import unweighted_ci
from algorithms.ite_interval import estimate_ite_interval

# -----------------------------------------------------------------------------#
# Helper: pick quantile model
# -----------------------------------------------------------------------------#
def get_quantile_model(name: str, quantiles):
    name = name.lower()
    if name in ["qgb", "lgbm", "quantile_gb"]:
        return QuantileGradientBoosting(quantiles=quantiles, n_estimators=200)
    else:
        raise ValueError(f"Unsupported quantile model: {name}")


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main(args):
    # 1. Load data
    df = load_dataset(args.data)
    
    # Remove rows with NaNs in relevant covariates
    covariate_cols = ["age", "education", "gender", "occupation", "hospital", "BMI", "height", "weight", "ldl_value_before_trt"]
    df = df[df[covariate_cols].notnull().all(axis=1)]
    print(f"[INFO] After dropping NaNs, data shape: {df.shape}")
    user_ids = df["user_id"].values

    X_raw = df[["age", "education", "gender", "occupation", "hospital", "BMI", "height", "weight", "ldl_value_before_trt"]]

    categorical_cols = ["education", "gender", "occupation", "hospital"]
    numerical_cols = [col for col in X_raw.columns if col not in categorical_cols]

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X_raw[categorical_cols])
    X_new = np.hstack([X_raw[numerical_cols].values, X_cat])

    # Outcome Y: ldl_value_after_trt for *all* users (controls included)
    T = df["treated"].values
    Y = df["ldl_value_after_trt"].values  # use the post‑measurement for every user

    # Drop any rows where Y is missing
    valid_idx = ~np.isnan(Y)
    X_new   = X_new[valid_idx]
    T       = T[valid_idx]
    Y       = Y[valid_idx]
    user_ids = user_ids[valid_idx]

    # 2. Choose quantile model
    q_lo, q_hi = args.alpha, 1 - args.alpha
    q_model = get_quantile_model(args.quantile_model, (q_lo, q_hi))

    # 3. Run ITE interval estimation
    Z = np.column_stack([X_new, T, Y])

    Z1, Z2 = split_train_calibration(Z, seed=42, split_ratio=0.5)
    user_ids1, user_ids2 = split_train_calibration(user_ids, seed=42, split_ratio=0.5)

    print("[DEBUG] Starting ITE estimation")
    res = estimate_ite_interval(
        Z=Z,
        # Z1=Z1,
        # Z2=Z2,
        X_new=X_new,
        user_ids=user_ids2,
        mode=args.mode,
        alpha=args.alpha,
        quantile_model=q_model,
        q_lo=q_lo,
        q_hi=q_hi,
        T=T,
        Y=Y,
    )
    print("[DEBUG] res['y0_cf_lower'] sample:", res.get("y0_cf_lower")[:5] if res.get("y0_cf_lower") is not None else "None")
    print("[DEBUG] res['y1_cf_lower'] sample:", res.get("y1_cf_lower")[:5] if res.get("y1_cf_lower") is not None else "None")

    # 4. Separate factual and counterfactual outcome intervals
    is_treated = T == 1
    factual_y = Y

    # Initialize full-length arrays
    cf_y0_lower_full = np.full_like(factual_y, np.nan)
    cf_y0_upper_full = np.full_like(factual_y, np.nan)
    cf_y1_lower_full = np.full_like(factual_y, np.nan)
    cf_y1_upper_full = np.full_like(factual_y, np.nan)

    # Populate only relevant parts
    if res.get("y0_cf_lower") is not None:
        print("[DEBUG] assigning cf_y0 to treated group:", res["y0_cf_lower"][:5], res["y0_cf_upper"][:5])
        idx_user_ids2 = pd.Series(user_ids).isin(user_ids2)
        treated_in_userids2_idx = np.where((T == 1) & idx_user_ids2)[0]
        cf_y0_lower_full[treated_in_userids2_idx] = res["y0_cf_lower"]
        cf_y0_upper_full[treated_in_userids2_idx] = res["y0_cf_upper"]
    if res.get("y1_cf_lower") is not None:
        cf_y1_lower_full[~is_treated] = res["y1_cf_lower"]
        cf_y1_upper_full[~is_treated] = res["y1_cf_upper"]

    # Conservative ITE intervals
    ite_lower = np.where(
        is_treated, 
        factual_y - cf_y0_upper_full,  # Y1 - Y0_upper
        cf_y1_lower_full - factual_y   # Y1_lower - Y0
    )
    ite_upper = np.where(
        is_treated, 
        factual_y - cf_y0_lower_full,  # Y1 - Y0_lower
        cf_y1_upper_full - factual_y   # Y1_upper - Y0
    )

    # Save all individuals' ITE intervals
    output_df = pd.DataFrame({
        "user_id": user_ids,
        "treated": T,
        "factual_outcome": factual_y,
        "cf_y0_lower": cf_y0_lower_full,
        "cf_y0_upper": cf_y0_upper_full,
        "cf_y1_lower": cf_y1_lower_full,
        "cf_y1_upper": cf_y1_upper_full,
        "ite_lower": ite_lower,
        "ite_upper": ite_upper,
    })
    output_path = f"results/{args.mode}_ITE_interval_results.csv"
    print("[DEBUG] Sample output_df:")
    print(output_df.head())
    output_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved ITE interval results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cfcausal‑py ITE interval pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--mode", type=str, default="ATE",
                        choices=["ATE", "ATT", "ATC", "general"],
                        help="Weighting mode")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate")
    parser.add_argument("--quantile_model", type=str, default="qrf",
                        help="Quantile model to use (qgb only)")
    # parser.add_argument("--q_lo", type=float, default=0.1, help="Lower quantile")
    # parser.add_argument("--q_hi", type=float, default=0.9, help="Upper quantile")
    args = parser.parse_args()
    main(args)