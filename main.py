"""
python main.py --data data/{} --mode {} --alpha {} --quantile_model {}
"""

import argparse
import numpy as np
import pandas as pd

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

    T = df["treated"].values
    Y = df["ldl_value_after_trt"].values

    # Remove rows with missing Y for treated
    valid_idx = ~((T == 1) & np.isnan(Y))
    X_new = X_new[valid_idx]
    T = T[valid_idx]
    Y = Y[valid_idx]
    user_ids = user_ids[valid_idx]

    # Remove any remaining NaNs in Y (regardless of T)
    nan_mask = ~np.isnan(Y)
    X_new = X_new[nan_mask]
    T = T[nan_mask]
    Y = Y[nan_mask]
    user_ids = user_ids[nan_mask]

    # 2. Choose quantile model
    q_lo, q_hi = args.alpha, 1 - args.alpha
    q_model = get_quantile_model(args.quantile_model, (q_lo, q_hi))

    # 3. Run ITE interval estimation
    Z = np.column_stack([X_new, T, Y])
    res = estimate_ite_interval(
        Z=Z,
        X_new=X_new,
        user_ids=user_ids,
        mode=args.mode,
        alpha=args.alpha,
        quantile_model=q_model,
        q_lo=q_lo,
        q_hi=q_hi,
    )

    # 4. Filter treated individuals
    treated_mask = T == 1
    user_ids_treated = user_ids[treated_mask]
    Y1 = Y[treated_mask]

    # 5. Compute counterfactual (Y0) interval from result
    Y0_lower = res["y0_cf_lower"][treated_mask]
    Y0_upper = res["y0_cf_upper"][treated_mask]
    ITE_lower = Y1 - Y0_upper  # conservative lower bound
    ITE_upper = Y1 - Y0_lower  # conservative upper bound

    # 6. Save treated individuals’ ITE intervals
    output_df = pd.DataFrame({
        "user_id": user_ids_treated,
        "treated": 1,
        "factual_y1": Y1,
        "cf_y0_lower": Y0_lower,
        "cf_y0_upper": Y0_upper,
        "ite_lower": ITE_lower,
        "ite_upper": ITE_upper,
    })
    output_path = f"results/{args.mode}_treated_counterfactual_results.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved treated-counterfactual results to: {output_path}")


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