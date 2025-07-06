"""
python main.py --data data/{} --mode {} --alpha {} --quantile_model {}
"""

import argparse
import numpy as np
import pandas as pd

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
    Z = load_dataset(args.data).values  # DataFrame -> ndarray
    X_new = Z[:, :-2]  # test on training features for demo

    # 2. Choose quantile model
    q_lo, q_hi = args.alpha, 1 - args.alpha
    q_model = get_quantile_model(args.quantile_model, (q_lo, q_hi))

    # 3. Run ITE interval estimation
    res = estimate_ite_interval(
        Z,
        X_new,
        mode=args.mode,
        alpha=args.alpha,
        quantile_model=q_model,
        q_lo=q_lo,
        q_hi=q_hi,
    )

    # 4. Print summary
    print("=== ITE Interval Summary ===")
    print(f"lower (first 5): {res['lower'][:5]}")
    print(f"upper (first 5): {res['upper'][:5]}")
    print(f"Avg length     : {np.mean(res['upper'] - res['lower']):.4f}")

    # 5. Save results
    output_df = pd.DataFrame({
        "lower": res["lower"],
        "upper": res["upper"],
        "length": res["upper"] - res["lower"]
    })
    output_path = f"results/{args.mode}_results.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")

    # Save summary CSV for comparison with literature
    coverage = np.nan
    if 'true' in res:
        coverage = np.mean((res['lower'] <= res['true']) & (res['true'] <= res['upper']))
    interval_length = np.mean(res["upper"] - res["lower"])
    summary_df = pd.DataFrame([{
        "method": f"{args.mode}_{args.quantile_model}",
        "cr": float(coverage) if not np.isnan(coverage) else np.nan,
        "len": interval_length,
        "alpha": args.alpha
    }])
    summary_path = f"results/{args.mode}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")


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