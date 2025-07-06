import pandas as pd

data = pd.read_csv("data/NLSM_data.csv")
true_vals = data["Y"]  

def load_and_compute(method_name, file_prefix):
    pred_df = pd.read_csv(f"results/{file_prefix}_results.csv")  # 예: ATE_results.csv
    print(f"[INFO] Columns in {file_prefix}_results.csv: {pred_df.columns.tolist()}")
    print(pred_df.head())

    pred_df["true"] = true_vals[:len(pred_df)].reset_index(drop=True)

    cr = ((pred_df["lower"] <= pred_df["true"]) & (pred_df["true"] <= pred_df["upper"])).mean()

    summary_df = pd.read_csv(f"results/{file_prefix}_summary.csv").rename(
        columns={"len": "len_py", "alpha": "alpha_py"}
    )
    summary_df["cr_py"] = cr
    summary_df["method"] = method_name
    return summary_df


ate = load_and_compute("ate", "ATE")
att = load_and_compute("att", "ATT")
atc = load_and_compute("atc", "ATC")
py_summary = pd.concat([ate, att, atc], ignore_index=True)

r_summary = pd.read_csv("results/baseline_results_marginal.csv").rename(
    columns={"cr": "cr_r", "len": "len_r", "alpha": "alpha_r"}
)

py_summary["method"] = py_summary["method"].str.lower().str.strip()
r_summary["method"] = r_summary["method"].str.lower().str.strip()
py_to_r = {
    "ate": "cqr_exact_quantboosting",
    "att": "cqr_inexact_quantboosting",
    "atc": "cqr_naive_quantboosting"
}
py_summary["r_method"] = py_summary["method"].map(py_to_r)

merged = pd.merge(py_summary, r_summary, left_on="r_method", right_on="method", suffixes=["_py", "_r"])
merged["cr_diff"] = merged["cr_py"] - merged["cr_r"]
merged["len_diff"] = merged["len_py"] - merged["len_r"]

final = merged[[
    "r_method", "cr_py", "len_py", "alpha_py",
    "cr_r", "len_r", "alpha_r",
    "cr_diff", "len_diff"
]].rename(columns={"r_method": "method"})

final.to_csv("results/comparison_summary.csv", index=False)
print(final)