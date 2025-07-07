import pandas as pd

df = pd.read_csv("results/ATT_treated_counterfactual_results.csv")
print("avg of ite_lower :",(df["ite_lower"]).sum() / len(df), "\navg of ite_upper :",(df["ite_upper"]).sum() / len(df))