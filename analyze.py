import pandas as pd

df = pd.read_csv("results/ATT_ITE_interval_results.csv")
print("avg of ite_lower :", (df["ite_lower"]).sum() / len(df), 
      "\navg of ite_upper :", (df["ite_upper"]).sum() / len(df))

print("negative ite_lower count:", (df["ite_lower"] < 0).sum())
print("negative ite_upper count:", (df["ite_upper"] < 0).sum())
print("both lower and upper negative count:", ((df["ite_lower"] < 0) & (df["ite_upper"] < 0)).sum())
print("valid ITE interval count:", ((~df["ite_lower"].isna()) & (~df["ite_upper"].isna())).sum())