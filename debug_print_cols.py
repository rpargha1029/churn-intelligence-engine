import pandas as pd
p = "data/raw/Telco-Customer-Churn-ALT-clean.csv"
df = pd.read_csv(p)
print("Loaded file:", p)
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))
