# debug_show.py
import pandas as pd
import sys

p = "data/raw/Telco-Customer-Churn-ALT-clean.csv"

def try_csv(path, enc="utf-8"):
    try:
        df = pd.read_csv(path, engine='python', encoding=enc)
        return df, f"CSV (encoding={enc})"
    except Exception as e:
        return None, f"CSV read failed (encoding={enc}): {repr(e)}"

# Try CSV with a few encodings
encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
for e in encodings:
    df, msg = try_csv(p, enc=e)
    print(msg)
    if df is not None:
        print(" -> OK, rows=", len(df), " cols=", len(df.columns))
        print("Columns:", df.columns.tolist())
        print("\nHead (first 5 rows):")
        print(df.head(5).to_string(index=False))
        sys.exit(0)

# If CSV failed for all encodings, try Excel read
print("\nAll CSV attempts failed. Trying Excel read with openpyxl...")
try:
    df = pd.read_excel(p, engine="openpyxl")
    print("Excel read OK, rows=", len(df), "cols=", len(df.columns))
    print("Columns:", df.columns.tolist())
    print("\nHead (first 5 rows):")
    print(df.head(5).to_string(index=False))
    sys.exit(0)
except Exception as e:
    print("Excel read failed:", repr(e))
    print("\nFINAL: Could not read file. Please paste the raw first 30 lines using PowerShell:")
    print(r"Get-Content -Path data\raw\Telco_customer_churn.xlsx -TotalCount 30")
    sys.exit(2)
