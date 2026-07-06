import pandas as pd

path = "data/raw/Telco-Customer-Churn-ALT-clean.csv"
delims = [",", ";", "\t", "|"]
encodings = ["utf-8", "latin1", "utf-8-sig"]

def try_read(sep, enc):
    try:
        df = pd.read_csv(path, sep=sep, encoding=enc, nrows=5)
        full = pd.read_csv(path, sep=sep, encoding=enc)
        print(f"SEP='{sep}' ENC='{enc}' -> rows={len(full)}, cols={len(full.columns)}")
        print("Columns:", full.columns.tolist())
        print("Sample head:\n", full.head(2).to_string(index=False))
        return True
    except Exception as e:
        print(f"SEP='{sep}' ENC='{enc}' -> FAILED with: {repr(e)}")
        return False

for e in encodings:
    for d in delims:
        ok = try_read(d, e)
        if ok:
            print("-" * 60)
