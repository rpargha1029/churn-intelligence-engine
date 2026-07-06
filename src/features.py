import pandas as pd

def _normalize_col(df, candidates):
    """
    Given a list of candidate column names (variants), return the first match in the df.
    """
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in cols:
            return cols[key]
    return None

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Column variants for InternetService
    internet_col = _normalize_col(df, ["InternetService", "Internet Service"])
    if internet_col is None:
        raise KeyError(f"InternetService column missing. Available: {list(df.columns)}")

    # Create is_fiber feature
    df["is_fiber"] = df[internet_col].astype(str).str.contains("Fiber", case=False, na=False).astype(int)

    # Tenure column variants
    tenure_col = _normalize_col(df, ["tenure", "Tenure Months"])
    if tenure_col is None:
        raise KeyError(f"Tenure column missing. Available: {list(df.columns)}")

    # Create tenure buckets
    df["tenure_bucket"] = pd.cut(
        df[tenure_col].astype(float),
        bins=[0, 6, 12, 24, 48, 72, float("inf")],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72", "72+"],
        include_lowest=True
    )

    return df
