import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional

# Canonical categorical names we want in the final dataframe
CANONICAL_CATEGORICALS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# For each canonical column, a list of candidate variants that might appear in datasets
COLUMN_VARIANTS = {
    "gender": ["gender", "Gender"],
    "SeniorCitizen": ["SeniorCitizen", "Senior Citizen", "senior citizen", "seniorcitizen"],
    "Partner": ["Partner", "partner"],
    "Dependents": ["Dependents", "dependents"],
    "PhoneService": ["PhoneService", "Phone Service", "Phone_Service"],
    "MultipleLines": ["MultipleLines", "Multiple Lines", "Multiple_Lines"],
    "InternetService": ["InternetService", "Internet Service", "Internet_Service"],
    "OnlineSecurity": ["OnlineSecurity", "Online Security", "Online_Security"],
    "OnlineBackup": ["OnlineBackup", "Online Backup", "Online_Backup"],
    "DeviceProtection": ["DeviceProtection", "Device Protection", "Device_Protection"],
    "TechSupport": ["TechSupport", "Tech Support", "Tech_Support"],
    "StreamingTV": ["StreamingTV", "Streaming TV", "Streaming_TV"],
    "StreamingMovies": ["StreamingMovies", "Streaming Movies", "Streaming_Movies"],
    "Contract": ["Contract", "contract"],
    "PaperlessBilling": ["PaperlessBilling", "Paperless Billing", "Paperless_Billing"],
    "PaymentMethod": ["PaymentMethod", "Payment Method", "Payment_Method"],
    # numeric / target variants are handled separately below
}

# Candidate lists for numeric/target columns
MONTHLY_CANDIDATES = [
    "MonthlyCharges",
    "Monthly Charges",
    "Monthly_Charges",
    "monthlycharges",
    "monthly charge",
    "monthly charge (usd)",
    "Monthly Charge",
    "Monthly charges",
]
TENURE_CANDIDATES = [
    "tenure",
    "Tenure",
    "Tenure Months",
    "Tenure_Months",
    "tenure_months",
    "tenure months",
    "Tenure Months",
    "TenureMonths",
]
TOTAL_CANDIDATES = [
    "TotalCharges",
    "Total Charges",
    "Total_Charges",
    "totalcharges",
    "total_charges",
    "Total Charges (USD)",
]
CHURN_CANDIDATES = [
    "Churn",
    "Churn Label",
    "Churn_Label",
    "Churn Value",
    "Churn_Value",
    "churn",
    "ChurnFlag",
    "churn_label",
]


def _normalize_header_name(name: str) -> str:
    """Return normalized key for a header (lower, no spaces/underscores/hyphens)."""
    if not isinstance(name, str):
        name = str(name)
    # remove BOM if present
    name = name.encode("utf-8").decode("utf-8-sig")
    n = name.strip()
    n = n.replace("-", " ")
    n = " ".join(n.split())  # collapse whitespace
    return n.lower().replace(" ", "_")


def _build_normalized_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping from normalized_key -> original_pretty_column_name (post-strip).
    The pretty name is the cleaned (stripped, collapsed) column string as it will appear in df.columns.
    """
    mapping = {}
    pretty_cols = []
    for c in df.columns:
        c_str = str(c)
        # clean up pretty name
        c_clean = c_str.encode("utf-8").decode("utf-8-sig").strip()
        c_clean = c_clean.replace("-", " ")
        c_clean = " ".join(c_clean.split())
        pretty_cols.append(c_clean)
        mapping[_normalize_header_name(c_clean)] = c_clean
    # set df.columns to pretty cleaned names
    df.columns = pretty_cols
    return mapping


def _find_first(mapping: Dict[str, str], candidates: list) -> Optional[str]:
    """
    Given mapping normalized_key -> pretty_name, and a list of candidate human names,
    return the first pretty_name found in the mapping or None.
    """
    for cand in candidates:
        key = _normalize_header_name(cand)
        if key in mapping:
            return mapping[key]
    # fuzzy fallback: check if any mapping key contains candidate token
    for cand in candidates:
        token = "".join(str(cand).lower().split()).replace("_", "")
        for k, pretty in mapping.items():
            if token in k:
                return pretty
    return None


def clean_and_basic_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust cleaning + canonicalization:
      - normalize header names (strip, remove BOM, collapse spaces)
      - detect and rename common variants to canonical names required by the pipeline
      - ensure numeric columns exist and coerce types
      - fill missing TotalCharges using MonthlyCharges * tenure
      - standardize SeniorCitizen (0/1 -> 'Yes'/'No') when possible
      - trim whitespace for object columns
    """
    df = df.copy()

    # Normalize headers & build mapping
    mapping = _build_normalized_map(df)

    # Find numeric / target columns
    monthly_col = _find_first(mapping, MONTHLY_CANDIDATES)
    tenure_col = _find_first(mapping, TENURE_CANDIDATES)
    total_col = _find_first(mapping, TOTAL_CANDIDATES)
    churn_col = _find_first(mapping, CHURN_CANDIDATES)

    # If required columns are missing, raise a helpful error listing available columns
    missing = []
    if monthly_col is None:
        missing.append("MonthlyCharges (variants)")
    if tenure_col is None:
        missing.append("tenure (variants)")
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. Available columns: {list(df.columns)}"
        )

    # Rename detected numeric/target columns to canonical names
    rename_map = {}
    rename_map[monthly_col] = "MonthlyCharges"
    rename_map[tenure_col] = "tenure"
    if total_col is not None:
        rename_map[total_col] = "TotalCharges"
    if churn_col is not None:
        rename_map[churn_col] = "Churn"
    df = df.rename(columns=rename_map)

    # Also map canonical categorical variants (if present) to canonical names
    for canonical, variants in COLUMN_VARIANTS.items():
        found = _find_first(mapping, variants)
        if found is not None:
            # only rename if that pretty column exists (it should, mapping was built)
            if found in df.columns and canonical not in df.columns:
                df = df.rename(columns={found: canonical})

    # Ensure numeric types for monthly and tenure
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    # If TotalCharges missing, create safe fallback
    if "TotalCharges" not in df.columns:
        df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    else:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        mask = df["TotalCharges"].isna()
        if mask.any():
            df.loc[mask, "TotalCharges"] = (
                df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
            )

    # Standardize SeniorCitizen if present (0/1 -> Yes/No)
    if "SeniorCitizen" in df.columns:
        try:
            df["SeniorCitizen"] = df["SeniorCitizen"].apply(
                lambda x: "Yes" if int(float(x)) == 1 else "No"
            )
        except Exception:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(str).str.strip()

    # If Churn was detected and renamed, normalize values ('Yes'/'No')
    if "Churn" in df.columns:
        # common churn values: Yes/No, 1/0, True/False, 'Churned'/'Stayed'
        def normalize_churn(v):
            if pd.isna(v):
                return v
            s = str(v).strip().lower()
            if s in ("yes", "y", "true", "1", "1.0", "churned"):
                return "Yes"
            if s in ("no", "n", "false", "0", "0.0", "stayed"):
                return "No"
            return s.capitalize()

        df["Churn"] = df["Churn"].apply(normalize_churn)

    # Trim whitespace on object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def encode_categoricals(
    df: pd.DataFrame, pre_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode canonical categorical columns defined in CANONICAL_CATEGORICALS.
    If pre_encoders is provided, use them to transform. Otherwise fit and transform.
    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in CANONICAL_CATEGORICALS:
        if col in df.columns:
            if pre_encoders and col in pre_encoders:
                le = pre_encoders[col]
                # Map unseen classes to the first known class in the encoder to avoid crashing
                classes = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in classes else list(classes)[0])
                df[col] = le.transform(df[col])
                encoders[col] = le
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    # Map the target column if present
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        # If mapping produces NaN (unrecognized labels), try coercion to numeric
        if df["Churn"].isna().any():
            # attempt to coerce numeric-like values
            df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(df["Churn"])

    return df, encoders


def prepare_inference_features(df: pd.DataFrame, encoders: Dict[str, LabelEncoder], model_features: list) -> pd.DataFrame:
    """
    Prepare features for inference, aligning with the trained model columns.
    1. Clean and basic process.
    2. Extract engineered features (is_fiber, tenure_bucket).
    3. Categorical encoding using pre-trained encoders.
    4. One-hot encode tenure_bucket if it exists.
    5. Drop text and non-numeric columns.
    6. Align with model feature order and handle missing features.
    """
    from src.features import create_features

    # 1. Clean and basic process
    df = clean_and_basic_process(df)

    # 2. Extract engineered features
    try:
        df = create_features(df)
    except Exception:
        pass

    # 3. Categorical encoding using pre-trained encoders
    df, _ = encode_categoricals(df, pre_encoders=encoders)

    # 4. Drop standard text columns
    drop_candidates = [
        "CustomerID", "Customer Id", "Customer Id ",
        "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Reason", "Churn Reason ",
        "ChurnScore", "Churn Score", "Churn Value", "Churn Value ",
        "CLTV", "Churn"
    ]
    for c in drop_candidates:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 5. One-hot encode tenure bucket if it exists
    if "tenure_bucket" in df.columns:
        dummies = pd.get_dummies(df["tenure_bucket"].astype(str), prefix="tenure")
        df = pd.concat([df.drop(columns=["tenure_bucket"]), dummies], axis=1)

    # 6. Keep only numeric columns
    df = df.select_dtypes(include=["number"])

    # 7. Fill missing numeric values
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    # 8. Align EXACTLY to model training features (handling missing columns)
    aligned = pd.DataFrame()
    for col in model_features:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = 0  # Fill 0 for missing feature columns (e.g., missing one-hot dummies)

    return aligned[model_features]
