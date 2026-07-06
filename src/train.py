import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib

from src.preprocessing import clean_and_basic_process, encode_categoricals
from src.features import create_features
from src.utils import save_model


# Canonical categorical features we encoded in preprocessing
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def _drop_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are identifiers or large free-text which are not useful
    for the churn model.
    """
    drop_candidates = [
        "CustomerID", "Customer Id", "Customer Id ",
        "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Reason", "Churn Reason ",
        "ChurnScore", "Churn Score", "Churn Value", "Churn Value ",
        "CLTV"
    ]
    for c in drop_candidates:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


def _prepare_features(df: pd.DataFrame, encoders: dict):
    """
    Returns X (numeric dataframe) and y (Series).
    """
    if "Churn" not in df.columns:
        raise KeyError(f"Churn column not found. Available columns: {list(df.columns)}")

    # 1. Remove text columns
    df = _drop_text_columns(df)

    # 2. One-hot encode tenure bucket if exists
    if "tenure_bucket" in df.columns:
        dummies = pd.get_dummies(df["tenure_bucket"].astype(str), prefix="tenure")
        df = pd.concat([df.drop(columns=["tenure_bucket"]), dummies], axis=1)

    # 3. Guarantee canonical categorical columns are numeric
    for c in CANONICAL_CATEGORICALS:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.factorize(df[c].astype(str))[0]

    # 4. Split target
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # 5. Remove leftover objects
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"Warning: dropping non-numeric columns: {obj_cols}")
        X = X.drop(columns=obj_cols)

    # 6. Fill missing numeric values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # 7. Ensure numeric only
    if any(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("Non-numeric columns remain.")

    return X, y


def main():
    args = parse_args()

    # SAFE READ CSV
    try:
        df = pd.read_csv(args.data_path, encoding="utf-8")
    except:
        df = pd.read_csv(args.data_path, encoding="latin1")

    # Clean, engineer features, and encode categoricals
    df = clean_and_basic_process(df)
    df = create_features(df)
    df, encoders = encode_categoricals(df)

    # Create features (dropping ID and text columns, one-hot encoding tenure buckets)
    X, y = _prepare_features(df, encoders)

    # Save **feature names for inference alignment**
    os.makedirs(args.out_dir, exist_ok=True)
    feature_path = os.path.join(args.out_dir, "feature_names.joblib")
    joblib.dump(list(X.columns), feature_path)
    print(f"Saved feature list: {feature_path}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # XGBoost training with GPU fallback
    tree_method = "hist"
    predictor = "auto"
    try:
        # Check if GPU is available by trying to fit a trivial model
        xgb.XGBClassifier(tree_method="gpu_hist").fit(np.array([[1]]), np.array([1]))
        tree_method = "gpu_hist"
        predictor = "gpu_predictor"
        print("Using GPU for training...")
    except Exception:
        print("GPU not available. Using CPU ('hist' tree method)...")

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method=tree_method,
        predictor=predictor,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=50,
    )

    preds = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, prob))

    # Save model + encoders
    save_model(model, os.path.join(args.out_dir, "xgb_model.joblib"))
    joblib.dump(encoders, os.path.join(args.out_dir, "encoders.joblib"))

    print(f"Model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()