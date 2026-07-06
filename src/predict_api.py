import pandas as pd
import joblib
import os
from src.preprocessing import prepare_inference_features


def predict_single(model_path: str, df: pd.DataFrame):
    """
    Load model and artifacts, preprocess the input DataFrame, and make predictions.
    """
    model_dir = os.path.dirname(model_path)
    model = joblib.load(model_path)

    # Load pre-trained encoders
    encoders_path = os.path.join(model_dir, "encoders.joblib")
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
    else:
        encoders = {}

    # Load feature names expected by the model
    feature_path = os.path.join(model_dir, "feature_names.joblib")
    if os.path.exists(feature_path):
        model_features = joblib.load(feature_path)
    else:
        model_features = None

    # Preprocess and align data
    df_prepared = prepare_inference_features(df, encoders, model_features)

    # Run predictions
    probs = model.predict_proba(df_prepared)[:, 1]

    # Return original dataframe with prediction column appended
    df_output = df.copy()
    df_output['churn_probability'] = probs
    return df_output