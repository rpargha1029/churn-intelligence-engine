import streamlit as st
import pandas as pd
import joblib
import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.preprocessing import prepare_inference_features

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Intelligence Engine")
st.write("Upload a CSV file with customer records, and the model will predict churn probability.")

# --------------------------------------------------
# LOAD MODEL + ARTIFACTS
# --------------------------------------------------
MODEL_PATH = os.path.join(ROOT_DIR, "models", "xgb_model.joblib")
ENC_PATH = os.path.join(ROOT_DIR, "models", "encoders.joblib")
FEATURE_PATH = os.path.join(ROOT_DIR, "models", "feature_names.joblib")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    
    try:
        encoders = joblib.load(ENC_PATH)
    except Exception:
        encoders = {}
        
    try:
        model_features = joblib.load(FEATURE_PATH)
    except Exception:
        model_features = None
        st.warning("⚠ Model feature_names not found — aligning using numeric columns only.")
        
    return model, encoders, model_features

try:
    model, encoders, model_features = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {str(e)}")
    st.stop()

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded:
    # STEP 1: Read CSV safely
    try:
        df = pd.read_csv(uploaded, encoding="utf-8")
    except Exception:
        df = pd.read_csv(uploaded, encoding="latin1")

    st.subheader("📁 Raw Input Data (first 10 rows)")
    st.dataframe(df.head(10))

    # Debug counts
    st.write("🔍 Rows in upload:", len(df))

    # STEP 2: Preprocess and Align Features
    with st.spinner("Processing input data..."):
        try:
            df_aligned = prepare_inference_features(df, encoders, model_features)
            st.success("✅ Data preprocessing and feature alignment successful!")
            st.write("🔍 Features count for prediction:", df_aligned.shape[1])
            st.write("🔍 Rows processed for prediction:", len(df_aligned))
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            st.stop()

    # STEP 3: Prediction
    with st.spinner("Running prediction model..."):
        try:
            preds = model.predict_proba(df_aligned)[:, 1]
            df_output = df.copy()
            df_output["churn_probability"] = preds
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.stop()

    st.subheader("📈 Predictions")
    st.dataframe(df_output)

    # STEP 4: Downloadable results
    csv = df_output.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Predictions", data=csv, file_name="churn_predictions.csv")
