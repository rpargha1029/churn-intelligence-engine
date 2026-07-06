# 📊 Customer Churn Intelligence Engine

An end-to-end Machine Learning pipeline and Streamlit application to predict customer churn using XGBoost. The engine includes robust data preprocessing, feature engineering, and inference alignment.

---

## 📁 Repository Structure

```text
├── app/
│   └── streamlit_app.py        # Streamlit web application interface
├── data/
│   ├── raw/                    # Raw Telco customer datasets
│   │   └── Telco-Customer-Churn-ALT-clean.csv
│   └── test_customers_50.csv   # Sample customer file for testing inference
├── models/                     # Saved model artifacts
│   ├── xgb_model.joblib        # Trained XGBoost model
│   ├── encoders.joblib         # Pre-trained LabelEncoder objects
│   └── feature_names.joblib    # Feature list matching the model schema
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory Data Analysis notebook (placeholder)
├── src/
│   ├── __init__.py
│   ├── config.py               # Path configurations
│   ├── data_loader.py          # Clean data reading functions
│   ├── evaluate.py             # Model evaluation utilities (ROC AUC, Confusion Matrix)
│   ├── explain.py              # SHAP model explainability utilities
│   ├── features.py             # Feature engineering functions
│   ├── predict_api.py          # API entry point for programmatic inference
│   ├── preprocessing.py        # Cleaning, standardizing, and categorical encoding
│   ├── train.py                # Main training script (XGBoost + GPU Auto-detection)
│   └── utils.py                # Model saving & loading helper utilities
├── environment.yml             # Conda environment definition
├── requirements.txt            # Pip packages list
└── README.md                   # Project documentation
```

---

## ⚙️ Setup and Installation

### Option 1: Conda Environment (Recommended)

1. Clone this repository.
2. Create the environment from `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the new environment:
   ```bash
   conda activate churnengine
   ```

### Option 2: Pip Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Training the Model

To run the training pipeline, prepare features, train XGBoost (with automatic GPU fallback if CUDA is unavailable), evaluate the results, and save the artifacts under `models/`:

```bash
python src/train.py --data_path data/raw/Telco-Customer-Churn-ALT-clean.csv
```

### 2. Running the Streamlit App

Launch the interactive web application to upload a CSV file and view predictions:

```bash
streamlit run app/streamlit_app.py
```

### 3. Programmatic Inference

Use the `predict_single` function in `src/predict_api.py` to get predictions for any pandas DataFrame:

```python
import pandas as pd
from src.predict_api import predict_single

# Load target data
df_test = pd.read_csv("data/test_customers_50.csv")

# Generate predictions
df_results = predict_single("models/xgb_model.joblib", df_test)
print(df_results[["CustomerID", "churn_probability"]].head())
```

---

## 🛠️ Highlights and Key Logic

* **Pipeline Consistency**: The system uses `prepare_inference_features` during inference to align columns, handle missing features, and transform labels using pre-trained encoders. Unseen categories are mapped to known classes dynamically.
* **GPU Auto-Detection**: The model training automatically senses if a CUDA-enabled GPU is available. If not, it falls back to standard CPU histogram estimation (`tree_method='hist'`) without crashing.
* **Tenure Bucket Coercion**: A robust binning process transforms tenure into groups and includes a boundary guard (`include_lowest=True`) to handle new customers (`tenure = 0`) correctly.