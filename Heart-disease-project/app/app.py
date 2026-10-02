from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_logistic_pipeline.pkl"


# =========================
# Load model
# =========================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================
# Page configuration
# =========================

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)


# =========================
# Title
# =========================

st.title("🫀Heart Disease Prediction")

st.write(
    "Enter the patient's clinical features below "
    "to generate a prediction using the trained Logistic Regression model."
)

st.warning(
    "Educational project only. This is not a medical diagnostic tool and "
    "must not be used for clinical decisions."
)


# =========================
# Numerical Features
# =========================

st.subheader("Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol (chol)",
    min_value=50,
    max_value=700,
    value=200
)

thalach = st.number_input(
    "Maximum Heart Rate (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

oldpeak = st.number_input(
    "ST Depression (oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


# =========================
# Categorical Features
# =========================

st.subheader("Clinical Features")

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3]
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    options=[0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
)

restecg = st.selectbox(
    "Resting ECG (restecg)",
    options=[0, 1, 2]
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    options=[0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
)

slope = st.selectbox(
    "Slope (slope)",
    options=[0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels (ca)",
    options=[0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thalassemia (thal)",
    options=[0, 1, 2, 3]
)


# =========================
# Prediction
# =========================

if st.button("Predict", use_container_width=True):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")

    st.metric(
        "Predicted Probability",
        f"{probability:.2%}"
    )

    if exang == 0 and thal == 2:
        st.caption(
            "Note: this combination of features (exang=0, thal=2) is where the "
            "model made its most confident mistakes during testing. Treat this "
            "prediction with extra caution."
        )

with st.expander("About this model"):
    st.markdown(
        """
- Logistic Regression with scaling and one-hot encoding inside one pipeline;
  the regularization strength (C) was tuned with 5-fold cross-validation on
  the training set only.
- Test set (61 patients): accuracy about 0.87 (95% CI 0.79-0.95), ROC-AUC
  about 0.90 (0.81-0.97), recall about 0.91 (0.80-1.00).
- The model's most confident mistakes on the test set all shared the same
  combination of features: no exercise-induced angina (exang=0) and
  thal=2. This may reflect a blind spot rather than general noise.
- This dataset's `ca` and `thal` columns contain a few values outside their
  originally documented ranges, most likely remapped missing-value markers
  from the source data.
"""
    )
    