from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "breast_cancer_svm_pipeline.pkl"
)


# =========================
# Load model
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# Page configuration
# =========================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================
# Title
# =========================

st.title("🩺 Breast Cancer Prediction")

st.write(
    "This application uses a Machine Learning model "
    "to classify a breast cancer sample as Malignant or Benign."
)

st.warning(
    "Educational project only. This prediction is not a medical diagnosis."
)


# =========================
# Input section
# =========================

st.subheader("Enter Patient Features")

st.write(
    "Enter the 30 features used by the trained Machine Learning model."
)


# =========================
# Feature groups
# =========================

mean_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension"
]

error_features = [
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error"
]

worst_features = [
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# =========================
# Create input dictionary
# =========================

input_data = {}


# =========================
# Mean features
# =========================

st.markdown("### Mean Features")

col1, col2 = st.columns(2)

for i, feature in enumerate(mean_features):

    if i % 2 == 0:
        with col1:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )
    else:
        with col2:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )


# =========================
# Error features
# =========================

st.markdown("### Standard Error Features")

col1, col2 = st.columns(2)

for i, feature in enumerate(error_features):

    if i % 2 == 0:
        with col1:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )
    else:
        with col2:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )


# =========================
# Worst features
# =========================

st.markdown("### Worst Features")

col1, col2 = st.columns(2)

for i, feature in enumerate(worst_features):

    if i % 2 == 0:
        with col1:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )
    else:
        with col2:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )


# =========================
# Prediction
# =========================

if st.button("🔮 Predict", use_container_width=True):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    decision_score = model.decision_function(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 0:

        st.error("⚠️ Prediction: Malignant")

    else:

        st.success("✅ Prediction: Benign")

    st.metric(
        "Decision Score",
        f"{decision_score:.4f}"
    )