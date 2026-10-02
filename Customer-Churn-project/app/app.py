from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "knn_churn_pipeline.pkl"


# =========================
# Load model
# =========================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)


# =========================
# Title
# =========================

st.title("📊Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn "
    "based on their service and account information."
)

st.warning(
    "Educational project only. This is a screening aid trained on a public "
    "dataset, not a guaranteed prediction of real customer behavior."
)


# =========================
# Customer Information
# =========================

st.subheader("Customer Information")


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=18.25,
    max_value=118.75,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)


# =========================
# Prediction
# =========================

if st.button("Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Classes are ["No", "Yes"]
    churn_probability = probability[1]

    st.subheader("Prediction Result")

    if prediction == "Yes":
        st.error("Customer is predicted to churn")
    else:
        st.success("Customer is predicted to stay")

    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    if tenure > 20 and prediction == "No":
        st.caption(
            "Note: in testing, the model's most common mistake was missing "
            "churn among longer-tenured customers like this one. Treat a "
            "'not flagged' result with some caution for this profile."
        )

with st.expander("About this model"):
    st.markdown(
        """
- K-Nearest Neighbors (k=30, Euclidean distance, uniform weighting) with
  scaling and one-hot encoding inside one pipeline; hyperparameters tuned
  jointly with 5-fold cross-validation.
- Test set (1,409 customers): accuracy about 0.79, recall about 0.55
  (95% CI 0.51-0.60), ROC-AUC about 0.83 (0.81-0.85).
- The model tends to catch short-tenure, higher-paying churners well, but
  often misses longer-tenured customers who still leave: missed churners
  had about 25 months of tenure on average, versus about 10 months for
  correctly caught churners.
- Both missed churners and false alarms are concentrated among
  month-to-month contracts, the most volatile customer segment overall.
"""
    )