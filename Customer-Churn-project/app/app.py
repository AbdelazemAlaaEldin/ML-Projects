import streamlit as st
import pandas as pd
import joblib


# =========================
# Load Model
# =========================

MODEL_PATH = "models/knn_churn_pipeline.pkl"

model = joblib.load(MODEL_PATH)


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# =========================
# Title
# =========================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn "
    "based on their service and account information."
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
# Create Input DataFrame
# =========================

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


# =========================
# Prediction
# =========================

if st.button("🔮 Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    # Classes are ["No", "Yes"]
    churn_probability = probability[1]

    st.subheader("Prediction Result")

    if prediction == "Yes":

        st.error("⚠️ Customer is predicted to Churn")

    else:

        st.success("✅ Customer is predicted to Stay")

    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )