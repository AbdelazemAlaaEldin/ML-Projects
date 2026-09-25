import os

import joblib
import pandas as pd
import streamlit as st


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# =========================
# Load Model
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "heart_disease_logistic_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)


# =========================
# Title
# =========================

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's clinical features below "
    "to generate a prediction using the trained Logistic Regression model."
)

st.warning(
    "This application is for educational purposes only "
    "and is not a medical diagnostic tool."
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
# Create Input DataFrame
# =========================

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


# =========================
# Prediction
# =========================

if st.button("Predict", use_container_width=True):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")

    st.metric(
        "Predicted Probability",
        f"{probability:.2%}"
    )