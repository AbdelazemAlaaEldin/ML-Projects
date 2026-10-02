import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "decision_tree_loan_pipeline.pkl"
META_PATH = PROJECT_ROOT / "models" / "app_meta.json"

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


model = load_model()
meta = load_meta()
num_defaults = meta["numeric_defaults"]
cat_options = meta["categorical_options"]
cat_defaults = meta["categorical_defaults"]

st.title("🏦Loan Approval Prediction")
st.write(
    "Predict whether a loan application is likely to be approved or rejected "
    "based on the applicant's information."
)
st.warning(
    "Educational project only. This is not a real credit decision and must not "
    "be used for real-world financial decisions."
)

st.subheader("Applicant Information")
c1, c2 = st.columns(2)


def select(container, label, key):
    options = cat_options[key]
    return container.selectbox(label, options, index=options.index(cat_defaults[key]))


with c1:
    gender = select(st, "Gender", "Gender")
    married = select(st, "Married", "Married")
    dependents = select(st, "Dependents", "Dependents")
with c2:
    education = select(st, "Education", "Education")
    self_employed = select(st, "Self Employed", "Self_Employed")
    property_area = select(st, "Property Area", "Property_Area")

st.subheader("Financial Information")
c3, c4 = st.columns(2)
with c3:
    applicant_income = st.number_input(
        "Applicant Income", min_value=0.0, value=float(num_defaults["ApplicantIncome"])
    )
    coapplicant_income = st.number_input(
        "Coapplicant Income", min_value=0.0, value=float(num_defaults["CoapplicantIncome"])
    )
with c4:
    loan_amount = st.number_input(
        "Loan Amount (dataset units)", min_value=0.0, value=float(num_defaults["LoanAmount"])
    )
    loan_amount_term = st.number_input(
        "Loan Amount Term (months)", min_value=0.0, value=float(num_defaults["Loan_Amount_Term"])
    )

credit_options = cat_options["Credit_History"]
credit_history = st.selectbox(
    "Credit History",
    credit_options,
    index=credit_options.index(cat_defaults["Credit_History"]),
    format_func=lambda v: "1.0 (meets guidelines)" if v == 1.0 else "0.0 (does not meet guidelines)",
)

st.caption("Default values are the training-set median (numeric) or most frequent value (categorical).")

if st.button("Predict Loan Status", use_container_width=True):
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }])

    prediction = model.predict(input_df)[0]
    classes = list(model.classes_)
    leaf_share = model.predict_proba(input_df)[0][classes.index("Y")]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == "Y":
        st.success("Prediction: Approved")
    else:
        st.error("Prediction: Rejected")

    st.metric("Leaf approval share", f"{leaf_share * 100:.2f}%")
    st.caption(
        "This is the share of training applicants in the same tree leaf who were "
        "approved. It is not a calibrated probability."
    )

    if prediction == "Y":
        st.caption(
            "Caution: the model cannot separate safe approvals from risky ones. "
            "Among predicted approvals, 81 of 388 (20.88%) were actually rejected in "
            "cross-validation on the training data, and 14 of 95 (14.74%) on the "
            "test set."
        )
    elif credit_history == 0.0:
        st.caption(
            "The model rejects applicants with Credit_History 0.0. On the test set, "
            "21 of 22 such applicants were actually rejected."
        )

with st.expander("About this model"):
    st.write(
        "Model: Decision Tree (gini, ccp_alpha=0.005, depth 6, 9 leaves) inside a "
        "pipeline with median and most-frequent imputation and one-hot encoding. "
        "Tuned with 5-fold cross-validation on the training set using recall_macro "
        "(CV recall_macro 0.6926). The test set was used once, after tuning."
    )
    st.write(
        "Held-out test set (123 applications, 38 rejected): "
        "accuracy 0.8537 (95% CI [0.7886, 0.9108]), "
        "rejected recall 0.6316 (95% CI [0.4827, 0.7941]), "
        "rejected precision 0.8571 (95% CI [0.7241, 0.9667]), "
        "ROC-AUC 0.7621 (95% CI [0.6608, 0.8597]). "
        "Confusion matrix: 24 rejected caught, 14 rejected missed (predicted "
        "approved), 4 approved wrongly rejected, 81 approved correct."
    )
    st.write(
        "Main limitation: Credit_History dominates the model. A one-split rule "
        "on Credit_History alone reached the same test accuracy (0.8537). Among "
        "test applicants with Credit_History 1.0, 17 of 94 were actually "
        "rejected and the model caught only 3 of them. With 38 rejected cases "
        "in the test set, the true rejected recall could be below 50%."
    )