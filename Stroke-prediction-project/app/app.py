from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent.parent / "models" / "stroke_bagging.joblib"

st.set_page_config(page_title="Stroke Risk Screening", page_icon="🩺")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model, threshold = artifact["model"], artifact["threshold"]

st.title("🧠Stroke Risk Screening Demo")
st.warning(
    "Educational demo only. This is a screening aid trained on a small public "
    "dataset, not a medical diagnosis. Do not use it for clinical decisions."
)

with st.form("patient"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        ever_married = st.selectbox("Ever married", ["Yes", "No"])
        work_type = st.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence = st.selectbox("Residence type", ["Urban", "Rural"])
    with col2:
        glucose = st.number_input(
            "Average glucose level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0
        )
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=27.0)
        bmi_unknown = st.checkbox("BMI unknown")
        smoking = st.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart disease")
    submitted = st.form_submit_button("Assess")

if submitted:
    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "hypertension": int(hypertension),
                "heart_disease": int(heart_disease),
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence,
                "avg_glucose_level": glucose,
                "bmi": np.nan if bmi_unknown else bmi,
                "smoking_status": smoking,
            }
        ]
    )
    score = float(model.predict_proba(row)[0, 1])

    if score >= threshold:
        st.error("Flagged: higher-risk profile")
    else:
        st.success("Not flagged")

    st.metric(
        "Model score",
        f"{score:.2f}",
        help="A ranking score, not a calibrated probability of stroke.",
    )
    st.caption(f"Decision threshold: {threshold:.3f}")

with st.expander("About this model"):
    st.markdown(
        """
- Bagging of shallow decision trees with class balancing, threshold chosen on
  out-of-fold training predictions (F2).
- Test performance (50 strokes in 1022 patients): recall about 0.74
  (95% CI 0.61-0.86), precision about 0.15 (0.11-0.19).
- The score is **not** a calibrated probability, so do not read it as "x% chance".
- The model relies mostly on **age**. In the test set it flagged nobody under 40
  (missing 5 strokes) and about 92% of non-stroke patients over 70.
- Strokes in younger patients without hypertension or heart disease are
  usually missed.
"""
    )