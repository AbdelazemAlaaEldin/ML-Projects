import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_svm_pipeline.pkl"
META_PATH = PROJECT_ROOT / "models" / "app_meta.json"

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


model = load_model()
meta = load_meta()
defaults = meta["feature_defaults"]
margin = meta["uncertainty_margin"]

st.title("🩺Breast Cancer Prediction")
st.write(
    "Classify a breast tumor sample as Malignant or Benign from 30 measurements "
    "computed from a digitized image of a fine needle aspirate."
)
st.warning("Educational project only. This prediction is not a medical diagnosis.")

groups = {
    "Mean Features": [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
    ],
    "Standard Error Features": [
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
    ],
    "Worst Features": [
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension",
    ],
}

st.subheader("Enter Sample Features")
st.write("Default values are the median of the training data for each feature.")

input_data = {}
tabs = st.tabs(list(groups.keys()))
for tab, (group_name, features) in zip(tabs, groups.items()):
    with tab:
        col1, col2 = st.columns(2)
        for i, feature in enumerate(features):
            with (col1 if i % 2 == 0 else col2):
                input_data[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=float(defaults[feature]),
                    format="%.6f",
                )

if st.button("Predict", use_container_width=True):
    input_df = pd.DataFrame([input_data])[list(defaults.keys())]

    prediction = model.predict(input_df)[0]
    decision_score = model.decision_function(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 0:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")

    st.metric("Decision Score", f"{decision_score:.4f}")
    st.caption(
        "A negative score favors Malignant and a positive score favors Benign. "
        "The magnitude reflects distance from the decision boundary."
    )

    if abs(decision_score) < margin:
        st.caption(
            f"This sample is close to the decision boundary (|score| < {margin}). "
            "In cross-validation on the training data, most of the model's "
            "mistakes fell in this zone, so treat this prediction as uncertain."
        )

with st.expander("About this model"):
    st.write(
        "Model: StandardScaler + SVC (RBF kernel, C=10, gamma=0.01), tuned with "
        "5-fold cross-validation on the training set using recall_macro. "
        "The test set was used once, after tuning."
    )
    st.write(
        "Held-out test set (114 samples, 42 malignant): "
        "accuracy 0.9825 (95% CI [0.9561, 1.0000]), "
        "malignant recall 0.9762 (95% CI [0.9189, 1.0000]), "
        "malignant precision 0.9762 (95% CI [0.9200, 1.0000]), "
        "ROC-AUC 0.9977 (95% CI [0.9910, 1.0000]). "
        "Confusion matrix: 41 malignant caught, 1 malignant missed, "
        "1 benign flagged as malignant, 71 benign correct."
    )
    st.write(
        "Main limitation: the test set has only 42 malignant cases, so the "
        "true malignant recall could be near 92%. The uncertainty zone is a "
        "warning, not a guarantee: one of the two test errors (a missed "
        "malignant case at score +0.9144) fell outside the 0.75 zone."
    )