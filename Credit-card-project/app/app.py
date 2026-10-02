import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # the saved pipeline imports src.features
MODEL_PATH = ROOT / "models" / "credit_default_voting.joblib"

st.set_page_config(page_title="Credit Card Default Screening", page_icon="💳")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]

SEX = {1: "Male", 2: "Female"}
EDUCATION = {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}
MARRIAGE = {1: "Married", 2: "Single", 3: "Others"}
PAY_OPTIONS = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def pay_label(v):
    if v == -2:
        return "-2: No consumption"
    if v == -1:
        return "-1: Paid on time"
    if v == 0:
        return "0: No delay (revolving credit)"
    return f"{v}: {v} month(s) late"


def month_label(i):
    suffix = " (most recent)" if i == 1 else " (oldest)" if i == 6 else ""
    return f"Month {i}{suffix}"


st.title("💳Credit Card Default Screening Demo")
st.warning(
    "Educational demo only. A screening aid trained on a small public dataset "
    "(Taiwan, 2005), not a credit decision tool."
)

with st.form("client"):
    st.subheader("Client")
    c1, c2 = st.columns(2)
    with c1:
        limit = st.number_input(
            "Credit limit (NT$)", min_value=10000.0, max_value=1000000.0,
            value=140000.0, step=10000.0, key="limit",
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=34, key="age")
        sex = st.selectbox("Sex", list(SEX), format_func=SEX.get, key="sex")
    with c2:
        education = st.selectbox(
            "Education", list(EDUCATION), index=1, format_func=EDUCATION.get, key="edu"
        )
        marriage = st.selectbox(
            "Marital status", list(MARRIAGE), index=1, format_func=MARRIAGE.get, key="mar"
        )

    st.subheader("Last 6 months")
    tab_pay, tab_bill, tab_amt = st.tabs(
        ["Repayment status", "Bill amounts (NT$)", "Payment amounts (NT$)"]
    )
    pay, bill, amt = {}, {}, {}
    with tab_pay:
        for i in range(1, 7):
            pay[i] = st.selectbox(
                month_label(i), PAY_OPTIONS, index=PAY_OPTIONS.index(0),
                format_func=pay_label, key=f"pay_{i}",
            )
    with tab_bill:
        for i in range(1, 7):
            bill[i] = st.number_input(
                month_label(i), min_value=-400000.0, max_value=2000000.0,
                value=20000.0, step=1000.0, key=f"bill_{i}",
            )
    with tab_amt:
        for i in range(1, 7):
            amt[i] = st.number_input(
                month_label(i), min_value=0.0, max_value=2000000.0,
                value=2000.0, step=500.0, key=f"amt_{i}",
            )

    mode = st.radio(
        "Operating point",
        ["Balanced (F1)", "High recall (F2)"],
        horizontal=True,
        key="mode",
        help="High recall catches more defaulters but flags many more clients.",
    )
    submitted = st.form_submit_button("Assess")

if submitted:
    row = {"LIMIT_BAL": limit, "SEX": sex, "EDUCATION": education,
           "MARRIAGE": marriage, "AGE": age}
    row.update({f"PAY_{i}": pay[i] for i in range(1, 7)})
    row.update({f"BILL_AMT{i}": bill[i] for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": amt[i] for i in range(1, 7)})
    X_new = pd.DataFrame([row])[features]

    threshold = (
        artifact["threshold"] if mode.startswith("Balanced") else artifact["threshold_f2"]
    )
    score = float(model.predict_proba(X_new)[0, 1])

    if score >= threshold:
        st.error("Flagged: higher default risk")
    else:
        st.success("Not flagged")
        st.caption(
            "'Not flagged' does not mean 'safe': in testing, most missed defaults "
            "were clients with no visible delay in their repayment history."
        )

    st.metric(
        "Model score", f"{score:.2f}",
        help="A ranking score, not a calibrated probability of default.",
    )
    st.caption(f"Decision threshold ({mode}): {threshold:.3f}")

with st.expander("About this model"):
    st.markdown(
        """
- Soft **Voting Classifier** (Logistic Regression, KNN, Decision Tree) with scaling,
  one-hot encoding, and engineered repayment-behavior features inside one pipeline;
  thresholds chosen on out-of-fold training predictions.
- Test set (5,993 clients, 1,326 defaults), balanced threshold: precision about 0.51
  (95% CI 0.49-0.54), recall about 0.54 (0.51-0.57). It flags about 23% of clients.
- High-recall threshold: recall about 0.85 but precision about 0.31, and it flags
  about 61% of clients, so it is not practical as a screening filter.
- The model relies mostly on **repayment status**. It beats a one-line rule
  ("flag anyone at least 1 month late") by only about 0.025 F1 (95% CI 0.013-0.036).
- The score is **not** a calibrated probability.
- Sex, marital status and age are included for educational purposes. In real
  credit decisions their use may be restricted.
- The value `0` in repayment status is not officially documented in the dataset;
  here it behaves like "no delay".
"""
    )