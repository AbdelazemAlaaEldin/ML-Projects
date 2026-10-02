"""Feature engineering shared by the notebook and the Streamlit app.

Every feature is computed from the row's own values, so it can be applied
before or after the train/test split without leaking information.
"""
import pandas as pd

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["max_delay"] = d[PAY_COLS].max(axis=1)
    d["n_months_late"] = (d[PAY_COLS] >= 1).sum(axis=1)
    d["mean_delay"] = d[PAY_COLS].clip(lower=0).mean(axis=1)
    d["utilization"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    ratio = d["PAY_AMT1"] / d["BILL_AMT1"].where(d["BILL_AMT1"] > 0)
    d["pay_ratio"] = ratio.fillna(1).clip(0, 1)
    return d
