import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.transformers import LotFrontageImputer, FeatureEngineer

MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"

st.set_page_config(page_title="House Price Prediction", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🏠House Price Prediction")
st.write(
    "Estimate a house's sale price based on its characteristics. "
    "Educational project, not a real valuation tool."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Basics & Location", "Areas & Rooms", "Basement & Garage", "Exterior & Sale"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        ms_zoning = st.selectbox("MS Zoning", ["RL", "RM", "FV", "RH", "C (all)"])
        neighborhood = st.selectbox(
            "Neighborhood",
            ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert",
             "NridgHt", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor",
             "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr",
             "SWISU", "Blmngtn", "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste"],
        )
        lot_area = st.number_input("Lot Area (sqft)", min_value=1000, max_value=50000, value=9500)
        lot_frontage = st.number_input("Lot Frontage (ft)", min_value=0, max_value=200, value=70)
    with c2:
        year_built = st.number_input("Year Built", min_value=1872, max_value=2010, value=1973)
        year_sold = st.number_input("Year Sold", min_value=2006, max_value=2010, value=2008)
        year_remod = st.number_input("Year Remodeled", min_value=1950, max_value=2010, value=1994)
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        gr_liv_area = st.number_input("Above Ground Living Area (sqft)", min_value=300, max_value=6000, value=1500)
        total_bsmt_sf = st.number_input("Total Basement SF", min_value=0, max_value=6000, value=1000)
        first_flr_sf = st.number_input("1st Floor SF", min_value=300, max_value=4000, value=1100)
        second_flr_sf = st.number_input("2nd Floor SF", min_value=0, max_value=2500, value=400)
    with c2:
        full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
        half_bath = st.selectbox("Half Bathrooms", [0, 1, 2], index=1)
        bedroom = st.selectbox("Bedrooms Above Grade", [0, 1, 2, 3, 4, 5, 6], index=3)
        tot_rooms = st.number_input("Total Rooms Above Grade", min_value=2, max_value=15, value=6)
        fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=1)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        bsmt_qual = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "Po", "None"], index=2)
        bsmt_fin_sf1 = st.number_input("Basement Finished SF", min_value=0, max_value=3000, value=400)
        garage_type = st.selectbox("Garage Type", ["Attchd", "Detchd", "BuiltIn", "Basment", "CarPort", "2Types", "None"])
    with c2:
        garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4], index=2)
        garage_area = st.number_input("Garage Area (sqft)", min_value=0, max_value=1500, value=480)
        garage_finish = st.selectbox("Garage Finish", ["Fin", "RFn", "Unf", "None"], index=1)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        exterior1st = st.selectbox("Exterior Covering", ["VinylSd", "HdBoard", "MetalSd", "Wd Sdng", "Plywood", "CemntBd", "BrkFace", "WdShing", "Stucco", "AsbShng"])
        central_air = st.selectbox("Central Air", ["Y", "N"])
        heating = st.selectbox("Heating", ["GasA", "GasW", "Grav", "Wall", "OthW", "Floor"])
    with c2:
        sale_type = st.selectbox("Sale Type", ["WD", "New", "COD", "ConLD", "ConLI", "ConLw", "CWD", "Oth", "Con"])
        sale_condition = st.selectbox("Sale Condition", ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"])
        functional = st.selectbox("Functional", ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"])

if st.button("Predict Price", use_container_width=True):
    row = pd.DataFrame([{
        "MSSubClass": 20, "MSZoning": ms_zoning, "LotFrontage": lot_frontage, "LotArea": lot_area,
        "Street": "Pave", "Alley": "None", "LotShape": "Reg", "LandContour": "Lvl",
        "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl",
        "Neighborhood": neighborhood, "Condition1": "Norm", "Condition2": "Norm",
        "BldgType": "1Fam", "HouseStyle": "1Story", "OverallQual": overall_qual,
        "OverallCond": overall_cond, "YearBuilt": year_built, "YearRemodAdd": year_remod,
        "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": exterior1st,
        "Exterior2nd": exterior1st, "MasVnrType": "None", "MasVnrArea": 0,
        "ExterQual": "TA", "ExterCond": "TA", "Foundation": "PConc",
        "BsmtQual": bsmt_qual, "BsmtCond": "TA", "BsmtExposure": "No",
        "BsmtFinType1": "Unf", "BsmtFinSF1": bsmt_fin_sf1, "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0, "BsmtUnfSF": max(total_bsmt_sf - bsmt_fin_sf1, 0),
        "TotalBsmtSF": total_bsmt_sf, "Heating": heating, "HeatingQC": "TA",
        "CentralAir": central_air, "Electrical": "SBrkr", "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf, "LowQualFinSF": 0, "GrLivArea": gr_liv_area,
        "BsmtFullBath": 0, "BsmtHalfBath": 0, "FullBath": full_bath, "HalfBath": half_bath,
        "BedroomAbvGr": bedroom, "KitchenAbvGr": 1, "KitchenQual": "TA",
        "TotRmsAbvGrd": tot_rooms, "Functional": functional, "Fireplaces": fireplaces,
        "FireplaceQu": "None" if fireplaces == 0 else "TA", "GarageType": garage_type,
        "GarageYrBlt": year_built, "GarageFinish": garage_finish, "GarageCars": garage_cars,
        "GarageArea": garage_area, "GarageQual": "TA", "GarageCond": "TA",
        "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 0, "EnclosedPorch": 0,
        "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "None",
        "Fence": "None", "MiscFeature": "None", "MiscVal": 0, "MoSold": 6,
        "YrSold": year_sold, "SaleType": sale_type, "SaleCondition": sale_condition,
    }])

    prediction = model.predict(row)[0]

    st.divider()
    st.subheader("Predicted Sale Price")
    st.metric("Estimated Price", f"${prediction:,.0f}")
    st.caption(
        "Typical error on held-out test data is about 8.6% of the actual price "
        "(R² ≈ 0.93). Educational estimate, not a real valuation."
    )