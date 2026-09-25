import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================
# Page Configuration
# =========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# =========================================
# Paths
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"


# =========================================
# Custom Transformers
# =========================================

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

        X["TotalSF"] = (
            X["TotalBsmtSF"]
            + X["1stFlrSF"]
            + X["2ndFlrSF"]
        )

        return X


class LotFrontageImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.medians_ = X.groupby("Neighborhood")["LotFrontage"].median()
        return self

    def transform(self, X):
        X = X.copy()

        X["LotFrontage"] = X["LotFrontage"].fillna(
            X["Neighborhood"].map(self.medians_)
        )

        return X


# =========================================
# Load Model and Data
# =========================================

model = joblib.load(MODEL_PATH)

train_data = pd.read_csv(DATA_PATH)


# =========================================
# Title
# =========================================

st.title("🏠 House Price Prediction")

st.write(
    "Enter the house characteristics below to estimate "
    "its sale price."
)


# =========================================
# Tabs
# =========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Basics & Location",
    "📐 Areas & Rooms",
    "🏗️ Basement & Garage",
    "🎨 Exterior & Sale"
])


# =========================================
# TAB 1: Basics & Location
# =========================================

with tab1:

    st.header("🏠 Basic House Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        overall_qual = st.number_input(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

    with col2:
        overall_cond = st.number_input(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

    with col3:
        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=int(train_data["YearBuilt"].median()),
            step=1
        )

    with col1:
        year_remod_add = st.number_input(
            "Year Remodeled",
            min_value=1800,
            max_value=2026,
            value=int(train_data["YearRemodAdd"].median()),
            step=1
        )

    st.header("📍 Location & House Type")

    col1, col2, col3 = st.columns(3)

    with col1:
        neighborhood = st.selectbox(
            "Neighborhood",
            sorted(train_data["Neighborhood"].dropna().unique())
        )

    with col2:
        ms_subclass = st.selectbox(
            "MSSubClass",
            sorted(train_data["MSSubClass"].dropna().unique())
        )

    with col3:
        ms_zoning = st.selectbox(
            "MSZoning",
            sorted(train_data["MSZoning"].dropna().unique())
        )

    st.header("📐 Lot & Property")

    col1, col2, col3 = st.columns(3)

    with col1:
        lot_frontage = st.number_input(
            "Lot Frontage (ft)",
            min_value=0.0,
            value=float(train_data["LotFrontage"].median()),
            step=1.0
        )

    with col2:
        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=0,
            value=int(train_data["LotArea"].median()),
            step=1
        )

    with col3:
        lot_shape = st.selectbox(
            "Lot Shape",
            sorted(train_data["LotShape"].dropna().unique())
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        street = st.selectbox(
            "Street",
            sorted(train_data["Street"].dropna().unique())
        )

    with col2:
        alley_options = ["None"] + sorted(
            train_data["Alley"].dropna().unique().tolist()
        )

        alley = st.selectbox(
            "Alley Access",
            alley_options
        )

    with col3:
        lot_config = st.selectbox(
            "Lot Configuration",
            sorted(train_data["LotConfig"].dropna().unique())
        )

    col1, col2 = st.columns(2)

    with col1:
        land_contour = st.selectbox(
            "Land Contour",
            sorted(train_data["LandContour"].dropna().unique())
        )

    with col2:
        land_slope = st.selectbox(
            "Land Slope",
            sorted(train_data["LandSlope"].dropna().unique())
        )

    st.header("🏘️ Nearby Conditions & Building Type")

    col1, col2, col3 = st.columns(3)

    with col1:
        condition1 = st.selectbox(
            "Nearby Condition 1",
            sorted(train_data["Condition1"].dropna().unique())
        )

    with col2:
        condition2 = st.selectbox(
            "Nearby Condition 2",
            sorted(train_data["Condition2"].dropna().unique())
        )

    with col3:
        bldg_type = st.selectbox(
            "Building Type",
            sorted(train_data["BldgType"].dropna().unique())
        )

    house_style = st.selectbox(
        "House Style",
        sorted(train_data["HouseStyle"].dropna().unique())
    )


# =========================================
# TAB 2: Areas & Rooms
# =========================================

with tab2:

    st.header("📐 Living Area")

    col1, col2, col3 = st.columns(3)

    with col1:
        gr_liv_area = st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=0,
            value=int(train_data["GrLivArea"].median()),
            step=1
        )

    with col2:
        first_flr_sf = st.number_input(
            "1st Floor Area (sq ft)",
            min_value=0,
            value=int(train_data["1stFlrSF"].median()),
            step=1
        )

    with col3:
        second_flr_sf = st.number_input(
            "2nd Floor Area (sq ft)",
            min_value=0,
            value=int(train_data["2ndFlrSF"].median()),
            step=1
        )

    st.header("🛏️ Rooms & Bathrooms")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            value=int(train_data["BedroomAbvGr"].median()),
            step=1
        )

    with col2:
        full_bath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            value=int(train_data["FullBath"].median()),
            step=1
        )

    with col3:
        half_bath = st.number_input(
            "Half Bathrooms",
            min_value=0,
            value=int(train_data["HalfBath"].median()),
            step=1
        )

    with col4:
        total_rooms = st.number_input(
            "Total Rooms",
            min_value=0,
            value=int(train_data["TotRmsAbvGrd"].median()),
            step=1
        )

    st.header("🔧 Additional House Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        low_qual_fin_sf = st.number_input(
            "Low Quality Finished Area (sq ft)",
            min_value=0,
            value=int(train_data["LowQualFinSF"].median()),
            step=1
        )

    with col2:
        kitchen_abv_gr = st.number_input(
            "Above Ground Kitchens",
            min_value=0,
            value=int(train_data["KitchenAbvGr"].median()),
            step=1
        )

    with col3:
        fireplaces = st.number_input(
            "Fireplaces",
            min_value=0,
            value=int(train_data["Fireplaces"].median()),
            step=1
        )

    fireplace_qu = st.selectbox(
        "Fireplace Quality",
        ["None"] + sorted(
            train_data["FireplaceQu"].dropna().unique().tolist()
        )
    )

    st.header("🌳 Outdoor Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        wood_deck_sf = st.number_input(
            "Wood Deck Area (sq ft)",
            min_value=0,
            value=int(train_data["WoodDeckSF"].median()),
            step=1
        )

    with col2:
        open_porch_sf = st.number_input(
            "Open Porch Area (sq ft)",
            min_value=0,
            value=int(train_data["OpenPorchSF"].median()),
            step=1
        )

    with col3:
        enclosed_porch = st.number_input(
            "Enclosed Porch Area (sq ft)",
            min_value=0,
            value=int(train_data["EnclosedPorch"].median()),
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        three_ssn_porch = st.number_input(
            "3 Season Porch Area (sq ft)",
            min_value=0,
            value=int(train_data["3SsnPorch"].median()),
            step=1
        )

    with col2:
        screen_porch = st.number_input(
            "Screen Porch Area (sq ft)",
            min_value=0,
            value=int(train_data["ScreenPorch"].median()),
            step=1
        )

    with col3:
        pool_area = st.number_input(
            "Pool Area (sq ft)",
            min_value=0,
            value=int(train_data["PoolArea"].median()),
            step=1
        )

    st.header("🏊 Pool & Fence")

    col1, col2 = st.columns(2)

    with col1:
        pool_options = ["None"] + sorted(
            train_data["PoolQC"].dropna().unique().tolist()
        )

        pool_qc = st.selectbox(
            "Pool Quality",
            pool_options
        )

    with col2:
        fence_options = ["None"] + sorted(
            train_data["Fence"].dropna().unique().tolist()
        )

        fence = st.selectbox(
            "Fence",
            fence_options
        )


# =========================================
# TAB 3: Basement, Foundation & Garage
# =========================================

with tab3:

    st.header("🏗️ Basement")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            value=int(train_data["TotalBsmtSF"].median()),
            step=1
        )

    with col2:
        bsmt_fin_sf1 = st.number_input(
            "Finished Basement Area 1 (sq ft)",
            min_value=0,
            value=int(train_data["BsmtFinSF1"].median()),
            step=1
        )

    with col3:
        bsmt_fin_sf2 = st.number_input(
            "Finished Basement Area 2 (sq ft)",
            min_value=0,
            value=int(train_data["BsmtFinSF2"].median()),
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        bsmt_unf_sf = st.number_input(
            "Unfinished Basement Area (sq ft)",
            min_value=0,
            value=int(train_data["BsmtUnfSF"].median()),
            step=1
        )

    with col2:
        bsmt_full_bath = st.number_input(
            "Basement Full Bathrooms",
            min_value=0,
            value=int(train_data["BsmtFullBath"].median()),
            step=1
        )

    with col3:
        bsmt_half_bath = st.number_input(
            "Basement Half Bathrooms",
            min_value=0,
            value=int(train_data["BsmtHalfBath"].median()),
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        bsmt_qual = st.selectbox(
            "Basement Quality",
            ["None"] + sorted(
                train_data["BsmtQual"].dropna().unique().tolist()
            )
        )

    with col2:
        bsmt_cond = st.selectbox(
            "Basement Condition",
            ["None"] + sorted(
                train_data["BsmtCond"].dropna().unique().tolist()
            )
        )

    with col3:
        bsmt_exposure = st.selectbox(
            "Basement Exposure",
            ["None"] + sorted(
                train_data["BsmtExposure"].dropna().unique().tolist()
            )
        )

    col1, col2 = st.columns(2)

    with col1:
        bsmt_fin_type1 = st.selectbox(
            "Basement Finished Area 1 Type",
            ["None"] + sorted(
                train_data["BsmtFinType1"].dropna().unique().tolist()
            )
        )

    with col2:
        bsmt_fin_type2 = st.selectbox(
            "Basement Finished Area 2 Type",
            ["None"] + sorted(
                train_data["BsmtFinType2"].dropna().unique().tolist()
            )
        )

    st.header("🧱 Foundation")

    foundation = st.selectbox(
        "Foundation Type",
        sorted(train_data["Foundation"].dropna().unique())
    )

    st.header("🚗 Garage")

    col1, col2, col3 = st.columns(3)

    with col1:
        garage_cars = st.number_input(
            "Garage Capacity (Cars)",
            min_value=0,
            value=int(train_data["GarageCars"].median()),
            step=1
        )

    with col2:
        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            value=int(train_data["GarageArea"].median()),
            step=1
        )

    with col3:
        garage_year = st.number_input(
            "Garage Year Built",
            min_value=1800,
            max_value=2026,
            value=int(train_data["GarageYrBlt"].median()),
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        garage_type = st.selectbox(
            "Garage Type",
            ["None"] + sorted(
                train_data["GarageType"].dropna().unique().tolist()
            )
        )

    with col2:
        garage_finish = st.selectbox(
            "Garage Finish",
            ["None"] + sorted(
                train_data["GarageFinish"].dropna().unique().tolist()
            )
        )

    with col3:
        garage_qual = st.selectbox(
            "Garage Quality",
            ["None"] + sorted(
                train_data["GarageQual"].dropna().unique().tolist()
            )
        )

    garage_cond = st.selectbox(
        "Garage Condition",
        ["None"] + sorted(
            train_data["GarageCond"].dropna().unique().tolist()
        )
    )


# =========================================
# TAB 4: Exterior, Utilities & Sale
# =========================================

with tab4:

    st.header("🏡 Exterior")

    col1, col2, col3 = st.columns(3)

    with col1:
        roof_style = st.selectbox(
            "Roof Style",
            sorted(train_data["RoofStyle"].dropna().unique())
        )

    with col2:
        roof_matl = st.selectbox(
            "Roof Material",
            sorted(train_data["RoofMatl"].dropna().unique())
        )

    with col3:
        exterior1 = st.selectbox(
            "Exterior Material 1",
            sorted(train_data["Exterior1st"].dropna().unique())
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        exterior2 = st.selectbox(
            "Exterior Material 2",
            sorted(train_data["Exterior2nd"].dropna().unique())
        )

    with col2:
        mas_vnr_type = st.selectbox(
            "Masonry Veneer Type",
            ["None"] + sorted(
                train_data["MasVnrType"].dropna().unique().tolist()
            )
        )

    with col3:
        mas_vnr_area = st.number_input(
            "Masonry Veneer Area (sq ft)",
            min_value=0.0,
            value=float(train_data["MasVnrArea"].median()),
            step=1.0
        )

    col1, col2 = st.columns(2)

    with col1:
        exter_qual = st.selectbox(
            "Exterior Quality",
            sorted(train_data["ExterQual"].dropna().unique())
        )

    with col2:
        exter_cond = st.selectbox(
            "Exterior Condition",
            sorted(train_data["ExterCond"].dropna().unique())
        )

    st.header("🔥 Heating")

    heating = st.selectbox(
        "Heating Type",
        sorted(train_data["Heating"].dropna().unique())
    )

    st.header("⭐ Quality & Utilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        kitchen_qual = st.selectbox(
            "Kitchen Quality",
            sorted(train_data["KitchenQual"].dropna().unique())
        )

    with col2:
        functional = st.selectbox(
            "Home Functionality",
            sorted(train_data["Functional"].dropna().unique())
        )

    with col3:
        heating_qc = st.selectbox(
            "Heating Quality",
            sorted(train_data["HeatingQC"].dropna().unique())
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        central_air = st.selectbox(
            "Central Air",
            sorted(train_data["CentralAir"].dropna().unique())
        )

    with col2:
        electrical = st.selectbox(
            "Electrical System",
            sorted(train_data["Electrical"].dropna().unique())
        )

    with col3:
        utilities = st.selectbox(
            "Utilities",
            sorted(train_data["Utilities"].dropna().unique())
        )

    st.header("🧰 Miscellaneous")

    col1, col2 = st.columns(2)

    with col1:
        misc_feature_options = ["None"] + sorted(
            train_data["MiscFeature"].dropna().unique().tolist()
        )

        misc_feature = st.selectbox(
            "Miscellaneous Feature",
            misc_feature_options
        )

    with col2:
        misc_val = st.number_input(
            "Miscellaneous Feature Value",
            min_value=0,
            value=int(train_data["MiscVal"].median()),
            step=1
        )

    st.header("🚘 Driveway")

    paved_drive = st.selectbox(
        "Paved Driveway",
        sorted(train_data["PavedDrive"].dropna().unique())
    )

    st.header("📅 Sale Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        mo_sold = st.selectbox(
            "Month Sold",
            sorted(train_data["MoSold"].dropna().unique())
        )

    with col2:
        yr_sold = st.number_input(
            "Year Sold",
            min_value=2000,
            max_value=2030,
            value=int(train_data["YrSold"].median()),
            step=1
        )

    with col3:
        sale_type = st.selectbox(
            "Sale Type",
            sorted(train_data["SaleType"].dropna().unique())
        )

    sale_condition = st.selectbox(
        "Sale Condition",
        sorted(train_data["SaleCondition"].dropna().unique())
    )


# =========================================
# Prediction (outside the tabs, always visible)
# =========================================

st.divider()
st.header("💰 House Price Prediction")

if st.button("🔮 Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([{
        "MSSubClass": ms_subclass,
        "MSZoning": ms_zoning,
        "LotFrontage": lot_frontage,
        "LotArea": lot_area,
        "Street": street,
        "Alley": alley,
        "LotShape": lot_shape,
        "LandContour": land_contour,
        "Utilities": utilities,
        "LotConfig": lot_config,
        "LandSlope": land_slope,
        "Neighborhood": neighborhood,
        "Condition1": condition1,
        "Condition2": condition2,
        "BldgType": bldg_type,
        "HouseStyle": house_style,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod_add,
        "RoofStyle": roof_style,
        "RoofMatl": roof_matl,
        "Exterior1st": exterior1,
        "Exterior2nd": exterior2,
        "MasVnrType": mas_vnr_type,
        "MasVnrArea": mas_vnr_area,
        "ExterQual": exter_qual,
        "ExterCond": exter_cond,
        "Foundation": foundation,
        "BsmtQual": bsmt_qual,
        "BsmtCond": bsmt_cond,
        "BsmtExposure": bsmt_exposure,
        "BsmtFinType1": bsmt_fin_type1,
        "BsmtFinType2": bsmt_fin_type2,
        "BsmtFinSF1": bsmt_fin_sf1,
        "BsmtFinSF2": bsmt_fin_sf2,
        "BsmtUnfSF": bsmt_unf_sf,
        "TotalBsmtSF": total_bsmt_sf,
        "Heating": heating,
        "HeatingQC": heating_qc,
        "CentralAir": central_air,
        "Electrical": electrical,
        "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf,
        "LowQualFinSF": low_qual_fin_sf,
        "GrLivArea": gr_liv_area,
        "BsmtFullBath": bsmt_full_bath,
        "BsmtHalfBath": bsmt_half_bath,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "KitchenAbvGr": kitchen_abv_gr,
        "KitchenQual": kitchen_qual,
        "TotRmsAbvGrd": total_rooms,
        "Functional": functional,
        "Fireplaces": fireplaces,
        "FireplaceQu": fireplace_qu,
        "GarageType": garage_type,
        "GarageYrBlt": garage_year,
        "GarageFinish": garage_finish,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "GarageQual": garage_qual,
        "GarageCond": garage_cond,
        "PavedDrive": paved_drive,
        "WoodDeckSF": wood_deck_sf,
        "OpenPorchSF": open_porch_sf,
        "EnclosedPorch": enclosed_porch,
        "3SsnPorch": three_ssn_porch,
        "ScreenPorch": screen_porch,
        "PoolArea": pool_area,
        "PoolQC": pool_qc,
        "Fence": fence,
        "MiscFeature": misc_feature,
        "MiscVal": misc_val,
        "MoSold": mo_sold,
        "YrSold": yr_sold,
        "SaleType": sale_type,
        "SaleCondition": sale_condition
    }])

    # -----------------------------------------
    # Predict
    # -----------------------------------------

    prediction_log = model.predict(input_data)

    predicted_price = np.expm1(prediction_log[0])

    # -----------------------------------------
    # Display Result
    # -----------------------------------------

    st.success("Prediction completed successfully!")

    st.metric(
        label="Estimated House Price",
        value=f"${predicted_price:,.0f}"
    )

    st.info(
        "The prediction is generated using the trained "
        "Linear Regression model with a log-transformed target."
    )