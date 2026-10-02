"""Custom preprocessing transformers shared by the notebook and the Streamlit app."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LotFrontageImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        self.medians_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.global_median_ = X["LotFrontage"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["LotFrontage"] = X.apply(
            lambda row: self.medians_.get(row["Neighborhood"], self.global_median_)
            if pd.isnull(row["LotFrontage"]) else row["LotFrontage"],
            axis=1,
        )
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        return X