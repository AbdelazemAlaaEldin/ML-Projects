# 🏠House Price Prediction

## Project Overview

This project builds an end-to-end machine learning solution to predict the sale price of a house based on its characteristics: size, quality, location, age, and other features.

The project compares Linear Regression, Ridge, and Lasso for regression. The workflow covers data understanding, cleaning, exploratory data analysis, preprocessing, feature engineering, model comparison, hyperparameter tuning, error analysis, model packaging, and Streamlit deployment.

> **Disclaimer:** This project is for educational purposes only. The predicted price is an estimate generated from historical data and should not be considered a professional real-estate valuation.

## Problem Statement

The goal of this project is to predict a house's sale price based on its physical characteristics and sale information.

This is a:

- Supervised learning problem
- Regression problem (the target, `SalePrice`, is a continuous value)

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Regression
- **Algorithms compared:** Linear Regression, Ridge, Lasso (all trained on a log-transformed target)
- **Target variable:** `SalePrice`

## Dataset

The project uses the House Prices - Advanced Regression Techniques Kaggle competition dataset, downloaded using `kagglehub.competition_download`.

Dataset source:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The dataset contains:

- 1,460 training samples
- 80 input features (house size, quality ratings, location, age, garage and basement details, sale information, etc.)
- 19 columns with missing values
- `SalePrice`, right-skewed (skewness ≈ 1.88)

Only `train.csv` is used for analysis and modeling, since it is the only file with a known `SalePrice`. `test.csv`, `sample_submission.csv`, and `data_description.txt` are also downloaded but not used in training.

## Data Cleaning

Two clear outliers were identified and removed: two houses with an unusually large above-ground living area (`GrLivArea` > 4,000 sqft) sold for a relatively low price (under $300,000), which would otherwise distort the linear models. This leaves 1,458 training samples.

## Exploratory Data Analysis

The notebook examines the distribution of the target, its relationship with individual features, and correlations between numerical features.

Key observations:

- `OverallQual` has the strongest correlation with `SalePrice` (≈ 0.79), followed by `GrLivArea` (≈ 0.71).
- `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`, and `YearRemodAdd` are also meaningfully correlated with the target.
- Several feature pairs are highly correlated with each other, indicating multicollinearity: `GarageCars`/`GarageArea` (≈ 0.88), `GrLivArea`/`TotRmsAbvGrd` (≈ 0.83), and `TotalBsmtSF`/`1stFlrSF` (≈ 0.82).
- `SalePrice` varies substantially by `Neighborhood`, with `NoRidge`, `NridgHt`, and `StoneBr` the most expensive, and `MeadowV`, `IDOTRR`, and `BrDale` among the cheapest.

## Preprocessing

The preprocessing is implemented as a scikit-learn `ColumnTransformer` wrapped together with feature engineering inside a single `Pipeline`, so every training and evaluation step (including cross-validation) fits preprocessing fresh on each training fold, avoiding leakage from validation data into the transformers.

- **Numerical features:** missing values imputed with the median, then standardized with `StandardScaler`.
- **Categorical features (regular):** missing values imputed with the most frequent value, then one-hot encoded.
- **Categorical features where missing means "absence" of something** (e.g. no pool, no garage, no basement): imputed with the constant value `"None"`, then one-hot encoded. This applies to `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, and `MasVnrType`.
- **`LotFrontage`:** handled separately by a custom `LotFrontageImputer` transformer that fills missing values using the median `LotFrontage` within the same `Neighborhood`, falling back to the global median for a neighborhood not seen during training.

Scaling the numerical features matters here not for Linear Regression itself, but because it is needed for a fair comparison with Ridge and Lasso, whose regularization is sensitive to feature scale.

## Feature Engineering

Two additional features are created inside the pipeline via a custom `FeatureEngineer` transformer:

```
HouseAge = YrSold - YearBuilt
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

Both custom transformers (`LotFrontageImputer` and `FeatureEngineer`) are implemented in a shared `src/transformers.py` module rather than defined inline in the notebook. This is required for the saved pipeline to be loadable outside the notebook: `joblib`/`pickle` store a reference to a class's module and name rather than its code, so a class defined in the notebook (effectively living in `__main__`) cannot be found when the pipeline is reloaded in a separate process such as the Streamlit app. Importing both classes from a real, shared module fixes this.

## Train/Test Split

The dataset is divided using an 80/20 train/test split (not stratified, since this is a regression problem).

The final split contains:

- 1,166 training houses
- 292 test houses

## Model Training

Three regression models were compared, all trained on `log1p(SalePrice)` via `TransformedTargetRegressor`, so the target transform and its inverse (`expm1`) are applied automatically and consistently during cross-validation, training, and prediction:

1. **Linear Regression** (log target) — baseline
2. **Ridge** (log target) — L2-regularized, `alpha` tuned by grid search
3. **Lasso** (log target) — L1-regularized, `alpha` tuned by grid search

5-fold cross-validation on the training set:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression (raw target) | 19,094 ± 1,976 | 41,858 ± 19,153 | 0.679 ± 0.295 |
| Linear Regression (log target) | 14,910 ± 1,223 | 22,299 ± 1,756 | 0.921 ± 0.011 |
| Ridge (log target, tuned) | – | – | 0.935 |
| Lasso (log target, tuned) | – | – | 0.937 |

The raw-target Linear Regression is noticeably less stable across folds (large standard deviations), reflecting the target's skew. The log transform alone gives a large, stable improvement. Given the multicollinearity observed in the EDA, Ridge and Lasso were expected to help further by shrinking correlated coefficients, which the cross-validation results confirm.

### Hyperparameter Tuning

Ridge and Lasso's `alpha` were each tuned with a 5-fold cross-validated grid search, scored on R²:

- **Ridge:** best `alpha = 10`
- **Lasso:** best `alpha = 0.0005`

## Final Model Performance

Lasso (log target, tuned) was selected as the final model. Performance of all three final candidates on the held-out test set:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression (log target) | 15,489 | 21,820 | 0.9138 |
| Ridge (log target, tuned) | 14,529 | 20,044 | 0.9273 |
| **Lasso (log target, tuned)** | **14,125** | **19,425** | **0.9317** |

### Bootstrap Confidence Intervals

For the final (Lasso) model, 95% bootstrap confidence intervals on the test set:

| Metric | 95% Bootstrap CI |
|---|---|
| MAE | [12,804, 15,739] |
| RMSE | [17,336, 21,508] |
| R² | [0.9100, 0.9477] |

## Error Analysis

- The mean absolute percentage error is about 8.6%.
- Errors are reasonably balanced: 151 test houses are underpredicted and 141 are overpredicted.
- Breaking the test set into four price bands shows a clear pattern: the model slightly overpredicts in the two cheapest bands (mean residual negative) and slightly underpredicts in the two most expensive bands (mean residual positive, and most pronounced in the top band). This is consistent with the typical limitation of a linear model on a target with nonlinear tail behavior: it compresses the extremes toward the middle of the price distribution.
- The ten largest individual errors are a mix of both directions and span a wide range of price levels, without an obvious common feature driving them beyond general pricing noise.

### Feature Importance

The largest Lasso coefficients (on standardized features) include `MSZoning_C (all)` (negative — commercial-zoned residential properties sell for less), `Neighborhood_Crawfor` and `Neighborhood_StoneBr` (positive location premiums), `GrLivArea` and `TotalSF` (positive, confirming size matters even after controlling for other features), `OverallQual` (positive), and `HouseAge` (negative, as expected). This aligns with the correlations seen during EDA.

## Machine Learning Pipeline

The final model is saved as a single scikit-learn `Pipeline` combining the `LotFrontageImputer`, the `FeatureEngineer`, the `ColumnTransformer`, and the `TransformedTargetRegressor`-wrapped Lasso model, so raw house feature values can be passed in directly and a price in the original dollar scale is returned. The saved pipeline was verified to produce identical predictions after being reloaded with `joblib`.

## Model Saving

The final trained pipeline is saved with `joblib` to:

```
models/house_price_model.pkl
```

Because the pipeline's custom transformers are imported from `src/transformers.py`, that module must be importable (on the Python path) both when saving the model and when loading it elsewhere, such as in the Streamlit app.

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app.py`). The app collects the house characteristics most relevant to the model (based on the EDA and the final coefficients) across four tabs — basics and location, areas and rooms, basement and garage, and exterior and sale — and fills in the remaining, less influential columns with their most common value from the training data. It builds a single-row DataFrame matching the training schema and passes it directly into the saved pipeline, which performs all required preprocessing and feature engineering before generating a prediction, displayed together with a note on the model's typical error.

### Run the application

From the project root:

```
streamlit run app/app.py
```

## Project Structure

```
House-price-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       ├── data_description.txt
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   └── transformers.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- kagglehub

## How to Run

1. Clone the repository

```
git clone <repository-url>
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the Streamlit application

```
streamlit run app/app.py
```

4. Run the notebook

Open `notebooks/house_price_analysis.ipynb`. Note that the dataset is hosted as a Kaggle competition rather than a plain dataset, so the Kaggle account used must first join the competition (accept its rules) at the link above before `kagglehub.competition_download` will succeed. The notebook then performs the complete analysis, compares the models, and regenerates the saved model artifact.

## Learning Objective

This project was built as part of a practical machine learning learning path, with the goal of understanding linear models for regression through a complete real-world problem.

The project focuses on understanding:

- Why a skewed regression target often benefits from a log transform, and how to apply it safely and consistently with `TransformedTargetRegressor`
- How Ridge and Lasso regularization address multicollinearity, and how their hyperparameter (`alpha`) is tuned with cross-validation
- How to compare regression models with cross-validation and bootstrap confidence intervals, rather than a single train/test split
- How to perform error analysis for a regression problem, including residuals by price band
- Why custom pipeline transformers must live in an importable module, not inline in a notebook, for a saved pipeline to be reusable elsewhere
- How to package a trained pipeline for deployment and serve it with Streamlit