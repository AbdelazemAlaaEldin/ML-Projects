# House Price Prediction

An end-to-end machine learning project for predicting house sale prices using Linear Regression, with a Streamlit application for interactive predictions.

## Project Overview

The goal of this project is to build a machine learning model that predicts the selling price of a house based on its features. The project covers the full workflow: exploratory data analysis, preprocessing, feature engineering, model training and evaluation, saving the trained pipeline, and deployment through a Streamlit app.

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Regression
- **Target variable:** `SalePrice`
- **Final model:** Linear Regression, trained on a log-transformed target

## Dataset

The project uses the House Prices - Advanced Regression Techniques dataset from Kaggle.

Dataset files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

The training data contains 1,460 samples and 80 input features, with `SalePrice` as the target variable. Only `train.csv` is used in the analysis and modeling notebook.

## Exploratory Data Analysis

The notebook examines the distribution of the target variable, its relationship with individual features, and correlations between numerical features. Key observations:

- `SalePrice` is right-skewed.
- `OverallQual` has a strong positive relationship with `SalePrice`.
- `GrLivArea` also has a strong positive relationship with `SalePrice`.
- `GarageCars`, `GarageArea`, `TotalBsmtSF`, and `1stFlrSF` show strong correlations with the target.
- Some features are correlated with each other, indicating possible multicollinearity.
- `SalePrice` also varies noticeably by `Neighborhood`.

## Data Preprocessing

The preprocessing is implemented as a scikit-learn pipeline (`ColumnTransformer`) with separate handling for numerical and categorical features:

- **Numerical features:** missing values imputed with the median.
- **Categorical features (regular):** missing values imputed with the most frequent value, then one-hot encoded.
- **Categorical features where missing means "absence" of something** (e.g. no pool, no garage): imputed with the constant value `"None"`, then one-hot encoded. This applies to `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, and `BsmtFinType2`.
- **`LotFrontage`:** handled separately with a custom transformer that fills missing values using the median `LotFrontage` within the same `Neighborhood`.

## Feature Engineering

Two additional features are created inside the pipeline via a custom transformer:

```
HouseAge = YrSold - YearBuilt
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

## Model Training

The model is Linear Regression, trained inside a single scikit-learn `Pipeline` that combines feature engineering, preprocessing, and the estimator.

A baseline model was first trained directly on `SalePrice`. A second version was then trained on a log-transformed target:

```python
y_train_log = np.log1p(y_train)
```

Predictions are converted back to the original price scale with:

```python
predicted_price = np.expm1(prediction_log)
```

The log-target version performed better on the held-out test set and was selected as the final model.

## Final Model Performance

| Metric | Score |
|---|---|
| MAE | 17,142.15 |
| RMSE | 26,045.47 |
| R² | 0.9116 |

### Train vs test performance

| Dataset | R² |
|---|---|
| Training | 0.9151 |
| Test | 0.9116 |

The small gap between training and test R² indicates that the model generalizes reasonably well to unseen data.

## Error Analysis

Predictions were compared against actual prices to inspect the largest errors. The model performs well overall but has larger errors on some extreme prices: certain high-priced houses were underpredicted, and certain lower-priced houses were overpredicted. This reflects a known limitation of a linear model on a dataset with nonlinear relationships.

## Saved Model

The trained pipeline (preprocessing + feature engineering + Linear Regression) is saved with `joblib` to:

```
models/house_price_model.pkl
```

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app.py`). The app collects house characteristics across four tabs (basics and location, areas and rooms, basement and garage, exterior and sale), builds a single-row DataFrame matching the training schema, and passes it directly into the saved pipeline, which performs all required preprocessing before generating a prediction.

Run the application with:

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
- Joblib
- Streamlit
- Jupyter Notebook

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

## Future Improvements

- Experimenting with regularized linear models such as Ridge and Lasso.
- Additional feature engineering.
- More detailed residual analysis.
- Better input validation in the Streamlit application.
- Improving the visual design of the application.

## Disclaimer

This project is an educational machine learning project. The predicted house price is an estimate generated from historical data and should not be considered a professional real-estate valuation.