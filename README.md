# 🏠 House Price Prediction

An end-to-end Machine Learning project for predicting house sale prices using Linear Regression.

## 📌 Project Overview

The goal of this project is to build a Machine Learning model that predicts the selling price of a house based on its features.

This project follows a complete Machine Learning workflow, starting from raw data analysis and preprocessing to model training, evaluation, saving the model, and deploying it using Streamlit.

## 🤖 Machine Learning Problem

- **Learning Type:** Supervised Learning
- **Problem Type:** Regression
- **Target Variable:** `SalePrice`
- **Final Model:** Linear Regression

## 📊 Dataset

The project uses the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

Dataset files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

The original dataset contains 1,460 training samples and 80 input features, with `SalePrice` as the target variable.

## 🔍 Exploratory Data Analysis

Several EDA techniques were used to understand the relationship between the house features and the target variable.

Important observations included:

- `SalePrice` is right-skewed.
- `OverallQual` has a strong positive relationship with `SalePrice`.
- `GrLivArea` also has a strong positive relationship with `SalePrice`.
- Features such as `GarageCars`, `GarageArea`, `TotalBsmtSF`, and `1stFlrSF` also show strong relationships with the target.
- Some features show high correlation with each other, which can indicate possible multicollinearity.

## 🧹 Data Preprocessing

The preprocessing pipeline handles:

- Missing values
- Numerical features
- Categorical features
- One-hot encoding
- Features where missing values represent the absence of something

Examples include:

- Basement
- Garage
- Pool
- Fence
- Alley
- Fireplace

### LotFrontage

Missing `LotFrontage` values are handled using the median value within the property's `Neighborhood`.

## ⚙️ Feature Engineering

Two additional features were created:

### HouseAge

```text
HouseAge = YrSold - YearBuilt
```

### TotalSF

```text
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

These features are generated automatically inside the preprocessing pipeline.

## 📈 Model Training

The project uses Linear Regression.

A baseline model was first trained using the original `SalePrice` target.

After evaluation, a second experiment was performed using a logarithmic transformation of the target:

```python
y_train_log = np.log1p(y_train)
```

The predictions were converted back to the original price scale using:

```python
predicted_price = np.expm1(prediction_log)
```

The log-target approach achieved better performance on the held-out test set and was selected as the final model.

## 📏 Final Model Performance

| Metric | Score |
|---|---|
| MAE | 17,142.15 |
| RMSE | 26,045.47 |
| R² | 0.9116 |

| | R² |
|---|---|
| Training | 0.9151 |
| Test | 0.9116 |

## 🔎 Error Analysis

The model predictions were compared with the actual house prices.

The analysis showed that the model performs well for many observations but has larger errors for some extreme house prices.

In particular, some high-priced houses were underpredicted, while some lower-priced houses were overpredicted.

This highlights one of the limitations of using a linear model for a dataset containing complex and potentially nonlinear relationships.

## 💾 Saved Model

The final trained pipeline is saved as:

```text
models/house_price_model.pkl
```

The saved pipeline contains the preprocessing steps and the trained Linear Regression model.

## 🌐 Streamlit Application

The trained model is deployed using Streamlit.

The application allows the user to enter house characteristics and receive an estimated house price.

Run the application with:

```bash
streamlit run app/app.py
```

The application automatically performs the required preprocessing before generating the prediction.

## 📁 Project Structure

```text
House-price-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       ├── sample_submission.csv
│       └── data_description.txt
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app/app.py
```

## 📌 Future Improvements

Possible future improvements include:

- Experimenting with regularized Linear Regression such as Ridge and Lasso.
- Additional feature engineering.
- More detailed residual analysis.
- Better input validation in the Streamlit application.
- Improving the visual design of the application.

## ⚠️ Disclaimer

This project is an educational Machine Learning project.

The predicted house price is an estimate generated from historical data and should not be considered a professional real-estate valuation.