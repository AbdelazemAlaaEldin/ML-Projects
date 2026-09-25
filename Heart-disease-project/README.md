# Heart Disease Prediction

## Project Overview

This project builds an end-to-end machine learning solution for predicting the presence of heart disease from clinical patient features.

The project uses Logistic Regression as the classification algorithm and includes data analysis, preprocessing, feature representation experiments, hyperparameter tuning, evaluation, error analysis, model saving, and Streamlit deployment.

> **Disclaimer:** This application is for educational purposes only and is not a medical diagnostic tool.

## Problem Statement

The goal of this project is to predict whether a patient has heart disease based on clinical features.

This is a:

- Supervised learning problem
- Binary classification problem

The target variable represents whether heart disease is present (`1`) or absent (`0`).

## Dataset

The dataset is the processed Cleveland heart disease dataset, downloaded via `kagglehub` from the `mdzawharulislam/uci-processedclevelanddata` dataset and stored locally as `heart_disease_data.csv`.

It contains:

- 303 observations
- 13 input features
- 1 target variable, with no missing values

### Features

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

### Target

`target`

- `0` → No heart disease (138 cases)
- `1` → Heart disease (165 cases)

## Exploratory Data Analysis

The dataset was explored using:

- Target distribution analysis
- Numerical feature distributions
- Categorical feature distributions, including their relationship with the target
- Boxplots of numerical features against the target
- A correlation heatmap across all features

The correlation and distribution analysis were used as exploratory tools, not as the only basis for feature selection.

## Train/Test Split

The dataset was split into:

- 80% training data
- 20% test data

A stratified split was used to preserve the target class distribution.

## Data Preprocessing

Preprocessing is implemented with a scikit-learn `ColumnTransformer`:

- **Numerical features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are standardized using `StandardScaler`.
- **Categorical features** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`, plus `ca` in the final configuration) are transformed using `OneHotEncoder`.

Preprocessing and the model are combined inside a single scikit-learn `Pipeline`, so preprocessing and prediction are handled consistently for both training and inference.

## Model Development

The model was developed iteratively, moving from a baseline to a tuned final version:

1. **Baseline Logistic Regression** on the standard preprocessing (`ca` treated as numerical): accuracy 80.33%, ROC-AUC 88.42%.
2. **Cross-validated hyperparameter search** over the regularization parameter `C`, using 5-fold stratified cross-validation.
3. **Feature representation experiment:** `ca` was tested as both a numerical and a categorical feature; treating it as categorical produced a better validation ROC-AUC.
4. **Grid search** over `C` values `[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]` combined with both preprocessing variants (`ca` numerical vs. categorical), using 5-fold stratified cross-validation optimized for ROC-AUC.

The grid search selected `C = 0.2` with `ca` treated as categorical, with a cross-validated ROC-AUC of 89.82%, as the best configuration.

## Final Model Performance

The final model was evaluated on the held-out test split:

| Metric | Score |
|---|---:|
| Accuracy | 88.52% |
| Precision | 86.11% |
| Recall | 93.94% |
| F1-Score | 89.86% |
| ROC-AUC | 91.13% |

### Confusion Matrix

- True Negatives: 23
- False Positives: 5
- False Negatives: 2
- True Positives: 31

The model correctly classified 54 out of 61 test samples.

> The same test split was used throughout iterative model development (baseline, C=0.1 experiment, and final grid search evaluation), so it should not be treated as a completely untouched final holdout set.

## Error Analysis

The final model's errors on the test set consisted of:

- 5 false positives (predicted heart disease, actual negative)
- 2 false negatives (predicted no heart disease, actual positive)

This breakdown was used to evaluate the model beyond a single accuracy figure, since it shows the model is somewhat more prone to false positives than false negatives at the default 0.5 threshold.

## Model Saving

The final trained pipeline (preprocessing and Logistic Regression) is saved with `joblib` to:

```
models/heart_disease_logistic_pipeline.pkl
```

## Streamlit Application

A Streamlit application (`app.py`) loads the saved pipeline and lets the user enter a patient's clinical features through numeric inputs and dropdowns. On submission, it builds a single-row DataFrame matching the training schema and returns:

- The predicted class (heart disease / no heart disease)
- The predicted probability of heart disease

The application is located at `app/app.py` and includes an on-screen disclaimer that it is for educational purposes only.

### Run the application

From the project's root directory:

```
streamlit run app/app.py
```

## Project Structure

```
Heart-disease-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── heart_disease_data.csv
│
├── models/
│   └── heart_disease_logistic_pipeline.pkl
│
├── notebooks/
│   └── heart_disease_analysis.ipynb
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

## Future Improvements

- Evaluating the model on a separate, completely untouched holdout set.
- More extensive hyperparameter tuning.
- Comparing Logistic Regression with additional classification algorithms.
- Improving the Streamlit interface.
- Adding more detailed visualizations.

## Disclaimer

This project is developed for educational and machine learning practice purposes. It should not be used as a medical diagnostic system or as a substitute for professional medical advice.