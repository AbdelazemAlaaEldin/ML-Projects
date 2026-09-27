# Customer Churn Prediction using KNN

## Project Overview

This project builds an end-to-end machine learning solution to predict whether a telecommunications customer is likely to churn.

The project uses the K-Nearest Neighbors (KNN) algorithm for binary classification. The workflow covers data understanding, cleaning, EDA, preprocessing, model training, hyperparameter tuning, error analysis, model saving, and Streamlit deployment.

## Problem Statement

Customer churn is an important business problem for telecommunications companies. The objective of this project is to predict whether a customer will stay with the company or churn, which can help identify customers who may need retention efforts.

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Target variable:** `Churn`

Target classes:

- `No` → Customer stays
- `Yes` → Customer churns

## Dataset

The project uses the Telco Customer Churn dataset, downloaded via `kagglehub` from the `blastchar/telco-customer-churn` dataset.

The dataset contains:

- 7,043 customers
- 21 columns

The `customerID` column was removed since it is an identifier and carries no predictive value. The final model uses 19 features.

## Data Cleaning

`TotalCharges` was originally stored as an object/string column. Some rows had blank values in this column, all corresponding to customers with zero tenure. The column was converted to numeric, and the resulting missing values were filled with `0`. No rows were removed.

## Exploratory Data Analysis

EDA covered the target distribution, numerical feature distributions, and churn behavior across customer groups. Notable observations:

- Customers with shorter tenure showed higher churn rates.
- Higher monthly charges were associated with higher churn rates.
- Month-to-month contracts showed substantially higher churn than one- or two-year contracts.
- Electronic check users showed a relatively high churn rate.
- Customers without certain add-on services (security, backup, tech support, etc.) showed higher churn rates.

These are patterns observed in the data and do not imply causation.

## Preprocessing

The dataset contains both numerical and categorical features, handled with a scikit-learn `ColumnTransformer`:

- **Numerical features** (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardized using `StandardScaler`.
- **Categorical features** (the remaining 16 columns, e.g. `gender`, `Contract`, `PaymentMethod`, `InternetService`, etc.) are transformed using `OneHotEncoder` with `handle_unknown="ignore"`.

Scaling matters here specifically because KNN is a distance-based algorithm: without it, features with larger numeric ranges would dominate the distance calculation.

## Baseline Model

The initial KNN model used `n_neighbors = 5`. Baseline test performance:

| Metric | Score |
|---|---:|
| Accuracy | 75.73% |
| Precision | 54.19% |
| Recall | 55.35% |
| F1-Score | 54.76% |
| ROC-AUC | 78.83% |

## Hyperparameter Tuning

The number of neighbors was tuned with 5-fold cross-validation, testing values from 1 to 30 using ROC-AUC as the scoring metric. This selected `n_neighbors = 30` (CV ROC-AUC ≈ 83.61%).

`weights` and `p` were then each tested at `n_neighbors = 30` using the same 5-fold CV setup:

- `weights`: `uniform` (83.61%) outperformed `distance` (81.83%).
- `p`: `p=2` (83.61%) outperformed `p=1` (83.47%).

Final configuration: `n_neighbors = 30`, `weights = "uniform"`, `p = 2`.

## Final Model Performance

Performance on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 78.78% |
| Precision | 61.13% |
| Recall | 55.08% |
| F1-Score | 57.95% |
| ROC-AUC | 83.10% |

Confusion matrix:

```
[[904, 131],
 [168, 206]]
```

- True Negatives: 904
- False Positives: 131
- False Negatives: 168
- True Positives: 206

## Error Analysis

Out of 1,409 test samples, the final model produced 1,110 correct predictions, 168 false negatives, and 131 false positives.

Comparing false negatives (churners the model missed) to true positives (churners correctly identified), false negatives had on average:

- Longer tenure (24.7 vs 9.8 months)
- Lower monthly charges ($64.8 vs $79.3)
- Higher total charges ($2,082 vs $845)

Both false negatives and false positives were heavily concentrated among month-to-month contracts and, especially for false positives, electronic check payment users. These are patterns observed in the model's errors and are not necessarily causal.

## Machine Learning Pipeline

The final model is saved as a single scikit-learn `Pipeline` combining the `ColumnTransformer` and the tuned KNN classifier, so the Streamlit application can pass in raw customer data without applying preprocessing manually.

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app.py`). It collects customer information through a form and returns the churn prediction along with the predicted churn probability.

### Run the application

From the project root:

```
streamlit run app/app.py
```

## Project Structure

```
Customer-Churn-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── knn_churn_pipeline.pkl
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
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

## Learning Objective

This project was built as part of a practical machine learning learning path. The goal was not only to train a KNN model but to practice the full workflow end to end: problem framing, data cleaning, EDA, preprocessing, model training, evaluation, error analysis, hyperparameter tuning, pipeline packaging, and deployment.