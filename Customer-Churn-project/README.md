# Customer Churn Prediction using KNN

## Project Overview

This project builds an end-to-end machine learning solution to predict whether a telecommunications customer is likely to churn.

The project uses K-Nearest Neighbors (KNN) for binary classification. The workflow covers data understanding, cleaning, exploratory data analysis, leakage-free preprocessing, hyperparameter tuning, bootstrap confidence intervals, error analysis, model saving, and Streamlit deployment.

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
- No duplicate rows
- `Churn`, imbalanced: 73.5% stay vs. 26.5% churn

The `customerID` column is removed since it is an identifier and carries no predictive value, leaving 19 input features.

## Data Cleaning

`TotalCharges` was originally stored as text. 11 rows had blank values in this column, all corresponding to customers with `tenure = 0` (brand-new customers who have not yet been billed). The column is converted to numeric, and these 11 resulting missing values are filled with `0`, which is the correct value for a customer with no billing history yet, rather than a value to be estimated.

## Exploratory Data Analysis

EDA covered the target distribution, numerical feature distributions, and churn behavior across customer groups. Notable observations:

- Customers with shorter tenure and higher monthly charges show higher churn rates.
- Churn rate by contract type: 42.7% for month-to-month, 11.3% for one-year, and only 2.8% for two-year contracts.
- Churn rate by payment method: electronic check stands out at 45.3%, compared with 15–19% for the other three payment methods.
- Churn rate by internet service: 41.9% for fiber optic, 19.0% for DSL, and 7.4% for no internet service.
- Customers without add-on services (online security, tech support, online backup, device protection) consistently churn at roughly double the rate of customers who have them, while customers with no internet service at all (so these add-ons do not apply) have the lowest churn rate of the three groups in each comparison.

These are patterns observed in the data and do not imply causation.

## Train/Test Split

The dataset is divided using a stratified 80/20 train/test split.

The final split contains:

- 5,634 training customers
- 1,409 test customers

## Preprocessing

The preprocessing is implemented with a scikit-learn `ColumnTransformer`, combined with the model inside a single `Pipeline` so every training and evaluation step — including cross-validation and hyperparameter search — fits preprocessing fresh on each training fold, avoiding leakage from validation data into the transformers. This matters specifically for KNN: fitting the scaler on the full training set before cross-validation (as an earlier version of this project did) lets each fold's scaling reflect data it should not have seen yet, which can quietly bias a distance-based model's cross-validated scores.

- **Numerical features** (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardized using `StandardScaler`.
- **Categorical features** (the remaining 16 columns, e.g. `gender`, `Contract`, `PaymentMethod`, `InternetService`, etc.) are transformed using `OneHotEncoder` with `handle_unknown="ignore"`.

Scaling matters here specifically because KNN is a distance-based algorithm: without it, features with larger numeric ranges would dominate the distance calculation.

## Model Development

The model was developed iteratively:

1. **Baseline KNN** (`n_neighbors=5`), evaluated with 5-fold stratified cross-validation on the training set.
2. **Hyperparameter tuning:** a single joint `GridSearchCV` over `n_neighbors` (1–30), `weights` (`uniform`/`distance`), and `p` (1/2 — Manhattan/Euclidean distance), scored on ROC-AUC, searching all combinations together rather than tuning each parameter independently.

Since the target is text (`"Yes"`/`"No"`) rather than 0/1, precision, recall, and F1 require `pos_label="Yes"` to be specified explicitly; without it, scikit-learn raises `ValueError: pos_label=1 is not a valid label`.

Baseline cross-validated performance:

| Metric | Score |
|---|---:|
| Accuracy | 76.54% ± 0.50% |
| Precision | 56.10% ± 0.95% |
| Recall | 53.24% ± 1.98% |
| F1-Score | 54.62% ± 1.32% |
| ROC-AUC | 78.20% ± 0.52% |

The grid search selected `n_neighbors=30`, `weights="uniform"`, `p=2` (Euclidean distance), with a cross-validated ROC-AUC of 83.49%.

## Final Model Performance

Performance of the tuned model on the held-out test set:

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

### Bootstrap Confidence Intervals

95% bootstrap confidence intervals on the test set:

| Metric | 95% Bootstrap CI |
|---|---|
| Accuracy | [0.769, 0.808] |
| Precision | [0.560, 0.660] |
| Recall | [0.505, 0.599] |
| ROC-AUC | [0.809, 0.852] |

## Error Analysis

Out of 1,409 test samples, the final model produced 1,110 correct predictions, 168 false negatives, and 131 false positives.

Comparing false negatives (churners the model missed) to true positives (churners correctly identified):

- False negatives have on average much longer tenure (24.7 vs 9.8 months), lower monthly charges ($64.8 vs $79.3), and higher total charges ($2,082 vs $845).
- 73.2% of false negatives are on month-to-month contracts, compared with 96.2% of false positives.

This indicates the model is good at catching the "new, high-paying customer" churn profile that dominates the EDA, but tends to miss a different pattern: longer-tenured customers who still leave despite a long, accumulated relationship with the company. Both false negatives and false positives are heavily concentrated among month-to-month contracts, consistent with that contract type being the most volatile overall.

## Machine Learning Pipeline

The final model is saved as a single scikit-learn `Pipeline` combining the `ColumnTransformer` and the tuned KNN classifier, so the Streamlit application can pass in raw customer data without applying preprocessing manually. The saved pipeline was verified to produce identical predictions after being reloaded with `joblib`.

## Model Saving

The final trained pipeline is saved with `joblib` to:

```
models/knn_churn_pipeline.pkl
```

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

4. Run the notebook

Open `notebooks/customer_churn_analysis.ipynb`. It downloads the dataset using `kagglehub`, performs the complete analysis, tunes and evaluates the model, and regenerates the saved model artifact.

## Key Findings and Limitations

- The original version of this project fit `StandardScaler` on the full training set once, before running cross-validation and hyperparameter search, which let each fold's scaling see data it should not have — a mild form of data leakage for a distance-based model like KNN. Moving scaling inside the pipeline fixes this, and the final selected hyperparameters (`k=30`, uniform weights, Euclidean distance) and test-set performance turned out to be nearly identical either way, which suggests the leakage was not large enough here to meaningfully distort the result, but the leakage-free version is the trustworthy one to rely on going forward.
- Recall for the churn class (55%) means the model misses close to half of actual churners, and the kind of churner it is most likely to miss (long-tenured, lower-paying) is different from the kind it catches well (short-tenured, higher-paying).
- `pos_label` must be set explicitly for precision/recall/F1 whenever the target is text rather than 0/1; omitting it raises an error rather than silently giving a wrong answer, which is a useful safeguard rather than purely an inconvenience.

## Learning Objective

This project was built as part of a practical machine learning learning path, with the goal of understanding K-Nearest Neighbors and distance-based classification, and the specific ways cross-validation can be done incorrectly even when the final numbers look reasonable.

The project focuses on understanding:

- Why feature scaling matters specifically for distance-based algorithms like KNN
- How to keep preprocessing strictly inside cross-validation folds to avoid subtle leakage
- How to jointly tune multiple hyperparameters (`n_neighbors`, `weights`, `p`) rather than one at a time
- How to handle a text-valued binary target correctly when computing precision, recall, and F1
- How to use bootstrap confidence intervals to express uncertainty around test-set metrics
- How to perform error analysis by comparing the feature distributions of false negatives against true positives