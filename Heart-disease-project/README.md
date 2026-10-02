# Heart Disease Prediction

## Project Overview

This project builds an end-to-end machine learning solution to predict the presence of heart disease from clinical patient features.

The project uses Logistic Regression for binary classification. The workflow covers data understanding, cleaning, exploratory data analysis, preprocessing, hyperparameter tuning, threshold-free evaluation with bootstrap confidence intervals, error analysis, model saving, and Streamlit deployment.

> **Disclaimer:** This project is for educational purposes only and is not a medical diagnostic tool.

## Problem Statement

The goal of this project is to predict whether a patient has heart disease based on clinical features.

This is a:

- Supervised learning problem
- Binary classification problem

The target variable represents whether heart disease is present (`1`) or absent (`0`).

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Logistic Regression
- **Target variable:** `target`

Target classes:

- `0` → No heart disease
- `1` → Heart disease

## Dataset

The project uses the processed Cleveland heart disease dataset, downloaded via `kagglehub` from the `mdzawharulislam/uci-processedclevelanddata` dataset and stored locally as `heart_disease_data.csv`.

The dataset contains:

- 303 rows and 13 input features before cleaning
- No missing values
- One exact duplicate row

### Features

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

### Known data quirk

In the original UCI documentation, `ca` should only take values 0–3 and `thal` should take the values 3, 6, or 7. In this version of the dataset, `ca` includes a small number of `4`s (4 rows) and `thal` includes a small number of `0`s (2 rows), which most likely correspond to the original dataset's missing-value markers having been remapped into the valid-looking range. Given how few rows are affected, no special handling was applied: since both `ca` and `thal` are treated as categorical features, one-hot encoding naturally absorbs them as just another category.

## Data Cleaning

The one duplicate row is dropped, leaving 302 rows for analysis and modeling. This step was missing from the project's first version.

## Exploratory Data Analysis

The dataset was explored using:

- Target distribution (54.5% heart disease vs. 45.5% no heart disease — close to balanced)
- Boxplots of the five numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) against the target
- Cross-tabulations of the categorical features (`cp`, `exang`, `slope`, `ca`, `thal`, `sex`) against the target
- A correlation heatmap across all features

Notable observations:

- `cp` (chest pain type), `exang` (exercise-induced angina), `oldpeak`, `thalach`, and `ca` all show a clear split between the two classes.
- Patients with no chest pain (`cp = 0`) have a *lower* observed disease rate (27%) than patients with any type of chest pain (69–82%). This is counter-intuitive at first glance but clinically sensible: chest pain is often what brings a patient in for evaluation in the first place.
- `thalach` (max heart rate) is higher, and `oldpeak` (ST depression) is lower, in patients with heart disease.
- `cp` has the strongest correlation with the target among all features (≈ 0.43), followed by `exang` (≈ -0.44), `oldpeak` (≈ -0.43), `thalach` (≈ 0.42), `ca` (≈ -0.41), and `slope` (≈ 0.34).

## Train/Test Split

The dataset is split into:

- 80% training data (241 patients)
- 20% test data (61 patients)

A stratified split is used to preserve the target class distribution.

## Preprocessing

Preprocessing is implemented with a scikit-learn `ColumnTransformer`, combined with the model inside a single `Pipeline` so every training and evaluation step — including cross-validation — fits preprocessing fresh on each training fold, avoiding leakage from validation data into the transformers.

- **Numerical features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are standardized using `StandardScaler`.
- **Categorical features** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) are transformed using `OneHotEncoder`.

## Model Development

The model was developed iteratively:

1. **Baseline Logistic Regression** (`C = 1.0`), evaluated with 5-fold stratified cross-validation on the training set: accuracy 85.90% ± 4.00%, ROC-AUC 90.81% ± 3.47%.
2. **Hyperparameter tuning:** a `GridSearchCV` over `C ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5}`, using the same 5-fold stratified cross-validation, scored on ROC-AUC.

The search selected `C = 0.2`, with a cross-validated ROC-AUC of 91.22%.

## Final Model Performance

The tuned model was evaluated once, on the held-out test set, after hyperparameter selection was finalized using only cross-validation on the training set:

| Metric | Score |
|---|---:|
| Accuracy | 86.89% |
| Precision | 85.71% |
| Recall | 90.91% |
| F1-Score | 88.24% |
| ROC-AUC | 89.61% |

### Confusion Matrix

- True Negatives: 23
- False Positives: 5
- False Negatives: 3
- True Positives: 30

The model correctly classified 53 out of 61 test samples.

### Bootstrap Confidence Intervals

95% bootstrap confidence intervals on the test set:

| Metric | 95% Bootstrap CI |
|---|---|
| Accuracy | [0.787, 0.951] |
| Precision | [0.727, 0.969] |
| Recall | [0.800, 1.000] |
| ROC-AUC | [0.811, 0.968] |

## Error Analysis

The final model misclassified 8 of the 61 test patients.

- Four of the eight errors are confident false positives (predicted probability above 0.7 for a patient without heart disease): patients 193, 266, 301, and 285. All four share `exang = 0` and `thal = 2`, suggesting the model leans heavily on this specific combination as a disease signal, even when other features (such as `ca`) point the other way for some of these patients.
- The remaining errors are a mix of lower-confidence false positives and false negatives, without as clear a shared pattern.

### Feature Importance

The largest logistic regression coefficients (on standardized/encoded features) include `cp_0` (strongly negative — no chest pain lowers predicted risk), `ca_0` (positive — no colored vessels, consistent with the `cp` pattern above), `thal_2` (positive) and `thal_3` (negative), `oldpeak` (negative), and `sex` (male lowers predicted risk relative to female in this dataset). These align with the relationships seen during EDA.

## Machine Learning Pipeline

The final model is saved as a single scikit-learn `Pipeline` combining the `ColumnTransformer` and the tuned Logistic Regression model, so preprocessing and prediction are handled consistently for both training and inference. The saved pipeline was verified to produce identical predictions after being reloaded with `joblib`.

## Model Saving

The final trained pipeline is saved with `joblib` to:

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

4. Run the notebook

Open `notebooks/heart_disease_analysis.ipynb`. It downloads the dataset using `kagglehub`, performs the complete analysis, tunes and evaluates the model, and regenerates the saved model artifact.

## Key Findings and Limitations

- Removing the single duplicate row and selecting the hyperparameter purely from cross-validation (never touching the test set until final evaluation) gives a test ROC-AUC of 89.6%, slightly lower than this project's first version (91.1%). This is expected and considered a more trustworthy number: the earlier version reused the same test split throughout development, which risks a mild, hard-to-detect optimistic bias.
- `ca` and `thal` contain a handful of rows with values outside their originally documented ranges, most likely remapped missing-value markers from the source data; treating both as categorical absorbs this without special handling, given how few rows are affected.
- The model's most confident mistakes (4 of 8 test errors) all share the same `exang = 0`, `thal = 2` combination, suggesting a specific blind spot rather than generally noisy predictions.
- With only 302 patients and 61 in the test set, metric estimates carry real uncertainty, reflected in the bootstrap confidence intervals above.

## Disclaimer

This project is developed for educational and machine learning practice purposes. It should not be used as a medical diagnostic system or as a substitute for professional medical advice.