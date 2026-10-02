# Credit Card Default Prediction

## Project Overview

This project builds an end-to-end machine learning solution to predict whether a credit card client will default on their payment next month, based on demographics, credit limit, and six months of repayment history.

The project uses a Voting Classifier (Logistic Regression, KNN, and Decision Tree) for binary classification, combined with engineered repayment-behavior features. The workflow covers data understanding, cleaning, exploratory data analysis, preprocessing, model comparison, feature engineering, threshold selection, error analysis, model packaging, and Streamlit deployment.

> **Disclaimer:** This project is for educational purposes only. The model is a screening aid trained on a small public dataset from 2005. It is not a credit decision tool.

## Problem Statement

The goal of this project is to predict whether a credit card client will default on their payment next month, based on demographic information, credit limit, and repayment history.

This is a:

- Supervised learning problem
- Binary classification problem

Missing a defaulter (false negative) costs the lender the unpaid balance, while flagging a good client (false positive) costs a lost or restricted customer. The project therefore focuses on recall, precision, and PR-AUC together with the decision threshold, rather than on accuracy alone.

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Voting Classifier (soft voting over Logistic Regression, KNN, and Decision Tree)
- **Target variable:** `default`

Target classes:

- `0` → No default
- `1` → Default

## Dataset

The project uses the Default of Credit Card Clients dataset from Kaggle, downloaded using `kagglehub`.

Dataset source:

https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

The dataset covers credit card clients in Taiwan between April and September 2005 (UCI Machine Learning Repository, CC BY 4.0).

The dataset contains:

- 30,000 clients
- 25 columns, including an `ID` column and the target
- 35 duplicate rows (ignoring `ID`), leaving 29,965 clients after removal
- No missing values
- About 22.1% of clients defaulted

The notebook removes:

- The `ID` column, since it is an identifier rather than a predictive feature
- 35 duplicate rows

## Data Cleaning

- `PAY_0` is renamed to `PAY_1` for consistency with the other five repayment-status columns (`PAY_2` through `PAY_6`).
- `EDUCATION` contains undocumented codes `0`, `5`, and `6` alongside the documented `1` (graduate school), `2` (university), and `3` (high school); the undocumented codes are merged into category `4` ("others").
- `MARRIAGE` contains an undocumented code `0` alongside `1` (married) and `2` (single); it is merged into category `3` ("others").
- The repayment-status value `0` is not officially documented in the dataset; it is treated as "no delay" based on its behavior relative to the target.

## Exploratory Data Analysis

The notebook examines the distribution of the target, the relationship between demographic and financial variables and default, and correlations between features.

The analysis includes:

- Target distribution and default rate by `SEX`, `EDUCATION`, `MARRIAGE`, and `PAY_1`
- Boxplots of `LIMIT_BAL` and `AGE` against the target
- A full correlation heatmap, plus the correlation of every feature with the target
- Correlations among the six `BILL_AMT` columns

Notable observations:

- Repayment status (`PAY_1` through `PAY_6`) shows by far the strongest correlation with default, with `PAY_1` the strongest of the six (correlation ≈ 0.325).
- Default rate rises sharply with repayment delay: from about 12.8% for clients with no delay (`PAY_1 = 0`) up to 69–78% for clients 2 or more months late.
- `LIMIT_BAL` is mildly negatively correlated with default (≈ -0.154): clients with higher credit limits default somewhat less often.
- `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` all show weak correlations with the target compared with repayment status.
- The six `BILL_AMT` columns are highly correlated with each other (0.80–0.95), as expected since they represent a running balance over consecutive months.
- 1,930 rows have at least one negative bill amount, which likely reflects credit balances or overpayments rather than data errors.

## Preprocessing

The preprocessing is implemented inside a scikit-learn `Pipeline`, combined with the model so every training and evaluation step (including cross-validation) fits preprocessing fresh on each training fold, avoiding leakage from validation data into the transformers.

The pipeline includes:

- `StandardScaler` for the 20 numerical columns (credit limit, age, six repayment-status columns, six bill amounts, six payment amounts)
- One-hot encoding for the three categorical columns (`SEX`, `EDUCATION`, `MARRIAGE`)

## Train/Test Split

The dataset is divided using a stratified 80/20 train/test split.

The final split contains:

- 23,972 training clients, 5,993 test clients
- A default rate of 22.1% in both the training and test sets

## Baseline Models

Several models were compared with 5-fold stratified cross-validation on the training set, using the 23 raw features:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Dummy | 0.500 | 0.221 |
| Logistic Regression | 0.726 ± 0.011 | 0.501 ± 0.015 |
| KNN (k=15) | 0.740 ± 0.005 | 0.476 ± 0.008 |
| Decision Tree | 0.761 ± 0.005 | 0.518 ± 0.008 |

The dummy baseline (always predicting the majority class) demonstrates why accuracy alone is not useful for this problem: it would achieve roughly 78% accuracy while never identifying a single default.

## Model Training

The main machine learning approach is a Voting Classifier that combines Logistic Regression, KNN (k=15), and a shallow, class-balanced Decision Tree.

The model was developed progressively:

1. Individual base models compared via cross-validation (table above)
2. Hard and soft voting compared against the individual models
3. Soft voting selected as the stronger ensemble
4. Engineered repayment-behavior features added and compared against the raw-feature voting model
5. Final threshold selection on the chosen (feature-engineered) model

Hard vs. soft voting, 5-fold CV:

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Voting (hard) | 0.546 ± 0.017 | 0.513 ± 0.019 | 0.529 ± 0.010 | – | – |
| Voting (soft) | 0.504 ± 0.009 | 0.562 ± 0.012 | 0.532 ± 0.008 | 0.766 ± 0.007 | 0.535 ± 0.011 |

Soft voting outperformed the best individual model (Decision Tree) in PR-AUC and ROC-AUC on every one of the 5 CV folds (mean PR-AUC gain ≈ 0.018), and was carried forward as the base ensemble.

### Feature Engineering

Five features were engineered from the six months of repayment status and the most recent bill and payment amounts, implemented in a shared `src/features.py` module so the notebook and the Streamlit app use identical logic:

- `max_delay` — the worst (maximum) repayment-status value across the 6 months
- `n_months_late` — the number of months with a repayment-status value of 1 or more
- `mean_delay` — the average repayment delay, counting on-time months as 0
- `utilization` — the most recent bill amount divided by the credit limit
- `pay_ratio` — the most recent payment amount divided by the most recent bill amount, clipped to [0, 1], with a value of 1 when the most recent bill was zero or negative

Adding these features to the soft-voting pipeline and re-running the same 5-fold CV:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Voting (soft), raw features | 0.766 ± 0.007 | 0.535 ± 0.011 |
| **Voting (soft), engineered features** | **0.778 ± 0.007** | **0.542 ± 0.011** |

This gave a statistically significant ROC-AUC improvement over the raw-feature voting model (the feature-engineered model won on ROC-AUC in 5 of 5 CV folds) and was adopted as the final model.

## Decision Threshold Selection

Instead of automatically using the default classification threshold of `0.5`, the final threshold was selected using out-of-fold training predictions from the final (feature-engineered) model.

Two thresholds were computed by maximizing the F-beta score on these out-of-fold predictions:

- **F1 threshold** (balances precision and recall): `0.473`
- **F2 threshold** (weights recall more heavily): `0.265`

Both thresholds were selected without using the test set, and both are saved with the model so the application can offer either operating point.

## Model Performance

The final model was evaluated on the held-out test set containing 5,993 clients and 1,326 defaults.

At the F1 threshold (0.473):

| Metric | Value | 95% Bootstrap CI |
|---|---:|---|
| Precision | 0.513 | [0.487, 0.539] |
| Recall | 0.538 | [0.510, 0.566] |
| F1 | 0.526 | [0.502, 0.549] |
| ROC-AUC | 0.770 | [0.755, 0.785] |
| PR-AUC | 0.537 | [0.508, 0.568] |

At this threshold the model flags about 23% of clients.

At the F2 threshold (0.265): recall rises to about 0.85 but precision falls to about 0.31, and the model flags about 61% of clients, which makes it impractical as a standalone screening filter despite the higher recall.

Compared with the raw-feature voting model, the engineered-feature model's test-set improvement is statistically significant on ROC-AUC (95% CI for the difference: [0.0078, 0.0213], excludes zero) but not clearly distinguishable from zero on PR-AUC (95% CI: [-0.0027, 0.0149]).

Compared with a single-line rule ("flag any client at least 1 month late on their most recent payment"), which alone reaches precision 0.498, recall 0.503, F1 0.501, the final model's F1 advantage is small but real: 95% CI for the difference is [0.013, 0.036], excluding zero.

## Error Analysis

The final model's errors were analyzed to understand where the model performs poorly.

- **Most missed defaults come from clients with no visible delay.** The majority of false negatives have a recent repayment status of 0 or better; on recent payment history, these clients are largely indistinguishable from true negatives. With six months of history, a client who has paid on time and then defaults the next month is very hard to catch.
- **The model is driven mainly by repayment status.** For clients already 2 or more months late, the model flags nearly everyone, with precision close to the group's base default rate (little added value beyond the raw signal). It adds the most value for clients with no visible delay, where it still only recovers a small fraction of that group's defaults.
- **Recall falls as credit limit rises:** recall is noticeably higher for clients in the lowest credit-limit quartile than for clients in the highest, while precision stays comparatively stable across quartiles. This is likely partly explained by repayment status, which was not controlled for in this breakdown.

## Key Findings and Limitations

Several important observations were identified during the project:

- Soft voting beats the best single model (Decision Tree) in 5 of 5 CV folds on raw features, and the gap holds on the test set (paired bootstrap CI excludes zero), though the gain alone is modest (PR-AUC +0.025).
- Adding the engineered repayment-behavior features gives a further, statistically significant ROC-AUC gain over raw-feature voting; this is the configuration used for the deployed model. The bigger overall gain in this project came from feature engineering, not from the ensembling itself.
- The model is driven mainly by repayment status; a one-line rule based on it alone is already a strong baseline, and the full model only modestly outperforms it (F1 +0.025).
- Base models were not individually hyperparameter-tuned and are correlated with each other (especially the tree-based and linear models), which limits how much voting alone can add.
- Scores are not calibrated probabilities. The best operating threshold depends on the real cost of a missed default versus a false alarm, which is unknown here.
- Single bank, 2005 data, Taiwan only. Sex, marital status, and age are included for educational purposes; real-world credit decisions may legally restrict the use of such attributes.

## Machine Learning Pipeline

The final preprocessing, feature engineering, and model workflow is packaged as a single scikit-learn pipeline: a feature-engineering step (calling the shared `add_features` function), followed by the `ColumnTransformer` (scaling and one-hot encoding), followed by the soft `VotingClassifier`.

Because the feature-engineering step references a function imported from `src/features.py` rather than one defined inline in the notebook, both the training environment and the Streamlit app add the project root to `sys.path` so the pipeline can be unpickled and run correctly.

The pipeline handles preprocessing, feature engineering, and prediction consistently, so the Streamlit application can pass raw client data directly to the trained model.

## Model Saving

The final trained pipeline is saved with `joblib` to:

```
models/credit_default_voting.joblib
```

The saved artifact contains:

- The trained Voting Classifier pipeline (including the feature-engineering step)
- The F1 decision threshold
- The F2 decision threshold
- The list of raw input feature names expected by the pipeline

A round-trip check (reloading the saved file and comparing its predictions against the in-memory model) confirmed the saved pipeline reproduces the original predictions exactly.

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app.py`). It collects client information across three groups: basic client details (credit limit, age, sex, education, marital status), six months of repayment status, and six months each of bill and payment amounts. It builds a single-row DataFrame of the 23 raw input features and passes it directly into the saved pipeline, which performs feature engineering and preprocessing internally.

The user can choose between two operating points:

- **Balanced (F1):** the default, better-precision threshold
- **High recall (F2):** flags far more clients to catch more defaulters, at the cost of many more false alarms

The app displays whether the client is flagged, the model's raw score (explicitly labeled as a ranking score rather than a calibrated probability), and the threshold used. An expandable "About this model" section summarizes the model's test performance, its reliance on repayment status, and its main limitation (missing defaults from clients with no recent visible delay) directly in the app.

The application clearly states that the model is an educational screening aid and not a credit decision tool.

### Run the Application

From the project root:

```
streamlit run app/app.py
```

## Project Structure

```
Credit-card-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── UCI_Credit_Card.csv
│
├── models/
│   └── credit_default_voting.joblib
│
├── notebooks/
│   └── credit_card_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   └── features.py
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

Open `notebooks/credit_card_analysis.ipynb`. It downloads the dataset using `kagglehub`, performs the complete analysis, trains and compares the models, and regenerates the saved model artifact. The saved model must be loaded with the same scikit-learn version used during training.

## Learning Objective

This project was built as part of a practical machine learning learning path, with the goal of understanding ensemble voting and feature engineering through a complete real-world classification problem.

The project focuses on understanding:

- How hard and soft voting combine diverse base models, and when the combination helps
- How to compare an ensemble against its base models with cross-validation and bootstrap confidence intervals, rather than a single point estimate
- How to engineer features from raw time-series-like columns (repayment status and amounts over several months) without leaking information
- How to select a classification threshold based on the desired trade-off between recall and precision
- How to perform error analysis and sanity-check a model against a simple, interpretable rule
- How to share feature-engineering logic between a training notebook and a deployed application through a common module
- How to package a trained pipeline for deployment and serve it with Streamlit