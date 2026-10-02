# Stroke Prediction

## Project Overview

This project builds an end-to-end machine learning solution to predict whether a patient is likely to have had a stroke based on demographic and health information.

The project uses Bagging with Decision Trees for binary classification. The workflow covers data understanding, cleaning, exploratory data analysis, preprocessing, model training, hyperparameter tuning, threshold selection, error analysis, model packaging, and Streamlit deployment.

> **Disclaimer:** This project is for educational purposes only. The model is a screening aid trained on a small public dataset. It is not a diagnostic tool and must not be used for clinical decisions.

## Problem Statement

The goal of this project is to predict whether a patient is likely to have had a stroke based on demographic and health-related information.

This is a:

- Supervised learning problem
- Binary classification problem

Because stroke cases are relatively rare in the dataset, recall is especially important when evaluating the model.

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Bagging (Bootstrap Aggregating)
- **Base estimator:** Shallow Decision Trees
- **Target variable:** `stroke`

Target classes:

- `0` → No Stroke
- `1` → Stroke

## Dataset

The project uses the Stroke Prediction Dataset from Kaggle, downloaded using `kagglehub`.

Dataset source:

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

The dataset contains:

- 5,110 patients
- 12 columns
- Demographic and health-related information
- `stroke` as the target variable

The dataset is highly imbalanced, with only 249 stroke cases in the full dataset, representing about 4.9% of all observations.

The notebook removes:

- One row where `gender = "Other"`
- The `id` column, since it is an identifier rather than a predictive feature

## Data Cleaning

The dataset contains missing values in the `bmi` column.

The missing BMI values are handled using median imputation inside the preprocessing pipeline.

The preprocessing is fitted only on the training data to avoid data leakage.

The cleaned feature set contains:

- Numerical features such as `age`, `avg_glucose_level`, and `bmi`
- Binary health indicators such as `hypertension` and `heart_disease`, passed through unchanged since they are already 0/1
- Categorical features such as `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status`

## Exploratory Data Analysis

The notebook examines the distribution of the target and the relationship between demographic and health variables and stroke.

The analysis includes:

- Target distribution
- Numerical feature distributions
- Boxplots
- Categorical feature analysis
- Stroke rates across different health conditions
- Relationships between age, glucose level, BMI, hypertension, heart disease, smoking status, and stroke

Some notable observations include:

- Stroke cases are much less frequent than non-stroke cases.
- Patients with hypertension have a higher observed stroke rate than patients without hypertension (13.3% vs 4.0%).
- Patients with heart disease also show a higher observed stroke rate (17.0% vs 4.2%).
- Age is strongly related to the observed stroke outcome, with the median age of stroke patients far higher than non-stroke patients.
- Glucose level and BMI were also examined across the two target classes; glucose shows a clearer separation than BMI.

## Preprocessing

The preprocessing is implemented inside a scikit-learn `Pipeline`, combined with the model so every training and evaluation step (including cross-validation) fits preprocessing fresh on each training fold, avoiding leakage from validation data into the transformers.

The pipeline includes:

- Median imputation for missing `bmi` values
- One-hot encoding for categorical features
- Passing the already-binary `hypertension` and `heart_disease` columns through unchanged

## Train/Test Split

The dataset is divided using a stratified 80/20 train/test split.

Stratification is used because the target classes are highly imbalanced and the stroke class is relatively small.

The final test set contains:

- 1,022 patients
- 50 stroke cases

## Baseline Models

Several models were evaluated during the development process:

- Dummy baseline
- Plain Bagging
- Bagging with balanced shallow Decision Trees
- Tuned Bagging model

The dummy baseline demonstrates why accuracy alone is not useful for this problem.

Always predicting the majority class achieves about 95.1% accuracy while completely failing to identify any stroke case (0% recall).

## Model Training

The main machine learning approach is Bagging (Bootstrap Aggregating) using shallow Decision Trees.

The model was developed progressively:

1. Dummy baseline
2. Plain Bagging
3. Balanced Bagging with shallow Decision Trees (`class_weight="balanced"`, limited tree depth)
4. Hyperparameter tuning using randomized search
5. Final threshold selection

The tuning process used:

- 5-fold Stratified Cross-Validation
- PR-AUC as the main scoring metric
- Randomized hyperparameter search over the number of estimators, bootstrap sample/feature fractions, and the base tree's depth and minimum leaf size

## Decision Threshold Selection

Instead of automatically using the default classification threshold of `0.5`, the final threshold was selected using out-of-fold training predictions from the tuned model.

The threshold was chosen by maximizing the **F2-score**, which gives more importance to recall than precision.

The final decision threshold was:

`0.447`

This threshold was selected without using the test set.

## Model Performance

The final model was evaluated on the held-out test set containing 1,022 patients and 50 stroke cases.

| Model | Recall | Precision | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Dummy baseline | 0.00 | 0.00 | 0.500 | 0.049 |
| Bagging (plain) | 0.00 | 0.00 | 0.767 | 0.122 |
| Bagging (balanced, default threshold) | 0.82 | 0.160 | 0.840 | 0.215 |
| **Bagging (tuned, threshold 0.447)** | **0.74** | **0.149** | **0.825** | **0.212** |

The tuned model's test performance is close to, and on some metrics slightly below, the simpler balanced-bagging baseline evaluated at the default threshold. This is consistent with the cross-validation results, where the tuning search's best PR-AUC (0.204) was within one standard deviation of the untuned balanced model's CV PR-AUC (0.177 ± 0.032) — see Key Findings below.

### Bootstrap Confidence Intervals

For the final model, the 95% bootstrap confidence intervals (computed by resampling the test set) were:

| Metric | 95% Bootstrap CI |
|---|---|
| Recall | [0.61, 0.86] |
| Precision | [0.11, 0.19] |
| PR-AUC | [0.14, 0.32] |
| ROC-AUC | [0.77, 0.88] |

These intervals reflect the uncertainty caused by the relatively small number of positive stroke cases.

## Error Analysis

The final model's errors were analyzed to understand where the model performs poorly.

Out of 1,022 test patients, the final model produced 760 true negatives, 212 false positives, 37 true positives, and 13 false negatives.

The analysis showed that:

- The model flags no patients under 40 in the test set, missing all 5 actual stroke cases in that age group.
- It flags approximately 92% of non-stroke patients over 70 (97 of 105).
- Age dominates the feature importance by a wide margin.
- Every one of the 13 false negatives had both `hypertension = 0` and `heart_disease = 0`. Their average age (47) was far below that of correctly identified stroke cases (75).

This indicates that the model behaves largely like an age-based screening filter rather than learning a broad and reliable representation of stroke risk: it reliably flags older patients and those with other risk factors, but is essentially blind to stroke in younger patients with no other flagged risk factors.

## Feature Importance

Permutation importance was used to investigate which features contribute most to the model's predictions, measured as the drop in PR-AUC when a feature is shuffled.

The analysis showed that:

- `age` is the dominant feature by a wide margin (importance ≈ 0.119), roughly nine times larger than the next most important feature.
- `heart_disease`, `avg_glucose_level`, `work_type`, and `bmi` contribute modestly.
- `gender`, `hypertension`, and `smoking_status` contribute little to nothing, with the latter two showing a slightly negative score, meaning shuffling them did not hurt (and marginally helped) model performance on this test set.

The strong dependence on age is an important limitation of the final model.

## Key Findings and Limitations

Several important observations were identified during the project:

- Accuracy is misleading because the majority class already represents about 95% of the dataset.
- Class imbalance makes recall, precision, PR-AUC, and ROC-AUC more informative than accuracy alone.
- The tuned model achieved approximately 74% recall, but precision was only about 15%, meaning most flagged patients are false alarms.
- Hyperparameter tuning did not provide a clear improvement over the balanced baseline: the tuned model's cross-validated PR-AUC (0.204) was within one standard deviation of the untuned balanced model's (0.177 ± 0.032), and its test-set performance was comparable to, not better than, the simpler model.
- Age dominates the model's feature importance.
- Only 249 stroke cases are available in the full dataset, including 50 in the test set.
- The relatively small number of positive cases results in considerable uncertainty around the evaluation metrics.
- The model score is not a calibrated probability and should not be interpreted as the probability that a patient will have a stroke.

## Machine Learning Pipeline

The final preprocessing and model workflow is packaged as a single scikit-learn pipeline.

The pipeline handles preprocessing and prediction consistently so that the Streamlit application can pass raw feature values directly to the trained model.

The complete trained artifact is saved with `joblib` together with the selected decision threshold.

## Streamlit Application

The trained model is served through a small Streamlit application.

The application allows the user to enter:

- Gender
- Age
- Hypertension
- Heart disease
- Ever married
- Work type
- Residence type
- Average glucose level
- BMI, or a "BMI unknown" checkbox, which passes a missing value through to the pipeline so it is median-imputed exactly as it was during training
- Smoking status

After submitting the information, the application displays:

- Whether the profile is flagged as higher risk
- The model score, labeled as a ranking score rather than a calibrated probability
- The decision threshold used by the model

An expandable "About this model" section summarizes the model's test performance, confidence intervals, and known limitations (its reliance on age and its tendency to miss strokes in younger patients without hypertension or heart disease) directly in the app.

The application clearly states that the model is an educational screening aid and not a medical diagnostic system.

### Run the Application

From the project root:

```
streamlit run app/app.py
```

## Model Saving

The final model artifact is saved using `joblib`:

```
models/stroke_bagging.joblib
```

The saved artifact contains:

- The trained Bagging model
- The selected decision threshold

The model should be loaded using the same scikit-learn version used during training.

## Project Structure

```
Stroke-prediction-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── healthcare-dataset-stroke-data.csv
│
├── models/
│   └── stroke_bagging.joblib
│
├── notebooks/
│   └── stroke_prediction_analysis.ipynb
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

Open `notebooks/stroke_prediction_analysis.ipynb`. It downloads the dataset using `kagglehub`, performs the complete analysis, trains the model, and regenerates the saved model artifact.

## Learning Objective

This project was built as part of a practical machine learning learning path, with the goal of understanding Bagging and ensemble learning through a complete real-world classification problem.

The project focuses on understanding:

- How Bagging works, and how Decision Trees can be combined using bootstrap aggregating
- How class imbalance affects classification, and why accuracy can be misleading for imbalanced datasets
- Why recall and PR-AUC are important in rare-event classification
- How to select a classification threshold based on the desired trade-off between recall and precision
- How to perform error analysis and interpret feature importance
- How to avoid data leakage by fitting preprocessing only within cross-validation folds
- How to package a trained model into a reusable pipeline and deploy it with Streamlit