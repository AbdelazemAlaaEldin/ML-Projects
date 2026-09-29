# Mushroom Classification using Naive Bayes

## Project Overview

This project builds an end-to-end machine learning solution to classify mushrooms as edible or poisonous based on their physical characteristics.

The project uses Categorical Naive Bayes for binary classification. The workflow covers data understanding, cleaning, EDA, encoding, model training, hyperparameter tuning, error analysis, pipeline packaging, and Streamlit deployment.

> **Disclaimer:** This project is for educational purposes only and is not a guide for identifying edible mushrooms in real life. Misclassifying a poisonous mushroom as edible can be dangerous.

## Problem Statement

The goal of this project is to predict whether a mushroom is edible or poisonous based on categorical physical features such as odor, gill size, and bruising.

This is a:

- Supervised learning problem
- Binary classification problem

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Categorical Naive Bayes (`CategoricalNB`)
- **Target variable:** `class`

Target classes:

- `e` → Edible
- `p` → Poisonous

## Dataset

The project uses the Mushroom Classification dataset, downloaded via `kagglehub` from the `uciml/mushroom-classification` dataset.

The dataset contains:

- 8,124 observations
- 22 categorical input features plus the target
- No numerical features; every feature, including the target, is categorical

Class distribution: 4,208 edible (51.8%) and 3,916 poisonous (48.2%).

`veil-type` was dropped because it is constant across the entire dataset and carries no predictive information, leaving 21 usable features.

## Data Cleaning

The `stalk-root` column uses `"?"` to represent missing values, affecting 2,480 rows (about 30.5% of the dataset). This is the only column with missing values. These were treated as their own category rather than dropped or imputed: `"?"` was replaced with the label `"Unknown"`, and the model was left to learn from it like any other category, since a missing stalk root turned out to be informative about the target as well.

## Exploratory Data Analysis

The notebook examines the relationship between each categorical feature and the target using cross-tabulations and count plots. Notable observations:

- `odor` is a very strong predictor. Certain odor categories are almost perfectly associated with one class.
- `gill-size` and `bruises` also show clear separation between edible and poisonous mushrooms.
- The `stalk-root` "Unknown" category is itself skewed toward poisonous mushrooms (about 71% poisonous).

## Preprocessing

All features are categorical, so preprocessing consists of:

- Replacing `"?"` in `stalk-root` with `"Unknown"`.
- Dropping the constant `veil-type` column.
- Encoding every remaining feature with `OrdinalEncoder` (`handle_unknown="use_encoded_value"`, `unknown_value=-1`), which is required because `CategoricalNB` expects non-negative integer-encoded categories.

These steps are implemented as a custom `MushroomPreprocessor` transformer (in a separate `preprocessing.py` module) so they can be reused consistently in both the training notebook and the final pipeline.

## Baseline Model

The initial `CategoricalNB` model (default `alpha`) achieved the following test performance:

| Metric | Score |
|---|---:|
| Accuracy | 94.58% |
| Precision | 99.01% |
| Recall | 89.66% |
| F1-Score | 94.10% |
| ROC-AUC | 99.72% |

Confusion matrix:

```
[[835,   7],
 [ 81, 702]]
```

## Hyperparameter Tuning

The Naive Bayes smoothing parameter `alpha` was tuned with 5-fold cross-validation over the values `[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`, using ROC-AUC as the scoring metric.

The search selected `alpha = 0.001`, with a cross-validated ROC-AUC of approximately 99.98%.

## Final Model Performance

Performance of the tuned model on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 99.08% |
| Precision | 99.10% |
| Recall | 98.98% |
| F1-Score | 99.04% |
| ROC-AUC | 99.96% |

Confusion matrix:

```
[[835,   7],
 [  8, 775]]
```

- True Negatives: 835
- False Positives: 7
- False Negatives: 8
- True Positives: 775

## Error Analysis

Out of 1,625 test samples, the final model produced 1,610 correct predictions, 8 false negatives, and 7 false positives.

The 8 false negatives (poisonous mushrooms predicted as edible) all shared the same characteristics: no odor, bruised (`bruises = t`), and the same `stalk-root` category, with gill sizes split between broad and narrow. This shows the small remaining error is concentrated in a specific, narrow combination of features rather than spread randomly across the dataset — a useful thing to know given that a false negative here is the costlier type of mistake.

## Machine Learning Pipeline

The final model is saved as a single scikit-learn `Pipeline` combining the `MushroomPreprocessor`, the `OrdinalEncoder`, and the tuned `CategoricalNB` model, so raw mushroom feature values can be passed in directly without preprocessing them manually. The saved pipeline was verified to produce identical predictions after being reloaded with `joblib`.

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app.py`), which imports the same `MushroomPreprocessor` used during training so the raw input is handled consistently with the notebook.

The app presents a dropdown for each of the dataset's 21 categorical features (cap shape, odor, gill size, stalk root, etc.), builds a single-row DataFrame matching the original column names, and passes it directly into the saved pipeline. It displays:

- The predicted class (edible or poisonous)
- The model's confidence in that prediction
- The full probability breakdown for both classes

The app also shows a warning that the model is for educational purposes only and must not be used to judge whether a real mushroom is safe to eat.

### Run the application

From the project root:

```
streamlit run app/app.py
```

## Model Saving

The final trained pipeline is saved with `joblib` to:

```
models/mushroom_categorical_nb_pipeline.pkl
```

## Project Structure

```
Mushroom-classification-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── mushrooms.csv
│
├── models/
│   └── mushroom_categorical_nb_pipeline.pkl
│
├── notebooks/
│   └── mushroom_classification_analysis.ipynb
│
├── preprocessing.py
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

This project was built as part of a practical machine learning learning path, focused on understanding Naive Bayes: its categorical variant, its independence assumption, how it handles purely categorical data, and how it compares to the previously used KNN and Logistic Regression models on a classification task.