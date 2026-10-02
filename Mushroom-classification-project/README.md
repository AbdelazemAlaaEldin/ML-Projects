# Mushroom Classification using Naive Bayes

## Project Overview

This project builds an end-to-end machine learning solution to classify mushrooms as edible or poisonous based on their physical characteristics.

The project uses Categorical Naive Bayes for binary classification. The workflow covers data understanding, cleaning, exploratory data analysis, leakage-free encoding, hyperparameter tuning, bootstrap confidence intervals, error analysis, model saving, and Streamlit deployment.

> **Disclaimer:** This project is for educational purposes only and is not a guide for identifying edible mushrooms in real life. Misclassifying a poisonous mushroom as edible can be dangerous.

## Problem Statement

The goal of this project is to predict whether a mushroom is edible or poisonous based on categorical physical features such as odor, gill size, and bruising.

This is a:

- Supervised learning problem
- Binary classification problem

Because a missed poisonous mushroom (a false negative) is far more dangerous than a mushroom mistakenly flagged as poisonous, the poisonous class is treated as the positive class throughout, and recall on that class is given particular attention.

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Categorical Naive Bayes (`CategoricalNB`)
- **Target variable:** `class`

Target classes:

- `e` → Edible
- `p` → Poisonous (treated as the positive class)

## Dataset

The project uses the Mushroom Classification dataset, downloaded via `kagglehub` from the `uciml/mushroom-classification` dataset.

The dataset contains:

- 8,124 observations
- 22 categorical input features plus the target
- No numerical features; every feature, including the target, is categorical
- No missing values represented as nulls, and no duplicate rows

Class distribution: 4,208 edible (51.8%) and 3,916 poisonous (48.2%).

`veil-type` was dropped because it has only a single value across the entire dataset and carries no predictive information, leaving 21 usable features.

## Data Cleaning

The `stalk-root` column uses `"?"` to represent missing values, affecting 2,480 rows (about 30.5% of the dataset) — the only column with missing values. These are treated as their own category rather than dropped or imputed: `"?"` is replaced with the label `"Unknown"`, and the model is left to learn from it like any other category, since a missing stalk root turns out to be informative about the target as well.

## Exploratory Data Analysis

The notebook examines the relationship between each categorical feature and the target using cross-tabulations.

Notable observations:

- `odor` is an extremely strong predictor: 7 of its 9 categories are perfectly or near-perfectly associated with a single class (e.g. `odor = a` or `odor = l` → 100% edible; `odor = c`, `f`, `m`, `p`, `s`, `y` → 100% poisonous). Only `odor = n` (no odor) is genuinely mixed (96.6% edible, 3.4% poisonous).
- `bruises` and `gill-size` also show a clear split: bruised mushrooms are 81.5% edible vs. 30.7% for unbruised ones; narrow gills are 88.5% poisonous vs. 30.1% for broad gills.
- The `stalk-root = "Unknown"` category is itself skewed toward poisonous (71.0%), compared with categories like `r` (0% poisonous) or `c` (7.9% poisonous).

## Preprocessing

All features are categorical, so preprocessing consists of encoding every remaining feature with `OrdinalEncoder` (`handle_unknown="use_encoded_value"`, `unknown_value=-1`), which is required because `CategoricalNB` expects non-negative integer-encoded categories.

The encoder is combined with the model inside a single scikit-learn `Pipeline`, so every training and evaluation step — including cross-validation and hyperparameter search — fits the encoder fresh on each training fold, rather than encoding the full training set once beforehand.

## Train/Test Split

The dataset is divided using a stratified 80/20 train/test split.

The final split contains:

- 6,499 training mushrooms
- 1,625 test mushrooms

## Model Development

The model was developed iteratively:

1. **Baseline Categorical Naive Bayes** (`alpha=1.0`, the default smoothing value), evaluated with 5-fold stratified cross-validation on the training set.
2. **Hyperparameter tuning:** a `GridSearchCV` over `alpha ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}`, scored on **macro-averaged recall** rather than ROC-AUC. ROC-AUC is already near-perfect even at the baseline (> 99.7%), so it does not meaningfully distinguish between candidate values of `alpha` or reflect the specific risk this project cares about; macro recall weights the poisonous and edible classes equally and pushes the search to favor configurations that catch poisonous mushrooms reliably.

Since the target is text (`"e"`/`"p"`) rather than 0/1, precision, recall, and F1 require `pos_label="p"` to be specified explicitly.

Baseline cross-validated performance (positive class = poisonous):

| Metric | Score |
|---|---:|
| Accuracy | 94.94% ± 1.09% |
| Precision | 99.33% ± 0.24% |
| Recall | 90.11% ± 2.15% |
| F1-Score | 94.48% ± 1.24% |
| ROC-AUC | 99.75% ± 0.06% |

The grid search selected `alpha = 0.001`, with a cross-validated macro recall of 99.28% — a large improvement over the baseline's 90.1% recall on the poisonous class specifically.

## Final Model Performance

Performance of the tuned model on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 99.08% |
| Precision | 99.10% |
| Recall | 98.98% |
| F1-Score | 99.04% |
| ROC-AUC | 99.96% |

Confusion matrix (rows = true, columns = predicted, order `[e, p]`):

```
[[835,   7],
 [  8, 775]]
```

- True Negatives: 835
- False Positives: 7
- False Negatives: 8
- True Positives: 775

### Bootstrap Confidence Intervals

95% bootstrap confidence intervals on the test set:

| Metric | 95% Bootstrap CI |
|---|---|
| Accuracy | [0.9858, 0.9957] |
| Precision | [0.9839, 0.9974] |
| Recall | [0.9819, 0.9962] |
| ROC-AUC | [0.9990, 1.0000] |

The narrow intervals, close to 1.0, indicate this performance is stable and not an artifact of a particular train/test split.

## Error Analysis

Out of 1,625 test samples, the final model produced 1,610 correct predictions, 8 false negatives, and 7 false positives.

- **All 8 false negatives** (poisonous mushrooms predicted edible) share the exact same `odor = n`, `bruises = t`, `stalk-root = b` combination (with `gill-size` split between broad and narrow).
- **All 7 false positives** (edible mushrooms predicted poisonous) share `odor = n`, `bruises = f`, `stalk-root = b`, `gill-size = n` — the same `odor`/`stalk-root` combination as the false negatives, but with `bruises = f` instead of `t`.

This shows every test-set mistake falls within one narrow slice of the feature space (`odor = n`, `stalk-root = b`), and that within that slice, `bruises` is the deciding factor the model leans on: it gets it right most of the time but is wrong in both directions right at that boundary. This is a more useful finding than "the errors are random" — it identifies a specific, narrow blind spot rather than general noise, which matters given that a false negative here is the costlier type of mistake.

## Machine Learning Pipeline

The final model is saved as a single scikit-learn `Pipeline` combining the `OrdinalEncoder` and the tuned `CategoricalNB` model, so raw mushroom feature values (after replacing `"?"` with `"Unknown"` and excluding `veil-type`) can be passed in directly. The saved pipeline was verified to produce identical predictions after being reloaded with `joblib`.

Unlike some of this project's earlier version, no custom transformer class is used here — only scikit-learn's own `OrdinalEncoder` — so there is no risk of the `__main__`-module pickling issue that affects pipelines with notebook-defined classes (see the House Price project for that issue and its fix).

## Model Saving

The final trained pipeline is saved with `joblib` to:

```
models/mushroom_categorical_nb_pipeline.pkl
```

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app.py`). It presents a dropdown for each of the dataset's 21 input features (cap shape, odor, gill size, stalk root, etc. — `veil-type` is excluded since it was dropped during training), builds a single-row DataFrame matching the training schema, and passes it directly into the saved pipeline. It displays:

- The predicted class (edible or poisonous)
- The model's confidence in that prediction
- The full probability breakdown for both classes
- A specific caution note when the input falls into the `odor = n`, `stalk-root = b` combination identified in the error analysis as the model's main blind spot

The app also shows a warning that the model is for educational purposes only and must not be used to judge whether a real mushroom is safe to eat.

### Run the application

From the project root:

```
streamlit run app/app.py
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

Open `notebooks/mushroom_classification_analysis.ipynb`. It downloads the dataset using `kagglehub`, performs the complete analysis, tunes and evaluates the model, and regenerates the saved model artifact.

## Key Findings and Limitations

- The original version of this project used a custom `MushroomPreprocessor` transformer (in a separate `preprocessing.py` module) and fit the `OrdinalEncoder` on the full training set once before cross-validation and hyperparameter search. The rebuilt version uses plain `OrdinalEncoder` inside the pipeline instead, both simplifying the code and removing the `__main__`-pickling risk, while reaching the same final hyperparameter (`alpha = 0.001`) and nearly identical test performance — the encoder's fit does not depend on the target, so the earlier leakage risk here was theoretical rather than something that visibly changed the result.
- Scoring the hyperparameter search on macro recall instead of ROC-AUC mattered here: ROC-AUC was already close to its ceiling at the baseline and would not have clearly distinguished between candidate `alpha` values or emphasized the poisonous-recall trade-off this project cares about most.
- Every test-set error, in both directions, is confined to mushrooms with `odor = n` and `stalk-root = b`. Outside that narrow combination, the model made zero mistakes on this test set.
- `pos_label` must be set explicitly for precision/recall/F1 whenever the target is text rather than 0/1.

## Learning Objective

This project was built as part of a practical machine learning learning path, focused on understanding Naive Bayes: its categorical variant, its independence assumption, how it handles purely categorical data, and how it compares to the previously used KNN and Logistic Regression models on a classification task.

The project also reinforced:

- Choosing a hyperparameter-search scoring metric that reflects the actual cost of different error types, not just the easiest metric to optimize
- Keeping encoding strictly inside cross-validation folds, and understanding when that matters more or less depending on the transformer involved
- Narrowing down the feature-space region where a model's errors cluster, rather than stopping at an aggregate error count