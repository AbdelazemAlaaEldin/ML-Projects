# Breast Cancer Classification using SVM

## Project Overview

This project builds an end-to-end machine learning solution to classify breast tumor samples as malignant or benign, based on measurements computed from digitized images of a fine needle aspirate.

The project uses a Support Vector Machine (SVM) for binary classification. The workflow covers data understanding, exploratory data analysis, baseline kernel comparison, hyperparameter tuning, final evaluation with bootstrap confidence intervals, error analysis, model saving, and Streamlit deployment. All model selection is done with cross-validation on the training set only, and the test set is used once at the end.

> **Disclaimer:** This project is for educational purposes only and is not a medical diagnostic tool.

## Problem Statement

The goal of this project is to predict whether a breast tumor is malignant or benign from 30 numerical features describing cell nuclei.

The more serious mistake in this setting is missing a malignant case (predicting benign for a malignant tumor). This shapes the choice of the tuning metric and the way final metrics are reported.

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Support Vector Machine (`SVC`)
- **Target variable:** `target` (0 = Malignant, 1 = Benign)
- **Positive class for reported metrics:** Malignant (`pos_label=0`)
- **Tuning metric:** `recall_macro`

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset, loaded from `sklearn.datasets.load_breast_cancer` and saved locally as a CSV in `data/raw/`.

The dataset contains:

- 569 observations and 31 columns (30 features and the target)
- 30 numerical features: "mean", "standard error", and "worst" versions of 10 cell characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- Target distribution: 357 benign (62.74%) and 212 malignant (37.26%)

## Data Cleaning

- Missing values: 0
- Duplicated rows: 0

No cleaning was required.

## Exploratory Data Analysis

The features are on very different scales (for example, `mean area` reaches 2501.0 while `mean smoothness` is around 0.1), which motivates scaling inside the pipeline.

**Correlation with the target** (0 = Malignant, 1 = Benign). The strongest correlations are all negative, meaning larger values are associated with malignancy:

| Feature | Correlation with target |
|---|---:|
| worst concave points | -0.7936 |
| worst perimeter | -0.7829 |
| mean concave points | -0.7766 |
| worst radius | -0.7765 |

Features with almost no linear correlation with the target: `smoothness error` (0.0670), `mean fractal dimension` (0.0128), `texture error` (0.0083), and `symmetry error` (0.0065).

**Correlation between features.** There are 21 feature pairs with absolute correlation above 0.9. The highest are:

| Feature 1 | Feature 2 | Absolute correlation |
|---|---|---:|
| mean radius | mean perimeter | 0.9979 |
| worst radius | worst perimeter | 0.9937 |
| mean radius | mean area | 0.9874 |

No features were removed because of this correlation (see Key Findings and Limitations).

**Class means of the four most correlated features** (full dataset):

| Feature | Benign | Malignant |
|---|---:|---:|
| worst concave points | 0.0744 | 0.1822 |
| worst perimeter | 87.0059 | 141.3703 |
| mean concave points | 0.0257 | 0.0880 |
| worst radius | 13.3798 | 21.1348 |

The two classes are clearly separated on these features, with some overlap at the edges.

## Preprocessing

All 30 features are numerical. Preprocessing consists of `StandardScaler`, placed inside a single scikit-learn `Pipeline` together with the SVM. The scaler is therefore fitted only on the training portion of each cross-validation fold, so no information from validation data leaks into the transformation. Scaling matters for SVM because it is margin and distance based, and features on large numeric scales would otherwise dominate.

## Train/Test Split

The data is split 80/20 with stratification on the target (`random_state=42`):

| Set | Samples | Benign | Malignant |
|---|---:|---:|---:|
| Train | 455 | 285 | 170 |
| Test | 114 | 72 | 42 |

The test set is not touched until tuning is complete.

## Baseline Models

Three SVM kernels were compared with 5-fold stratified cross-validation on the training set only, each inside a pipeline with `StandardScaler` (mean ± std across folds):

| Model | Accuracy | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---:|---:|---:|---:|
| Linear | 0.9648 ± 0.0162 | 0.9612 ± 0.0200 | 0.9623 ± 0.0176 | 0.9947 ± 0.0073 |
| RBF | 0.9692 ± 0.0146 | 0.9659 ± 0.0189 | 0.9669 ± 0.0160 | 0.9956 ± 0.0048 |
| Polynomial (degree 3) | 0.8967 ± 0.0330 | 0.8618 ± 0.0442 | 0.8805 ± 0.0408 | 0.9940 ± 0.0035 |

The polynomial kernel has a ROC-AUC (0.9940) close to the other two but much lower accuracy and macro recall. Its ranking of samples is good, but the default decision threshold is poorly placed for it. ROC-AUC alone would have hidden this, which is one reason `recall_macro` was used as the tuning metric.

## Hyperparameter Tuning

A `GridSearchCV` (5-fold stratified cross-validation, scored on `recall_macro`, fitted on the training set only) searched jointly over the kernel and its relevant hyperparameters:

- Linear: `C` in {0.01, 0.1, 1, 10, 100}
- RBF: `C` in {0.01, 0.1, 1, 10, 100}, `gamma` in {0.001, 0.01, 0.1, 1}
- Polynomial: `C` in {0.1, 1, 10}, `degree` in {2, 3, 4}, `gamma` in {"scale", 0.01, 0.1}

Best parameters: RBF kernel, `C = 10`, `gamma = 0.01`, with a cross-validated `recall_macro` of 0.9712.

Top configurations:

| Rank | Parameters | Mean recall_macro | Std |
|---:|---|---:|---:|
| 1 | RBF, C=10, gamma=0.01 | 0.9712 | 0.0207 |
| 2 | Linear, C=0.1 | 0.9700 | 0.0152 |
| 3 | RBF, C=100, gamma=0.001 | 0.9641 | 0.0155 |
| 3 | RBF, C=1, gamma=0.01 | 0.9641 | 0.0185 |
| 5 | RBF, C=10, gamma=0.001 | 0.9630 | 0.0271 |

The best RBF and best Linear configurations differ by 0.0012, which is much smaller than either standard deviation. The RBF model was kept because it had the highest mean cross-validation score, a rule fixed before looking at the test set.

The final model is `StandardScaler` followed by `SVC(C=10, gamma=0.01, random_state=42)`.

## Final Model Performance

The test set was evaluated once, with the tuned pipeline. Metrics for the malignant class use `pos_label=0`.

| Metric | Score | 95% Bootstrap CI |
|---|---:|---|
| Accuracy | 0.9825 | [0.9561, 1.0000] |
| Malignant precision | 0.9762 | [0.9200, 1.0000] |
| Malignant recall | 0.9762 | [0.9189, 1.0000] |
| Malignant F1 | 0.9762 | [0.9362, 1.0000] |
| Recall (macro) | 0.9812 | [0.9490, 1.0000] |
| ROC-AUC | 0.9977 | [0.9910, 1.0000] |

Confidence intervals come from 1000 bootstrap resamples of the test set (`np.random.RandomState(42)`); all 1000 resamples were valid.

Confusion matrix:

| | Pred Malignant | Pred Benign |
|---|---:|---:|
| True Malignant | 41 | 1 |
| True Benign | 1 | 71 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.9762 | 0.9762 | 0.9762 | 42 |
| Benign | 0.9861 | 0.9861 | 0.9861 | 72 |
| Accuracy | | | 0.9825 | 114 |
| Macro avg | 0.9812 | 0.9812 | 0.9812 | 114 |
| Weighted avg | 0.9825 | 0.9825 | 0.9825 | 114 |

The point estimates are high, but the intervals are wide because the test set contains only 42 malignant cases. The malignant recall interval starts at 0.9189.

## Error Analysis

The model made 2 errors on 114 test samples:

| Sample | True | Predicted | Decision score | Error type |
|---:|---|---|---:|---|
| 541 | Benign | Malignant | -0.2879 | False positive |
| 73 | Malignant | Benign | 0.9144 | False negative (malignant missed) |

Decision score distribution on the test set:

| True class | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Benign | 2.0646 | 1.0878 | -0.2879 | 1.1746 | 1.9725 | 2.8916 | 4.6178 |
| Malignant | -2.9260 | 1.4881 | -5.5392 | -3.9145 | -3.1142 | -1.9435 | 0.9144 |

The missed malignant case (score 0.9144) sits below the 25th percentile of the benign scores (1.1746), so it is on the weak edge of the benign side rather than deep inside it.

**Missed malignant case versus class means.** Comparing the false negative with the correctly caught malignant cases and the correctly identified benign cases (test set only):

| Feature | FN (missed malignant) | True malignant (caught) | True benign | FN position (0 = malignant, 1 = benign) |
|---|---:|---:|---:|---:|
| worst concave points | 0.1383 | 0.1808 | 0.0805 | 0.4235 |
| worst perimeter | 110.3000 | 146.7395 | 88.4085 | 0.6247 |
| mean concave points | 0.0507 | 0.0905 | 0.0287 | 0.6443 |
| worst radius | 16.5700 | 21.9417 | 13.5793 | 0.6424 |

For 22 of the 30 features, the missed case is closer to the benign mean than to the malignant mean. Among the top four features, only `worst concave points` is closer to the malignant side. The case looks like a malignant tumor with smaller size measurements than typical but relatively high concave points. This comes from a single sample, so it is documented as an individual case and not as a general pattern.

### Uncertainty zone

To flag predictions near the decision boundary, the margin was chosen from out-of-fold decision scores on the training set only (`cross_val_predict`, 11 out-of-fold errors out of 455), and the test set was used only to check the choice:

| Margin | Train zone size | Train errors in zone | Train errors outside | Test zone size | Test errors in zone | Test errors outside |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 26 | 8 | 3 | 7 | 1 | 1 |
| 0.75 | 42 | 10 | 1 | 10 | 1 | 1 |
| 1.00 | 56 | 10 | 1 | 16 | 2 | 0 |
| 1.25 | 80 | 10 | 1 | 28 | 2 | 0 |

A margin of 0.75 was chosen: it is the smallest margin that covers almost all out-of-fold errors, and widening it adds no further errors. At 0.75, the test set has 10 samples in the zone with 1 error inside (sample 541), while the missed malignant case (sample 73, score +0.9144) falls outside. One out-of-fold training error remains outside the zone even at 1.25. The zone is therefore a warning, not a guarantee.

A margin of 1.00 would have covered both test errors, but choosing it would have meant tuning the heuristic on the test errors themselves.

## Machine Learning Pipeline

The final model is a single scikit-learn `Pipeline` combining `StandardScaler` and the tuned `SVC`, so raw feature values can be passed in directly without manual scaling. The project has no custom transformers, so no `src/` module is needed.

## Model Saving

The trained pipeline is saved with `joblib` to:

```
models/breast_cancer_svm_pipeline.pkl
```

After saving, the pipeline was reloaded with `joblib.load` and its test-set predictions were confirmed identical to the original model (`Identical predictions: True`).

A small metadata file used by the app is saved to:

```
models/app_meta.json
```

It contains the median of each of the 30 features on the training set (used as the app's default input values) and the uncertainty margin (0.75).

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app/app.py`). The model and metadata are loaded once with `@st.cache_resource`. The 30 input features are grouped into three tabs (mean, standard error, and worst features), with defaults set to the training-set median of each feature instead of zeros. The app builds a single-row DataFrame, passes it to the saved pipeline, and shows the predicted class and the raw SVM decision score.

Since the model was not trained with probability estimates, the app reports the decision score rather than a probability: a negative score favors malignant and a positive score favors benign.

When the absolute decision score is below 0.75, the app shows a caption that the sample is close to the decision boundary and that the prediction should be treated as uncertain. An "About this model" section in the app summarizes the test performance, the bootstrap intervals, and the main limitation. The app also shows a warning that it is an educational project and not a medical diagnosis.

### Run the application

From the project root:

```
streamlit run app/app.py
```

## Project Structure

```
Breast-cancer-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── breast_cancer.csv
│
├── models/
│   ├── breast_cancer_svm_pipeline.pkl
│   └── app_meta.json
│
├── notebooks/
│   └── breast_cancer_analysis.ipynb
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

## Key Findings and Limitations

**Differences from the previous version of this project, and why:**

- The previous version compared kernels and ran a manual `C` and `gamma` sweep on the 80/20 train/test split, which meant the test set was used before tuning. In this version, the kernel comparison and the tuning use cross-validation on the training set only, and the test set is evaluated once at the end.
- The previous version tuned on ROC-AUC. This version tunes on `recall_macro`, because missing a malignant case is the costly error, and ROC-AUC hid the threshold problem of the polynomial kernel in the baseline comparison.
- Bootstrap 95% confidence intervals were added for all final test metrics. The previous version reported point estimates only.
- Final metrics for the malignant class are reported with an explicit `pos_label=0`.
- The error analysis now compares the missed malignant case against class means and examines the decision-score distribution, instead of only listing the two errors.
- The app now uses `@st.cache_resource`, has no emojis, uses training-set medians as input defaults instead of zeros, includes an "About this model" section, and shows a dynamic uncertainty caption.
- The tuning result is the same as before: RBF with `C = 10` and `gamma = 0.01`.

**Findings:**

- The tuned RBF model reached an accuracy of 0.9825 and a malignant recall of 0.9762 on the test set (41 of 42 malignant cases caught).
- The best RBF and best Linear configurations are statistically indistinguishable in cross-validation (0.9712 versus 0.9700, with standard deviations of 0.0207 and 0.0152). The data does not strongly favor either kernel.
- Radius, perimeter, and area features are almost perfectly correlated (21 pairs above 0.9). They were kept because SVM works on the margin and does not need independent features, and removing features should be decided by cross-validation, not assumed.

**Limitations:**

- The test set has only 42 malignant cases. One additional error moves the malignant recall by roughly 2.4 percentage points, and the bootstrap interval for malignant recall is [0.9189, 1.0000], so the true recall may be closer to 92% than to 97.6%.
- The missed malignant case looks like a smaller malignant tumor with relatively high concave points. This is a single sample and cannot be generalized to a pattern.
- The uncertainty zone (|score| < 0.75) is a warning and not a guarantee: the missed malignant case in the test set fell outside it, and one out-of-fold training error also falls outside even at a margin of 1.25.
- The dataset is small (569 samples) and comes from a single source, so the results may not transfer to other populations or imaging setups.
- The model is for educational use only and must not be used for medical diagnosis.

## Learning Objective

This project was built as part of a practical machine learning learning path, focused on understanding Support Vector Machines through a complete real-world problem.

The project focuses on understanding:

- How the kernel choice (linear, RBF, polynomial) and the `C` and `gamma` hyperparameters affect an SVM, and how to search over them jointly with `GridSearchCV` using a list of parameter grids
- Why feature scaling is essential for SVM and why it must live inside the pipeline
- Why the tuning metric should reflect the cost of errors (`recall_macro` here) and how ROC-AUC alone can hide a poorly placed decision threshold
- How to interpret the SVM decision function as a distance from the boundary, and how to choose an uncertainty margin from out-of-fold scores without using the test set
- How to report test performance honestly with bootstrap confidence intervals when the minority class is small
- How this compares to the previously used KNN, Logistic Regression, and Naive Bayes models on a classification task