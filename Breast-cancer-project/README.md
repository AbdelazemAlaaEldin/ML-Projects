# Breast Cancer Classification using SVM

## Project Overview

This project builds an end-to-end machine learning solution to classify breast cancer tumor samples as malignant or benign based on measurements computed from digitized images of a fine needle aspirate.

The project uses Support Vector Machines (SVM) for binary classification. The workflow covers data understanding, EDA, kernel and hyperparameter experimentation, tuning, error analysis, model saving, and Streamlit deployment.

> **Disclaimer:** This project is for educational purposes only and is not a medical diagnostic tool.

## Problem Statement

The goal of this project is to predict whether a breast tumor is malignant or benign based on 30 numerical features describing cell nuclei.

This is a:

- Supervised learning problem
- Binary classification problem

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Support Vector Machine (`SVC`)
- **Target variable:** `target`

Target classes:

- `0` → Malignant (212 cases)
- `1` → Benign (357 cases)

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset, loaded directly from `sklearn.datasets.load_breast_cancer` and saved locally as a CSV.

The dataset contains:

- 569 observations
- 30 numerical input features, grouped into "mean", "standard error", and "worst" measurements of 10 underlying cell characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- No missing values and no duplicate rows

## Exploratory Data Analysis

EDA covered the target distribution, boxplots of key mean features (radius, texture, perimeter, area) against the target, and a correlation heatmap across all features.

The features most correlated with the target were the "worst" and "mean" versions of concave points, perimeter, radius, and area, all negatively correlated with the target (recall `0 = malignant`, `1 = benign`), meaning larger values of these features are associated with malignancy.

## Preprocessing

All 30 features are numerical. Preprocessing consists of standardizing them with `StandardScaler`, combined with the SVM model inside a single scikit-learn `Pipeline`. Scaling matters for SVM because it is a distance/margin-based algorithm, and features on very different numeric scales would otherwise dominate the decision boundary.

## Model Experimentation

Several SVM configurations were compared on the same 80/20 stratified train/test split before tuning:

| Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Linear kernel (default C) | 97.37% | 98.59% | 97.22% | 97.90% | 99.64% |
| RBF kernel (default C, gamma) | 98.25% | 98.61% | 98.61% | 98.61% | 99.50% |
| Polynomial kernel (degree 3) | 91.23% | 87.80% | 100.00% | 93.51% | 99.54% |

A manual sweep over `C` (with the RBF kernel) and over `gamma` (at `C=1`) was also performed to build intuition for how each parameter affects the decision boundary: very small `C` or large `gamma` degraded performance sharply (down to 63.16% accuracy), while `C=1` with `gamma=0.01` gave the best manual result (98.25% accuracy, 99.57% ROC-AUC).

## Hyperparameter Tuning

A `GridSearchCV` (5-fold cross-validation, scored on ROC-AUC) searched jointly over the kernel type and its relevant hyperparameters:

- Linear: `C ∈ {0.01, 0.1, 1, 10, 100}`
- RBF: `C ∈ {0.01, 0.1, 1, 10, 100}`, `gamma ∈ {0.001, 0.01, 0.1, 1}`
- Polynomial: `C ∈ {0.1, 1, 10}`, `degree ∈ {2, 3, 4}`, `gamma ∈ {"scale", 0.01, 0.1}`

The search selected an RBF kernel with `C = 10` and `gamma = 0.01`, with a cross-validated ROC-AUC of approximately 99.54%.

## Final Model Performance

Performance of the tuned model on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 98.25% |
| Precision | 98.61% |
| Recall | 98.61% |
| F1-Score | 98.61% |
| ROC-AUC | 99.77% |

Confusion matrix:

```
[[41,  1],
 [ 1, 71]]
```

- True Negatives (malignant correctly identified): 41
- False Positives (malignant predicted as benign): 1
- False Negatives (benign predicted as malignant): 1
- True Positives (benign correctly identified): 71

## Error Analysis

The final model misclassified only 2 of the 114 test samples:

- One malignant case was predicted as benign, with a decision score of about +0.91, meaning the model was fairly confident in the wrong direction. This is the more consequential type of error in a cancer-screening context, since it corresponds to missing a malignant case.
- One benign case was predicted as malignant, with a decision score of about -0.29, close to the decision boundary.

With only 2 errors, no broader pattern could be drawn beyond these individual cases.

## Machine Learning Pipeline

The final model is saved as a single scikit-learn `Pipeline` combining `StandardScaler` and the tuned `SVC` model, so raw feature values can be passed in directly without scaling them manually. The saved pipeline was verified to produce identical predictions after being reloaded with `joblib`.

## Model Saving

The final trained pipeline is saved with `joblib` to:

```
models/breast_cancer_svm_pipeline.pkl
```

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app.py`). It presents the 30 input features as number inputs, grouped into three sections (mean, standard error, and worst features), builds a single-row DataFrame from them, and passes it into the saved pipeline.

Since the model was not trained with probability estimates enabled, the app reports the raw SVM decision score rather than a probability: a positive score favors the benign class and a negative score favors malignant, with magnitude reflecting distance from the decision boundary. The app displays the predicted class (malignant or benign) along with this decision score, and shows a warning that the prediction is not a medical diagnosis.

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
│   └── breast_cancer_svm_pipeline.pkl
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

## Learning Objective

This project was built as part of a practical machine learning learning path, focused on understanding Support Vector Machines: how the kernel choice (linear, RBF, polynomial) and the `C` and `gamma` hyperparameters shape the decision boundary, and how that compares to the previously used KNN, Logistic Regression, and Naive Bayes models on a classification task.