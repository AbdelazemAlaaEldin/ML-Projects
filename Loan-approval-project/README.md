# Loan Approval Prediction using Decision Tree

## Project Overview

This project builds an end-to-end machine learning solution to predict whether a loan application will be approved or rejected, based on the applicant's personal, financial, and credit-related information.

The project uses a Decision Tree classifier. The workflow covers data understanding, exploratory data analysis, preprocessing inside a pipeline, simple baselines, overfitting analysis with a validation curve, hyperparameter tuning (including cost-complexity pruning), final evaluation with bootstrap confidence intervals, error analysis, model saving, and Streamlit deployment. All model selection is done with cross-validation on the training set only, and the test set is used once at the end.

> **Disclaimer:** This project is for educational purposes only. It is not a real credit decision tool and must not be used for real-world financial decisions.

## Problem Statement

The goal of this project is to predict the outcome of a loan application (approved or rejected) from 11 applicant features.

The more costly mistake in this setting is approving a loan that should have been rejected. This shapes how the results are reported: the rejected class (`N`) is treated as the class of interest for precision and recall.

## Machine Learning Problem

- **Learning type:** Supervised learning
- **Problem type:** Binary classification
- **Algorithm:** Decision Tree (`DecisionTreeClassifier`)
- **Target variable:** `Loan_Status` (`Y` = Approved, `N` = Rejected)
- **Class of interest for reported metrics:** Rejected (`pos_label="N"`)
- **Tuning metric:** `recall_macro` (no cost matrix between the two error types is known, so both classes are weighted equally)

## Dataset

The project uses the Loan Prediction dataset downloaded with `kagglehub` from the Kaggle dataset `saunakrath/dataset`. The downloaded folder contains three files: `dataset.csv`, `test.csv`, and `train.csv`. Only `train.csv` is used, because it is the file that contains `Loan_Status`. `test.csv` (367 rows, 12 columns) has no `Loan_Status`, and `dataset.csv` uses lowercase column names. `train.csv` is copied to `data/raw/loan_approval.csv`, and the notebook reads from that copy.

The dataset contains:

- 614 observations and 13 columns (`Loan_ID`, 11 features, and the target)
- Categorical features: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`, and `Credit_History` (stored as 0.0 and 1.0, treated as categorical)
- Numerical features: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
- Target distribution: 422 approved (68.73%) and 192 rejected (31.27%)

## Data Cleaning

- Duplicated rows: 0
- Duplicated `Loan_ID`: 0
- Missing cells: 149 in total, spread over 7 columns:

| Column | Missing |
|---|---:|
| Credit_History | 50 |
| Self_Employed | 32 |
| LoanAmount | 22 |
| Dependents | 15 |
| Loan_Amount_Term | 14 |
| Gender | 13 |
| Married | 3 |

No rows or columns were removed. `Loan_ID` is an identifier and is dropped before the train/test split. Missing values are handled inside the pipeline (see Preprocessing), so imputation statistics are learned from training folds only.

## Exploratory Data Analysis

**Numerical features.** All four are skewed:

| Feature | Skewness |
|---|---:|
| ApplicantIncome | 6.5395 |
| CoapplicantIncome | 7.4915 |
| LoanAmount | 2.6776 |
| Loan_Amount_Term | -2.3624 |

Decision trees split on thresholds, so no transformation or scaling was applied.

**Median of numerical features by loan status:**

| Feature | Rejected (N) | Approved (Y) |
|---|---:|---:|
| ApplicantIncome | 3833.5 | 3812.5 |
| CoapplicantIncome | 268.0 | 1239.5 |
| LoanAmount | 129.0 | 126.0 |
| Loan_Amount_Term | 360.0 | 360.0 |

The numerical features separate the classes weakly. The boxplots overlap heavily, and `CoapplicantIncome` is the only feature with a visible median difference.

**Approval rate by category** (missing values shown as their own group):

| Feature | Value | Count | Approved % |
|---|---|---:|---:|
| Credit_History | 0.0 | 89 | 7.87 |
| Credit_History | 1.0 | 475 | 79.58 |
| Credit_History | Missing | 50 | 74.00 |
| Property_Area | Rural | 179 | 61.45 |
| Property_Area | Semiurban | 233 | 76.82 |
| Property_Area | Urban | 202 | 65.84 |
| Education | Graduate | 480 | 70.83 |
| Education | Not Graduate | 134 | 61.19 |
| Married | No | 213 | 62.91 |
| Married | Yes | 398 | 71.61 |
| Married | Missing | 3 | 100.00 |
| Dependents | 0 | 345 | 68.99 |
| Dependents | 1 | 102 | 64.71 |
| Dependents | 2 | 101 | 75.25 |
| Dependents | 3+ | 51 | 64.71 |
| Dependents | Missing | 15 | 60.00 |
| Gender | Female | 112 | 66.96 |
| Gender | Male | 489 | 69.33 |
| Gender | Missing | 13 | 61.54 |
| Self_Employed | No | 500 | 68.60 |
| Self_Employed | Yes | 82 | 68.29 |
| Self_Employed | Missing | 32 | 71.88 |

`Credit_History` is by far the strongest signal. Applicants with `Credit_History = 1.0` are approved at 79.58%, compared with 7.87% for `0.0`. Applicants with a missing `Credit_History` (50) are approved at 74.00%, which is close to the `1.0` group, so imputing the most frequent value (1.0) does not hide a large difference. With only 50 cases, this is an observation and not a firm conclusion. The groups with very few missing cases (for example `Married`, 3 cases) are too small to interpret.

## Preprocessing

Preprocessing is a `ColumnTransformer` inside a single scikit-learn `Pipeline` together with the model, built by a function that returns a fresh transformer for every pipeline. Every step, including cross-validation, fits the preprocessing on the training portion of each fold only, so no information from validation data leaks into the transformers.

- **Numerical features:** missing values imputed with the median. No scaling.
- **Categorical features** (including `Credit_History`): missing values imputed with the most frequent value, then one-hot encoded with `handle_unknown="ignore"`.

After preprocessing, the encoded feature space is wider than the 11 raw columns because of one-hot encoding. No custom transformers are used in this project, so no `src/` module is needed.

## Train/Test Split

The data is split 80/20 with stratification on the target (`random_state=42`):

| Set | Samples | Approved (Y) | Rejected (N) |
|---|---:|---:|---:|
| Train | 491 | 337 | 154 |
| Test | 123 | 85 | 38 |

The test set is not touched until tuning is complete.

## Baseline Models

Three baselines were evaluated with 5-fold stratified cross-validation on the training set only (mean ± std across folds). Metrics for the rejected class use `pos_label="N"`.

| Model | Accuracy | Recall (macro) | Rejected recall | Rejected precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy (always Approved) | 0.6864 ± 0.0040 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Tree depth 1 (stump) | 0.7983 ± 0.0457 | 0.6892 ± 0.0633 | 0.3961 ± 0.1125 | 0.8923 ± 0.1244 | 0.6892 ± 0.0633 |
| Tree fully grown | 0.6965 ± 0.0252 | 0.6415 ± 0.0316 | 0.4935 ± 0.0747 | 0.5191 ± 0.0425 | 0.6415 ± 0.0316 |

The fully grown tree reaches an accuracy close to the dummy classifier, and its macro recall is lower than the single-split stump. The ROC-AUC of both trees equals their macro recall exactly. Pure leaves return probabilities of only 0 or 1, so the ROC-AUC of a tree reduces to a balanced accuracy, which makes it a weak metric for tuning a tree.

The rule learned by the stump, read with `export_text`:

```
|--- cat__Credit_History_1.0 <= 0.50
|   |--- class: N
|--- cat__Credit_History_1.0 >  0.50
|   |--- class: Y
```

The stump is a one-rule model: reject if `Credit_History` is 0.0, approve otherwise.

### Overfitting Analysis

A validation curve over `max_depth` was computed with 5-fold cross-validation on the training set only, using the default values for all other parameters:

| max_depth | Train recall_macro | CV recall_macro | CV std |
|---|---:|---:|---:|
| 1 | 0.6892 | 0.6892 | 0.0633 |
| 2 | 0.6990 | 0.6832 | 0.0596 |
| 3 | 0.7112 | 0.6735 | 0.0530 |
| 4 | 0.7306 | 0.6676 | 0.0560 |
| 5 | 0.7488 | 0.6587 | 0.0371 |
| 6 | 0.7908 | 0.6323 | 0.0345 |
| 8 | 0.8578 | 0.6537 | 0.0319 |
| 10 | 0.9273 | 0.6564 | 0.0307 |
| 12 | 0.9637 | 0.6452 | 0.0338 |
| 15 | 0.9883 | 0.6504 | 0.0340 |
| None | 1.0000 | 0.6415 | 0.0316 |

The training score rises from 0.6892 to 1.0000 while the cross-validation score falls from 0.6892 to 0.6415. No depth beats the single split on the cross-validation score, and the difference between depth 1 and depth 2 (0.6892 versus 0.6832) is much smaller than the standard deviation. Extra splits beyond the `Credit_History` rule do not generalize in this setup.

## Hyperparameter Tuning

A `GridSearchCV` (5-fold stratified cross-validation, scored on `recall_macro`, fitted on the training set only) searched 480 combinations:

- `criterion`: gini, entropy
- `max_depth`: 1, 2, 3, 4, 5, None
- `min_samples_leaf`: 1, 5, 10, 20
- `ccp_alpha` (cost-complexity pruning strength): 0.0, 0.002, 0.005, 0.01, 0.02
- `class_weight`: None, balanced

Best parameters: `ccp_alpha = 0.005`, `class_weight = None`, `criterion = gini`, `max_depth = None`, `min_samples_leaf = 1`, with a cross-validated `recall_macro` of 0.6926. The final tree has depth 6 and 9 leaves.

Top configurations:

| Rank | Parameters | Mean recall_macro | Std |
|---:|---|---:|---:|
| 1 | gini, depth None, leaf 1, ccp_alpha 0.005 | 0.6926 | 0.0458 |
| 2 | gini, depth 5, leaf 1, ccp_alpha 0.005 | 0.6896 | 0.0443 |
| 3 | entropy, depth 3, leaf 10, ccp_alpha 0.005 | 0.6894 | 0.0634 |
| 3 | gini, depth 3, leaf 10, ccp_alpha 0.0 | 0.6894 | 0.0634 |
| 8 | entropy, depth None, leaf 20, ccp_alpha 0.01 | 0.6892 | 0.0633 |
| 8 | entropy, depth 1, leaf 1, ccp_alpha 0.0 | 0.6892 | 0.0633 |

(Several other configurations also share rank 3 with the same score and standard deviation; two are shown.) All ten top configurations use `class_weight = None`.

The best stump in the grid (depth 1) scored 0.6892 ± 0.0633 and ranked 8th. The difference between the best configuration and the best stump is 0.0034, far below either standard deviation, and the whole top 10 lies between 0.6892 and 0.6926. The grid does not meaningfully separate a pruned tree from the one-rule model. The pruned tree was kept because it had the highest mean cross-validation score, a rule fixed before looking at the test set.

Pruning mattered compared with the unpruned tree at the same depth: an unpruned depth-6 tree scored 0.6323 in the validation curve, and the pruned tree scored 0.6926.

## Final Model Performance

The test set was evaluated once. The Dummy and Stump columns are fixed references (not candidates for selection); the decision was made before the test set was used. Metrics for the rejected class use `pos_label="N"`.

| Metric | Final tree | Dummy | Stump |
|---|---:|---:|---:|
| Accuracy | 0.8537 | 0.6911 | 0.8537 |
| Rejected precision | 0.8571 | 0.0000 | 0.9545 |
| Rejected recall | 0.6316 | 0.0000 | 0.5526 |
| Rejected F1 | 0.7273 | 0.0000 | 0.7000 |
| Recall (macro) | 0.7923 | 0.5000 | 0.7704 |
| ROC-AUC | 0.7621 | 0.5000 | 0.7704 |

Bootstrap 95% confidence intervals for the final tree (1000 resamples of the test set, `np.random.RandomState(42)`, all 1000 valid):

| Metric | Test score | 95% CI |
|---|---:|---|
| Accuracy | 0.8537 | [0.7886, 0.9108] |
| Rejected precision | 0.8571 | [0.7241, 0.9667] |
| Rejected recall | 0.6316 | [0.4827, 0.7941] |
| Rejected F1 | 0.7273 | [0.5937, 0.8379] |
| Recall (macro) | 0.7923 | [0.7121, 0.8743] |
| ROC-AUC | 0.7621 | [0.6608, 0.8597] |

Confusion matrix (final tree):

| | Pred Rejected | Pred Approved |
|---|---:|---:|
| True Rejected | 24 | 14 |
| True Approved | 4 | 81 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Rejected | 0.8571 | 0.6316 | 0.7273 | 38 |
| Approved | 0.8526 | 0.9529 | 0.9000 | 85 |
| Accuracy | | | 0.8537 | 123 |
| Macro avg | 0.8549 | 0.7923 | 0.8136 | 123 |
| Weighted avg | 0.8540 | 0.8537 | 0.8466 | 123 |

The final tree is clearly better than the dummy classifier, but it has exactly the same test accuracy as the one-rule stump (0.8537), with a different balance between precision and recall for the rejected class. The ROC-AUC of the stump (0.7704) is slightly higher than that of the final tree (0.7621), and the difference lies well inside the confidence intervals. The intervals are wide because the test set contains only 38 rejected applications: the rejected recall interval starts at 0.4827.

## Error Analysis

The final tree made 18 errors on 123 test applications:

| Error type | Count |
|---|---:|
| Correct | 105 |
| Missed rejection (rejected loan predicted as approved) | 14 |
| Wrongly rejected (approved loan predicted as rejected) | 4 |

Predicted approval probability by error type (this is the share of training applicants in the same tree leaf who were approved, not a calibrated probability):

| Error type | Count | Mean | Min | Max |
|---|---:|---:|---:|---:|
| Correct | 105 | 0.6481 | 0.0000 | 1.0000 |
| Missed rejection | 14 | 0.8231 | 0.7660 | 0.9259 |
| Wrongly rejected | 4 | 0.0224 | 0.0000 | 0.0896 |

**Credit_History on the test set** (true class, predicted class):

| Credit_History | Rejected, pred Rejected | Rejected, pred Approved | Approved, pred Rejected | Approved, pred Approved |
|---|---:|---:|---:|---:|
| 0.0 | 21 | 0 | 1 | 0 |
| 1.0 | 3 | 14 | 3 | 74 |
| Missing | 0 | 0 | 0 | 7 |

All 14 missed rejections have `Credit_History = 1.0`. The model caught all 21 rejected applicants with `Credit_History = 0.0`, which is the stump rule, and caught only 3 of the 17 rejected applicants with `Credit_History = 1.0`.

**Missed rejections versus caught rejections versus correct approvals** (group sizes 14, 24, 81; the missed group is small, so this is descriptive and not a firm pattern):

| Feature | Missed rejection | Caught rejection | Correct approval |
|---|---:|---:|---:|
| Median ApplicantIncome | 3508.0 | 3081.5 | 3707.0 |
| Median CoapplicantIncome | 1445.0 | 1365.5 | 1600.0 |
| Median LoanAmount | 150.0 | 121.0 | 120.0 |
| Median Loan_Amount_Term | 360.0 | 360.0 | 360.0 |
| Credit_History = 1.0 (%) | 100.0 | 12.5 | 91.4 |
| Married = Yes (%) | 57.1 | 45.8 | 74.1 |
| Education = Graduate (%) | 85.7 | 70.8 | 86.4 |
| Property_Area = Semiurban (%) | 14.3 | 45.8 | 43.2 |
| Dependents = 0 (%) | 57.1 | 41.7 | 58.0 |
| Self_Employed = Yes (%) | 0.0 | 16.7 | 14.8 |
| Gender = Male (%) | 71.4 | 66.7 | 81.5 |

In the features available, the missed rejections look much like correct approvals (similar `Credit_History`, `Education`, `Dependents`, and income). The visible differences (`LoanAmount`, `Property_Area`, `Married`) rest on 14 cases and are not treated as a pattern. This points to a limit of the data and not only of the model.

**The final tree** (class weights are training counts `[N, Y]`):

```
|--- cat__Credit_History_1.0 <= 0.50
|   |--- weights: [61.00, 6.00] class: N
|--- cat__Credit_History_1.0 >  0.50
|   |--- num__CoapplicantIncome <= 9537.00
|   |   |--- num__ApplicantIncome <= 3357.50
|   |   |   |--- num__LoanAmount <= 76.50
|   |   |   |   |--- weights: [11.00, 24.00] class: Y
|   |   |   |--- num__LoanAmount >  76.50
|   |   |   |   |--- num__LoanAmount <= 164.00
|   |   |   |   |   |--- weights: [8.00, 100.00] class: Y
|   |   |   |   |--- num__LoanAmount >  164.00
|   |   |   |   |   |--- num__CoapplicantIncome <= 2824.50
|   |   |   |   |   |   |--- weights: [4.00, 1.00] class: N
|   |   |   |   |   |--- num__CoapplicantIncome >  2824.50
|   |   |   |   |   |   |--- weights: [0.00, 7.00] class: Y
|   |   |--- num__ApplicantIncome >  3357.50
|   |   |   |--- num__LoanAmount <= 95.50
|   |   |   |   |--- cat__Education_Not Graduate <= 0.50
|   |   |   |   |   |--- weights: [7.00, 18.00] class: Y
|   |   |   |   |--- cat__Education_Not Graduate >  0.50
|   |   |   |   |   |--- weights: [5.00, 0.00] class: N
|   |   |   |--- num__LoanAmount >  95.50
|   |   |   |   |--- weights: [55.00, 180.00] class: Y
|   |--- num__CoapplicantIncome >  9537.00
|   |   |--- weights: [3.00, 1.00] class: N
```

**Errors by leaf.** Leaf counts come from the training and test sets:

| Leaf | Train N | Train Y | Test N | Test Y | Prediction | Train approved share |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 61 | 6 | 21 | 1 | N | 0.0896 |
| 5 | 11 | 24 | 0 | 7 | Y | 0.6857 |
| 7 | 8 | 100 | 5 | 25 | Y | 0.9259 |
| 9 | 4 | 1 | 1 | 0 | N | 0.2000 |
| 10 | 0 | 7 | 0 | 2 | Y | 1.0000 |
| 13 | 7 | 18 | 0 | 5 | Y | 0.7200 |
| 14 | 5 | 0 | 1 | 3 | N | 0.0000 |
| 15 | 55 | 180 | 9 | 42 | Y | 0.7660 |
| 16 | 3 | 1 | 1 | 0 | N | 0.2500 |

The 14 missed rejections fall in two leaves: 9 in leaf 15 and 5 in leaf 7. Leaf 15 is the large leaf (55 N and 180 Y in training) where the tree has no feature left to separate rejected from approved applicants. Of the 4 wrongly rejected approvals, 3 come from leaf 14, a rule built on only 5 training samples (`Education = Not Graduate`, all 5 rejected), and 1 comes from leaf 1 (`Credit_History = 0.0`). Leaf 7 has 5 rejected out of 30 test applicants versus 8 out of 108 in training; with 30 test cases this may be chance or may mean the training share (0.9259) is optimistic, and the test set is not used to decide which.

### Feature Importance

Two importance measures are reported for documentation only (no decision was changed based on them). Impurity importance is summed over the one-hot columns of each original feature. Permutation importance shuffles one raw column in the test set (30 repeats) and measures the drop in `recall_macro`.

| Feature | Impurity importance | Permutation drop (mean) | Permutation std |
|---|---:|---:|---:|
| Credit_History | 0.7557 | 0.2563 | 0.0361 |
| CoapplicantIncome | 0.0821 | 0.0272 | 0.0079 |
| LoanAmount | 0.0739 | 0.0295 | 0.0189 |
| Education | 0.0591 | 0.0040 | 0.0072 |
| ApplicantIncome | 0.0292 | 0.0496 | 0.0175 |
| Dependents | 0.0000 | 0.0000 | 0.0000 |
| Gender | 0.0000 | 0.0000 | 0.0000 |
| Loan_Amount_Term | 0.0000 | 0.0000 | 0.0000 |
| Married | 0.0000 | 0.0000 | 0.0000 |
| Property_Area | 0.0000 | 0.0000 | 0.0000 |
| Self_Employed | 0.0000 | 0.0000 | 0.0000 |

`Credit_History` dominates under both measures. The order of the other features differs between the two measures (for example, `ApplicantIncome` has the lowest impurity importance of the used features but the second-highest permutation drop), so their ranking is not reliable. The permutation standard deviation reflects the randomness of the shuffling only, not the sampling uncertainty of a 123-row test set. `Education` has a permutation drop (0.0040) smaller than its standard deviation (0.0072).

An importance of 0.0000 means the pruned tree did not need that feature. It does not mean the feature has no predictive value; for example, `Property_Area` shows a clear difference in approval rate in the EDA but is not used by this tree.

### Does the leaf probability separate safe from risky approvals?

To see whether the model's approval probability can flag risky approvals, predicted approvals were grouped by probability range. Out-of-fold probabilities on the training set (`cross_val_predict`) are compared with the test set:

| Probability range | Train OOF approvals | Train OOF actually rejected | Test approvals | Test actually rejected |
|---|---:|---:|---:|---:|
| 0.50-0.70 | 16 | 0 | 7 | 0 |
| 0.70-0.80 | 113 | 22 | 56 | 9 |
| 0.80-0.90 | 175 | 47 | 0 | 0 |
| 0.90-1.00 | 84 | 12 | 32 | 5 |

The rejected share does not fall as the probability rises: the 0.80-0.90 range has the highest share in the training out-of-fold predictions (47 of 175), and the 0.90-1.00 range still has 5 of 32 rejected in the test set. The only clean range (0.50-0.70) has 16 training cases. The 0.80-0.90 range is empty in the test set because a single tree produces only discrete leaf values, whereas out-of-fold predictions combine five different trees. The leaf probability therefore cannot be used to flag risky approvals, so the app warns about approvals in general instead of using a probability threshold:

| Set | Predicted approvals | Of which actually rejected |
|---|---:|---|
| Train (out-of-fold) | 388 | 81 (20.88%) |
| Test | 95 | 14 (14.74%) |

Among applicants with `Credit_History = 1.0`, 80 of 381 were rejected in the training set and 17 of 94 in the test set.

## Machine Learning Pipeline

The final model is a single scikit-learn `Pipeline` combining the `ColumnTransformer` (median imputation for numerical features; most-frequent imputation and one-hot encoding for categorical features) and the tuned `DecisionTreeClassifier` (`criterion="gini"`, `ccp_alpha=0.005`, `random_state=42`). Raw applicant values can be passed in directly.

## Model Saving

The trained pipeline is saved with `joblib` to:

```
models/decision_tree_loan_pipeline.pkl
```

After saving, the pipeline was reloaded with `joblib.load` and compared with the original model on the full test set: the predictions are identical (`Identical predictions: True`) and the predicted probabilities are identical (`Identical probabilities: True`).

A metadata file used by the app is saved to:

```
models/app_meta.json
```

It contains the training-set median of each numerical feature (used as input defaults, for example `ApplicantIncome` 3859.0, `CoapplicantIncome` 1032.0, `LoanAmount` 128.0, `Loan_Amount_Term` 360.0), the category options and most frequent value of each categorical feature, and the share of training out-of-fold approvals that were actually rejected (0.2088).

## Streamlit Application

The trained pipeline is served through a Streamlit app (`app/app.py`). The model and metadata are loaded once with `@st.cache_resource`, and paths are resolved from the file location, so the app can be started from any directory. Input defaults and select-box options come from the training data through `app_meta.json`. The app builds a single-row DataFrame matching the training schema, passes it to the saved pipeline, and shows the predicted class together with the leaf approval share.

The leaf approval share is the fraction of training applicants in the same tree leaf who were approved. It is not a calibrated probability, and the app says so.

The app shows a caption for predicted approvals, since the model cannot separate safe approvals from risky ones: among predicted approvals, 81 of 388 (20.88%) were actually rejected in cross-validation on the training data, and 14 of 95 (14.74%) in the test set. For a rejection with `Credit_History = 0.0`, the app notes that the model rejects this case. An "About this model" section summarizes the test performance, the bootstrap intervals, and the main limitation. A warning at the top states that this is an educational project and not a real credit decision.

### Run the application

From the project root:

```
streamlit run app/app.py
```

## Project Structure

```
Loan-approval-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── loan_approval.csv
│
├── models/
│   ├── decision_tree_loan_pipeline.pkl
│   └── app_meta.json
│
├── notebooks/
│   └── loan_approval_analysis.ipynb
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

Open `notebooks/loan_approval_analysis.ipynb`. The notebook downloads the dataset with `kagglehub`, copies `train.csv` to `data/raw/loan_approval.csv`, and regenerates the saved model and the app metadata.

## Key Findings and Limitations

**Differences from the previous version of this project, and why:**

- The previous version evaluated the baseline tree on the test set, then built a `max_depth` table on the test set (train versus test accuracy for each depth), then evaluated the final model on the test set, so the test set was used several times before the end and the depth table was effectively tuned on it. In this version, the baselines, the overfitting analysis, and the tuning all use cross-validation on the training set only, and the test set is evaluated once.
- The previous version tuned on ROC-AUC. This version tunes on `recall_macro`, because a tree's ROC-AUC reduces to a balanced accuracy when its leaves are pure, and because the rejected class (the costly one) should not be hidden by the larger approved class.
- Baselines were added: a dummy classifier and a one-split stump. The previous version had no reference for how much a tuned tree actually adds.
- Cost-complexity pruning (`ccp_alpha`) and `class_weight` were added to the search, in addition to the depth and leaf-size parameters of the previous version.
- Bootstrap 95% confidence intervals were added for all final test metrics.
- The error analysis now compares missed rejections with caught rejections and correct approvals, examines errors per leaf, and checks whether the leaf probability can flag risky approvals. The previous version listed the errors and one `Credit_History` table.
- Feature importance now includes permutation importance on the original columns, next to the impurity importance.
- The round-trip check now compares predictions and probabilities on the whole test set. The previous version only printed the predictions for 5 samples without comparing them with the original model.
- The data loading is consistent: the previous notebook copied `dataset.csv` into `data/raw` but then read `train.csv` directly from the download folder. This version copies `train.csv` and reads from `data/raw`.
- The app now has no emojis, uses `@st.cache_resource`, resolves paths from the file location, labels `Loan_Amount_Term` in months (the previous label said days, but values such as 360 suggest months), takes defaults and options from the training data, and replaces "Approval Probability" with the leaf approval share and an explanation.

**Findings:**

- `Credit_History` dominates the model. A one-split rule on `Credit_History` alone reaches the same test accuracy (0.8537) as the tuned tree, and the cross-validation difference between them (0.6926 versus 0.6892) is much smaller than the standard deviation.
- Without pruning, deeper trees overfit clearly: the training score reaches 1.0000 while the cross-validation score falls to 0.6415. Pruning improved the cross-validation score at depth 6 from 0.6323 to 0.6926.
- The tuned tree reaches an accuracy of 0.8537 and a rejected recall of 0.6316 (24 of 38 rejected applications caught) on the test set, compared with a dummy accuracy of 0.6911.
- The previous version reported a test accuracy of 0.8130 and 23 errors (16 rejected loans predicted as approved) for its final tree, against 0.8537 and 18 errors (14 rejected loans predicted as approved) here. The test split is the same (same `random_state` and stratification), so the numbers are comparable on the same applications. The previous best cross-validation ROC-AUC (0.7375) cannot be compared with the current cross-validation `recall_macro` (0.6926), since the metrics differ.

**Limitations:**

- The test set has 123 applications, 38 of them rejected. The bootstrap interval for the rejected recall is [0.4827, 0.7941], so the true value may be below 50%.
- The 14 missed rejections all have `Credit_History = 1.0` and look similar to correct approvals in the available features. The model has no feature that separates them, so this is a limit of the data as much as of the model.
- The leaf probability is not calibrated and does not separate safe from risky approvals, so it should not be read as an approval probability.
- Leaf 14, built from 5 training samples, caused 3 of the 4 wrongly rejected approvals. Small leaves remain a source of instability even after pruning.
- The 50 missing `Credit_History` values are imputed with 1.0. Their approval rate in the data (74.00%) is close to the `1.0` group, but 50 cases are too few to be sure, and in the test set all 7 such cases were approved.
- The dataset is small (614 applications) and comes from a single source. The units of `LoanAmount` and `ApplicantIncome` are not documented in the data files used here.
- The data includes personal attributes such as `Gender`, `Married`, and `Dependents`. The final tree did not use them (importance 0.0000), but a model built on this kind of data must not be used for real lending decisions.
- The model is for educational use only.

## Learning Objective

This project was built as part of a practical machine learning learning path, focused on understanding Decision Trees through a complete real-world problem.

The project focuses on understanding:

- How a tree is built from splits, how to read it as rules with `export_text`, and why a fully grown tree memorizes the training data
- How to diagnose overfitting with a validation curve on the training set instead of looking at the test set
- How cost-complexity pruning (`ccp_alpha`) controls tree size differently from `max_depth`, and how `class_weight` shifts attention to the minority class
- Why simple baselines (a dummy classifier and a one-split stump) are essential for judging whether a tuned model adds real value
- Why ROC-AUC is a weak tuning metric for a single tree, and why the tuning metric should reflect the cost of errors
- The difference between impurity importance and permutation importance, and why one-hot encoded features should be aggregated back to the original columns
- Why a tree's leaf probability is not a calibrated probability, and how to check whether it can flag risky predictions
- How to report test performance honestly with bootstrap confidence intervals when the minority class is small
- How this compares to the previously used SVM, KNN, Logistic Regression, and Naive Bayes models on a classification task