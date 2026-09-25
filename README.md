# ML-Projects

A collection of small end-to-end machine learning projects, one per algorithm, built while learning. Each project applies a single algorithm to a real dataset, covering the full workflow from data analysis to a deployed Streamlit app.

## Goal

The purpose of this repository is to build practical intuition for machine learning algorithms by implementing each one in a complete, standalone project rather than in isolation. Coverage will grow progressively, from simpler models toward more advanced ones such as XGBoost.

## Projects

| Project | Algorithm | Task |
|---|---|---|
| [House-price-project](./House-price-project) | Linear Regression | Predicting house sale prices |
| [Heart-disease-project](./Heart-disease-project) | Logistic Regression | Predicting presence of heart disease |

More projects will be added here as new algorithms are covered.

## Structure

Each project folder is self-contained and follows the same layout:

```
<project-name>/
│
├── app/            # Streamlit application
├── data/raw/        # Dataset
├── models/          # Saved trained pipeline
├── notebooks/        # Analysis and training notebook
├── README.md         # Project-specific documentation
└── requirements.txt
```

See each project's own README for details on its dataset, methodology, and results.
