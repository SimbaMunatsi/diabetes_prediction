# Diabetes Risk Prediction --- Interpretable ML System

## Summary

Production-grade supervised machine learning system for diabetes risk prediction using structured healthcare data.
Designed with robust generalization, interpretability, and deployment readiness as first-class concerns.

The project demonstrates end-to-end ML system design: data validation, feature engineering, stratified cross-validation, explainability (SHAP), and a lightweight decision-support interface.

## System Highlights

Modular ML pipeline using scikit-learn + TensorFlow (SciKeras)

Feature engineering encapsulated in reusable transformers

Stratified cross-validation to prevent target leakage

Neural network classifier with reproducible training

SHAP-based interpretability for model transparency

Persisted inference pipeline for downstream consumption

Interactive Streamlit UI for risk exploration

## Architecture Overview 
```
┌────────────┐
│ Raw Data   │
└─────┬──────┘
      ↓
┌──────────────┐
│ Validation & │
│ Ingestion    │
└─────┬────────┘
      ↓
┌──────────────┐
│ Feature      │
│ Engineering  │
│ (Scaling)    │
└─────┬────────┘
      ↓
┌──────────────┐
│ Model        │
│ (Neural Net) │
└─────┬────────┘
      ↓
┌──────────────┐
│ Evaluation   │
│ (ROC-AUC)   │
└─────┬────────┘
      ↓
┌──────────────┐
│ Explainability│
│ (SHAP)       │
└─────┬────────┘
      ↓
┌──────────────┐
│ Inference &  │
│ Dashboard    │
└──────────────┘
```


## Project Structure
```
 diabetes-risk-prediction/
├── data/                    # Versioned dataset
├── src/
│   ├── data_loader.py       # Schema & data validation
│   ├── preprocessing.py    # Feature engineering pipeline
│   ├── model.py             # TF model + SciKeras wrapper
│   ├── evaluation.py        # Metrics & diagnostics
│   └── explainability.py    # SHAP analysis
├── app.py                   # Decision-support UI
├── train.py                 # Orchestration entrypoint
├── requirements.txt
└── README.md
```

## Dataset

Pima Indians Diabetes Dataset (768 samples, 8 clinical features, binary
target).

## Modeling & Evaluation

-  Modeling & Evaluation
-   Feature Engineering

-   Numeric standardization via StandardScaler

-   Implemented using ColumnTransformer

-   Fully pipeline-compatible (train/inference parity)

## Model

-   Feed-forward neural network

-   Integrated into sklearn via SciKeras

-   Deterministic configuration for reproducibility

## Validation Strategy

-   Stratified 5-fold cross-validation

-   Primary metric: ROC-AUC

-   Secondary diagnostics: accuracy, confusion matrix

## Explainability

Model predictions are interpreted using SHAP (Shapley Additive Explanations):

-   Quantifies per-feature contribution

-   Identifies dominant clinical risk drivers

-   Supports transparent decision-making

## Interactive Inference

A lightweight Streamlit dashboard enables:

-   Manual feature input

-   Real-time risk scoring

-   Intuitive exploration of model behavior

Demonstrates model usability beyond offline evaluation.

## Running the Project
## Set-up
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Train and evaluate
    python train.py

## Outputs:

    Cross-validated ROC-AUC

    Hold-out test performance

    Persisted pipeline (model.joblib)  

## Launch Dashboard
    streamlit run dashboard/app.py

## Design Trade-offs & Limitations

-   Dataset is a benchmark, not clinical-grade data

-   Zero values represent implicit missingness

-   Neural networks chosen for demonstration; tree-based models may yield higher tabular performance

-   Not intended for real medical diagnosis

These trade-offs are acknowledged explicitly. 

## Tech Stack

    Python 3.9+

    TensorFlow / Keras

    SciKeras

    scikit-learn

    SHAP

    pandas, NumPy

    Streamlit

## Future Extensions
    Future Extensions

    Tree-based model comparison (XGBoost / LightGBM)

    Integrated SHAP visualizations in dashboard

    Automated testing & CI

    Containerized deployment

    Cloud inference service

## Disclaimer

Educational use only. Not for clinical diagnosis.
