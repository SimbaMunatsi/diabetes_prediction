import os
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # MUST be before pyplot import
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_loader import load_diabetes_data
from src.preprocessing import build_preprocessor
from src.model import get_keras_classifier
from src.evaluation import evaluate_classifier, plot_confusion_matrix


# =========================
# Configuration
# =========================
DATA_PATH = "data/diabetes.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


# =========================
# Load & validate data
# =========================
df = load_diabetes_data(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# =========================
# Build preprocessing + model pipeline
# =========================
preprocessor = build_preprocessor(X.columns.tolist())

classifier = get_keras_classifier(input_dim=X.shape[1])

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ]
)


# =========================
# Cross-validation (robust generalization)
# =========================
print("\nRunning stratified cross-validation...")

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print(f"Cross-Validation ROC-AUC Scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f}")


# =========================
# Train final model
# =========================
print("\nTraining final model on full training set...")
pipeline.fit(X_train, y_train)


# =========================
# Evaluate on hold-out test set
# =========================
print("\nEvaluating model on test set...")
metrics = evaluate_classifier(pipeline, X_test, y_test)

print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test ROC-AUC : {metrics['roc_auc']:.4f}")

plot_confusion_matrix(metrics["confusion_matrix"])


# =========================
# Persist trained pipeline
# =========================

print("Reached model saving stage...")
print("\nSaving trained model pipeline...")
joblib.dump(pipeline, MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file was not created!")

print("Model saved successfully.")
