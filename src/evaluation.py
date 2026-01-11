from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a trained classifier using multiple metrics.

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return metrics


def plot_confusion_matrix(cm):
    """
    Visualize confusion matrix for interpretability.
    """

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
