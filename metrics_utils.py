import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, log_loss
)

def compute_metrics(y_true, y_pred, class_names=None, verbose=True):
    """
    Computes and optionally prints classification metrics.

    Returns:
        (accuracy, precision, recall, f1_score, cross_entropy_loss)
    """
    if verbose:
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, target_names=class_names))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        ce_loss = log_loss(y_true, np.eye(np.max(y_true) + 1)[y_pred])
    except:
        ce_loss = float('nan')

    if verbose:
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Cross-Entropy Loss: {ce_loss:.4f}")

    return acc, prec, rec, f1, ce_loss