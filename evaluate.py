# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, log_loss
)
import os

def load_predictions(model_name):
    y_pred = np.load(f"results/preds_{model_name}.npy")
    y_true = np.load(f"results/labels_{model_name}.npy")
    return y_true, y_pred

def evaluate_classification(y_true, y_pred, class_names=None):
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ce_loss = log_loss(y_true, np.eye(np.max(y_true)+1)[y_pred])  # Approximated from hard labels

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Cross-Entropy Loss: {ce_loss:.4f}")

    return acc, prec, rec, f1, ce_loss

def plot_confusion(y_true, y_pred, title, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def compare_models(metrics_dict):
    models = list(metrics_dict.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.bar(x + i * width, metrics_dict[model][:4], width, label=model)

    plt.ylabel("Score")
    plt.title("MLP vs CNN Evaluation Metrics")
    plt.xticks(x + width / 2, metric_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    class_names = [str(i) for i in range(10)]  # Adjust if EMNIST

    metrics_summary = {}

    for model_name in ["mlp", "cnn"]:
        print(f"\n--- Evaluating {model_name.upper()} ---")
        y_true, y_pred = load_predictions(model_name)
        metrics = evaluate_classification(y_true, y_pred, class_names=class_names)
        plot_confusion(y_true, y_pred, title=f"Confusion Matrix: {model_name.upper()}", class_names=class_names)
        metrics_summary[model_name.upper()] = metrics

    compare_models(metrics_summary)
