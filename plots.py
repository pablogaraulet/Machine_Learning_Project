# plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def plot_comparison_bar_chart(metrics_dict):
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = list(metrics_dict.keys())
    values = [metrics_dict[model][:4] for model in models]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    for i, (model, scores) in enumerate(zip(models, values)):
        plt.bar(x + i * width, scores, width, label=model)

    plt.xticks(x + width / 2, metric_names)
    plt.ylabel("Score")
    plt.title("MLP vs CNN Evaluation Metrics")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_metrics_table(metrics_dict, output_path="results/metrics_summary.csv"):
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "Cross-Entropy Loss"]
    df = pd.DataFrame(metrics_dict, index=metric_names).T
    print("\nSummary Table of Metrics:")
    print(df.round(4))
    df.to_csv(output_path)
    print(f"\nMetrics saved to {output_path}")
