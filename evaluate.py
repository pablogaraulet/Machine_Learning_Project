import numpy as np
from metrics_utils import compute_metrics
from plots import (
    plot_confusion_matrix,
    plot_comparison_bar_chart,
    save_metrics_table
)

def load_predictions(model_name):
    """Load predictions and true labels for a given model (mlp or cnn)."""
    y_pred = np.load(f"results/preds_{model_name}.npy")
    y_true = np.load(f"results/labels_{model_name}.npy")
    return y_true, y_pred

if __name__ == "__main__":
    # Change class_names if you're evaluating EMNIST instead
    class_names = [str(i) for i in range(10)]

    model_names = ["mlp", "cnn"]
    metrics_summary = {}

    for model_name in model_names:
        print(f"\n--- Evaluating {model_name.upper()} ---")
        y_true, y_pred = load_predictions(model_name)

        metrics = compute_metrics(y_true, y_pred, class_names=class_names, verbose=True)
        metrics_summary[model_name.upper()] = metrics

        plot_confusion_matrix(
            y_true, y_pred,
            class_names=class_names,
            title=f"{model_name.upper()} Confusion Matrix"
        )

    plot_comparison_bar_chart(metrics_summary)
    save_metrics_table(metrics_summary, output_path="results/metrics_summary.csv")