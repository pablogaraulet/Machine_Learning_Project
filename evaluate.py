import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss
)
import pandas as pd
import os

def load_data():
    data = {}
    for model in ['mlp', 'cnn']:
        try:
            data[f'preds_{model}'] = np.load(f'results/preds_{model}.npy')
            data[f'labels_{model}'] = np.load(f'results/labels_{model}.npy')
        except FileNotFoundError:
            print(f"Warning: Could not find prediction files for {model.upper()}")
    return data

def compute_metrics(y_true, y_pred, model_name):
    metrics = {
        'Model': model_name.upper(),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (macro)': precision_score(y_true, y_pred, average='macro'),
        'Recall (macro)': recall_score(y_true, y_pred, average='macro'),
        'F1 Score (macro)': f1_score(y_true, y_pred, average='macro'),
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, classes=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/confusion_matrix_{model_name}.png')
    plt.close()

def plot_metrics_comparison(metrics_mlp, metrics_cnn):
    metrics_df = pd.DataFrame([metrics_mlp, metrics_cnn])
    metrics_df = metrics_df.set_index('Model').transpose()
    
    plt.figure(figsize=(10, 6))
    metrics_df.plot(kind='bar', rot=0)
    plt.title('Model Comparison: MLP vs CNN')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.legend(title='Model')
    plt.tight_layout()
    
    plt.savefig('plots/model_comparison.png')
    plt.close()

def save_metrics_summary(metrics_mlp, metrics_cnn):
    metrics_df = pd.DataFrame([metrics_mlp, metrics_cnn])
    metrics_df.to_csv('results/metrics_summary.csv', index=False)
    print("\nMetrics summary saved to results/metrics_summary.csv")

def main():
    data = load_data()
    
    if not all(key in data for key in ['preds_mlp', 'labels_mlp', 'preds_cnn', 'labels_cnn']):
        print("Error: Missing prediction files for one or both models")
        return
    
    class_names = [str(i) for i in range(10)]
    
    metrics_mlp = compute_metrics(data['labels_mlp'], data['preds_mlp'], 'MLP')
    metrics_cnn = compute_metrics(data['labels_cnn'], data['preds_cnn'], 'CNN')
    
    print("\nEvaluation Metrics:")
    print(pd.DataFrame([metrics_mlp, metrics_cnn]))
    
    plot_confusion_matrix(data['labels_mlp'], data['preds_mlp'], 'MLP', class_names)
    plot_confusion_matrix(data['labels_cnn'], data['preds_cnn'], 'CNN', class_names)
    plot_metrics_comparison(metrics_mlp, metrics_cnn)
    
    save_metrics_summary(metrics_mlp, metrics_cnn)
    
    print("\nEvaluation completed. Plots saved to 'plots/' directory.")

if __name__ == '__main__':
    main()