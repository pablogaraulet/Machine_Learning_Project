import torch
import numpy as np
from cnn_model import CNN
from data_utils import get_cnn_loader
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset configurations
configs = [
    {
        "name": "MNIST",
        "num_classes": 10,
        "model_path": "cnn_mnist.pt",
        "preds_file": "results/preds_cnn_mnist.npy",
        "labels_file": "results/labels_cnn_mnist.npy"
    },
    {
        "name": "EMNIST",
        "num_classes": 47,  # For EMNIST Balanced
        "model_path": "cnn_emnist.pt",
        "preds_file": "results/preds_cnn_emnist.npy",
        "labels_file": "results/labels_cnn_emnist.npy"
    }
]

os.makedirs("results", exist_ok=True)

for cfg in configs:
    print(f"Evaluating {cfg['name']}...")
    val_loader = get_cnn_loader(dataset_name=cfg["name"], train=False)
    model = CNN(num_classes=cfg["num_classes"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    np.save(cfg["preds_file"], all_preds)
    np.save(cfg["labels_file"], all_labels)

    accuracy = (all_preds == all_labels).mean()
    print(f"{cfg['name']} Validation Accuracy: {accuracy:.4f}\n")