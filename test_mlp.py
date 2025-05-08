import torch
import numpy as np
from mlp_model import MLP
from data_utils import get_flattened_loader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = [
    {
        "name": "MNIST",
        "output_dim": 10,
        "model_path": "mlp_mnist.pt",
        "preds_file": "results/preds_mlp_mnist.npy",
        "labels_file": "results/labels_mlp_mnist.npy"
    },
    {
        "name": "EMNIST",
        "output_dim": 47,  # For EMNIST Balanced
        "model_path": "mlp_emnist.pt",
        "preds_file": "results/preds_mlp_emnist.npy",
        "labels_file": "results/labels_mlp_emnist.npy"
    }
]

os.makedirs("results", exist_ok=True)

for cfg in configs:
    print(f"Evaluating {cfg['name']}...")
    val_loader = get_flattened_loader(dataset_name=cfg["name"], train=False)
    model = MLP(output_dim=cfg["output_dim"])
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