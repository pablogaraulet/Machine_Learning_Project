import torch
import numpy as np
from mlp_model import MLP
from data_utils import get_flattened_loader

val_loader = get_flattened_loader(dataset_name="MNIST", train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(output_dim=10)
model.load_state_dict(torch.load("mlp_mnist.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

import os
os.makedirs("results", exist_ok=True)

np.save("results/preds_mlp.npy", all_preds)
np.save("results/labels_mlp.npy", all_labels)

accuracy = (all_preds == all_labels).mean()
print(f"MLP Validation Accuracy: {accuracy:.4f}")