import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_flattened_loader
from mlp_model import MLP

def train_mlp(dataset_name="MNIST", output_file="mlp_mnist.pt", num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_flattened_loader(dataset_name=dataset_name, train=True)
    val_loader = get_flattened_loader(dataset_name=dataset_name, train=False)

    model = MLP(output_dim=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{dataset_name}] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Save trainig model
    torch.save(model.state_dict(), output_file)
    print(f"Model saved to {output_file}")

# Trainings
train_mlp("MNIST", "mlp_mnist.pt", num_classes=10)
train_mlp("EMNIST", "mlp_emnist.pt", num_classes=47)