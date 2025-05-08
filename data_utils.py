import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def get_flattened_loader(dataset_name="MNIST", batch_size=64, train=True):
    """
    Loads MNIST or EMNIST dataset and returns DataLoader with flattened inputs.
    """
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 to 784
        ])
        dataset = datasets.MNIST(root="data", train=train, download=True, transform=transform)

    elif dataset_name == "EMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 to 784
        ])
        dataset = datasets.EMNIST(root="data", split='balanced', train=train, download=True, transform=transform)

    else:
        raise ValueError("Unsupported dataset. Use 'MNIST' or 'EMNIST'.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader

def get_cnn_loader(dataset_name="MNIST", batch_size=64, train=True):
    """
    Loads MNIST or EMNIST dataset and returns DataLoader with 2D inputs for CNN.
    """
    transform = transforms.ToTensor()  # No flattening, keep shape [1, 28, 28]

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root="data", train=train, download=True, transform=transform)
    elif dataset_name == "EMNIST":
        dataset = datasets.EMNIST(root="data", split='balanced', train=train, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Use 'MNIST' or 'EMNIST'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
