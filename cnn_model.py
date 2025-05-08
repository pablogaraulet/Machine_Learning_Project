import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 28, 28] → [B, 32, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → [B, 64, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # → [B, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),

            nn.Flatten(),                                # → [B, 64*14*14]
            nn.Linear(64*14*14, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)
