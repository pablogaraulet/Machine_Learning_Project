import torch.nn as nn

class MLP(nn.Module):
    def _init_(self, input_dim=784, hidden_dim=256, output_dim=10, dropout_prob=0.5):
        super(MLP, self)._init_()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)