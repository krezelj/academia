import torch.nn as nn


class CartPoleMLP(nn.Module):
    def __init__(self):
        super(CartPoleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def forward(self, x):
        return self.network(x)
