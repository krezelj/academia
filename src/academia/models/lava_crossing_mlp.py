import torch.nn as nn
import torch


class LavaCrossingMLP(nn.Module):

    def __init__(self):
        super(LavaCrossingMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(51, 160),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
