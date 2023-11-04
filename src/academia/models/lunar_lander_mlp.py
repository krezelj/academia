import torch.nn as nn
import torch


class LunarLanderMLP(nn.Module):

    def __init__(self):
        super(LunarLanderMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
