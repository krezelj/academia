import torch.nn as nn
import torch


class MLPStepDQN(nn.Module):

    def __init__(self):
        super(MLPStepDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
