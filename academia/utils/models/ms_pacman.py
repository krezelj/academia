import numpy as np
import torch.nn as nn
import torch

class MLPActor(nn.Module):

    def __init__(self):
        super(MLPActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
            nn.Softmax(dim=1)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif obs.dtype != torch.float32:
            obs = obs.float()

        return self.network(obs)
    
class MLPCritic(nn.Module):

    def __init__(self):
        super(MLPCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif obs.dtype != torch.float32:
            obs = obs.float()

        return self.network(obs)