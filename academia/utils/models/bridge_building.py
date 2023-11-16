import numpy as np
import torch
import torch.nn as nn

class MLPActor(nn.Module):

    def __init__(self):
        super(MLPActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        return self.network(obs)
    

class MLPCritic(nn.Module):

    def __init__(self):
        super(MLPCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
            
        return self.network(obs)
    