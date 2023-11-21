import torch.nn as nn


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
        return self.network(obs)
