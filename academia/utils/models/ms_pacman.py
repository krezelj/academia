"""
See Also:
    - :class:`academia.environments.MsPacman`
"""
import torch.nn as nn
import torch


# ================= #
# ====== DQN ====== #
# ================= #

class MLPDQN(nn.Module):
    """
    Neural network architecture for DQN compatible with Ms Pacman environment with the
    following parameters:

    - ``n_frames_stacked=1``
    - ``append_step_count=False``
    - ``obs_type='ram'``
    """

    def __init__(self):
        super(MLPDQN, self).__init__()
        layer_in = 128
        layer_1 = 512
        layer_2 = 512
        layer_out = 9
        self.network = nn.Sequential(
            nn.Linear(layer_in, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_out),
        )

    def forward(self, x) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.network(x)


class MLPStepDQN(nn.Module):
    """
    Neural network architecture for DQN compatible with Ms Pacman environment with the
    following parameters:

    - ``n_frames_stacked=1``
    - ``append_step_count=True``
    - ``obs_type='ram'``
    """

    def __init__(self):
        super(MLPStepDQN, self).__init__()
        layer_in = 129
        layer_1 = 512
        layer_2 = 512
        layer_out = 9
        self.network = nn.Sequential(
            nn.Linear(layer_in, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_out),
        )

    def forward(self, x) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.network(x)


# ================= #
# ====== PPO ====== #
# ================= #

class MLPActor(nn.Module):
    """
    Neural network architecture for PPO (actor) compatible with Ms Pacman environment with the
    following parameters:

    - ``n_frames_stacked=1``
    - ``append_step_count=False``
    - ``obs_type='ram'``
    """

    def __init__(self):
        super(MLPActor, self).__init__()
        layer_in = 128
        layer_1 = 512
        layer_2 = 512
        layer_out = 9
        self.network = nn.Sequential(
            nn.Linear(layer_in, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_out),
        )

    def forward(self, obs) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.network(obs)


class MLPCritic(nn.Module):
    """
    Neural network architecture for PPO (critic) compatible with Ms Pacman environment with the
    following parameters:

    - ``n_frames_stacked=1``
    - ``append_step_count=False``
    - ``obs_type='ram'``
    """

    def __init__(self):
        super(MLPCritic, self).__init__()
        layer_in = 128
        layer_1 = 512
        layer_2 = 512
        layer_out = 1
        self.network = nn.Sequential(
            nn.Linear(layer_in, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_out),
        )

    def forward(self, obs) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.network(obs)


class MLPStepActor(nn.Module):
    """
    Neural network architecture for PPO (actor) compatible with Ms Pacman environment with the
    following parameters:

    - ``n_frames_stacked=1``
    - ``append_step_count=True``
    - ``obs_type='ram'``
    """

    def __init__(self):
        super(MLPStepActor, self).__init__()
        layer_in = 129
        layer_1 = 512
        layer_2 = 512
        layer_out = 9
        self.network = nn.Sequential(
            nn.Linear(layer_in, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_out),
        )

    def forward(self, obs) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.network(obs)


class MLPStepCritic(nn.Module):

    """
    Neural network architecture for PPO (critic) compatible with Ms Pacman environment with the
    following parameters:

    - ``n_frames_stacked=1``
    - ``append_step_count=True``
    - ``obs_type='ram'``
    """

    def __init__(self):
        super(MLPStepCritic, self).__init__()
        layer_in = 129
        layer_1 = 512
        layer_2 = 512
        layer_out = 1
        self.network = nn.Sequential(
            nn.Linear(layer_in, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_out),
        )

    def forward(self, obs) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.network(obs)
