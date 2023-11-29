from typing import Optional

import numpy as np

from . import Agent


class EpsilonGreedyAgent(Agent):
    """
    A base class for all epsilon-greedy reinforcement learning algorithms.

    Args:
        n_actions: Number of possible actions in the environment.
        gamma: Discount factor. Defaults to 0.99.
        random_state: Seed for the random number generator. Defaults to ``None``.
        epsilon: Exploration-exploitation trade-off parameter. Defaults to 1.
        min_epsilon: Minimum value for epsilon during exploration. Defaults to 0.01.
        epsilon_decay: Decay rate for epsilon. Defaults to 0.999.

    Attributes:
        n_actions (int): Number of possible actions in the environment.
        gamma (float): Discount factor.
        epsilon (float): Exploration-exploitation trade-off parameter.
        min_epsilon (float): Minimum value for epsilon during exploration.
        epsilon_decay (float): Decay rate for epsilon.
    """

    def __init__(self, n_actions: int, epsilon: float = 1.,
                 epsilon_decay: float = 0.999, min_epsilon: float = 0.01,
                 gamma: float = 0.99, random_state: Optional[int] = None):
        super().__init__(
            n_actions=n_actions,
            gamma=gamma,
            random_state=random_state
        )
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def update_exploration(self):
        """
        Decays the exploration parameter epsilon based on epsilon_decay.
        """
        self.epsilon = float(np.max([self.min_epsilon, self.epsilon * self.epsilon_decay]))

    def reset_exploration(self, value=1):
        """
        Resets the exploration parameter epsilon to the specified value.

        Args:
            value: Value to reset epsilon to. Defaults to 1.
        """
        self.epsilon = value
