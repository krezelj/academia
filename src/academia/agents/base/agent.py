from abc import abstractmethod
from typing import Optional

import numpy as np

from academia.utils import SavableLoadable


class Agent(SavableLoadable):

    def __init__(self, n_actions: int, epsilon: float = 1.,
                 epsilon_decay: float = 0.999, min_epsilon: float = 0.01,
                 gamma: float = 0.99, random_state: Optional[int] = None):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = n_actions
        self.gamma = gamma
        self._rng = np.random.default_rng(seed=random_state)

    @abstractmethod
    def get_action(self, state, legal_mask=None, greedy=False):
        pass

    @abstractmethod
    def update(self, state, action, reward: float, new_state, is_terminal: bool):
        pass

    def decay_epsilon(self):
        self.epsilon = np.maximum(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, epsilon=1):
        self.epsilon = epsilon
