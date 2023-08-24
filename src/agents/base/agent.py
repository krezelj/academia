from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):

    def __init__(self, epsilon=1, epsilon_decay=0.999, min_epsilon=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    @abstractmethod
    def get_action(self, state, legal_mask=None, greedy=False):
        pass

    @abstractmethod
    def update(self, state, action, reward: float, new_state, is_terminal: bool):
        pass

    def _update_epsilon(self):
        self.epsilon = np.maximum(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, epsilon=1):
        self.epsilon = epsilon
