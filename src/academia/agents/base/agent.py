from abc import abstractmethod
from typing import Optional

import numpy as np

from academia.utils import SavableLoadable


class Agent(SavableLoadable):
    """
    Agent class represents a generic reinforcement learning agent.

    This class serves as the base class for various reinforcement learning agents.
    It defines common attributes and methods necessary for interacting with environments.

    Args:
        n_actions: Number of possible actions in the environment.
        epsilon: Exploration-exploitation trade-off parameter. Defaults to 1.
        epsilon_decay: Decay rate for epsilon. Defaults to 0.999.
        min_epsilon: Minimum value for epsilon during exploration. Defaults to 0.01.
        gamma: Discount factor. Defaults to 0.99.
        random_state: Seed for the random number generator. Defaults to ``None``.

    Attributes:
        epsilon (float): Exploration-exploitation trade-off parameter.
        min_epsilon (float): Minimum value for epsilon during exploration.
        epsilon_decay (float): Decay rate for epsilon.
        n_actions (int): Number of possible actions in the environment.
        gamma (float): Discount factor.
    """

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
    def get_action(self, state, legal_mask=None, greedy=False) -> int:
        """
        Gets an action for the given state.

        Args:
            state: Current state in the environment.
            legal_mask: A mask representing legal actions in the current state.
            greedy: Whether to choose the greedy action. Defaults to False.

        Returns:
            Action to be taken in the given state.
        """
        pass

    @abstractmethod
    def update(self, state, action, reward: float, new_state, is_terminal: bool):
        """
        Abstract method to update the agent's knowledge based on the observed reward and new state.

        Args:
            state: Current state in the environment.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            new_state: New state observed after taking the action.
            is_terminal: Whether the new state is a terminal state or not.
        """
        pass

    def decay_epsilon(self):
        """
        Decay the exploration parameter epsilon based on epsilon_decay.
        """
        self.epsilon = float(np.max([self.min_epsilon, self.epsilon * self.epsilon_decay]))

    def reset_epsilon(self, epsilon=1):
        """
        Reset the exploration parameter epsilon to the specified value.

        Args:
            epsilon: Value to reset epsilon to. Defaults to 1.
        """
        self.epsilon = epsilon
