from abc import abstractmethod
from typing import Optional, Any

import numpy as np
import numpy.typing as npt

from academia.utils import SavableLoadable


class Agent(SavableLoadable):
    """
    Agent class represents a generic reinforcement learning agent.

    This class serves as the base class for various reinforcement learning agents.
    It defines common attributes and methods necessary for interacting with environments.

    Args:
        n_actions: Number of possible actions in the environment.
        gamma: Discount factor. Defaults to 0.99.
        random_state: Seed for the random number generator. Defaults to ``None``.

    Attributes:
        n_actions (int): Number of possible actions in the environment.
        gamma (float): Discount factor.
    """

    def __init__(self, n_actions: int, gamma: float = 0.99, random_state: Optional[int] = None):
        self.n_actions = n_actions
        self.gamma = gamma
        self._rng = np.random.default_rng(seed=random_state)

    @abstractmethod
    def get_action(self, state: Any, legal_mask: npt.NDArray[np.int32] = None, greedy: bool = False) -> int:
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
    def update(self, state: Any, action: int, reward: float, new_state: Any, is_terminal: bool):
        """
        Updates the agent's knowledge based on the observed reward and new state.

        Args:
            state: Current state in the environment.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            new_state: New state observed after taking the action.
            is_terminal: Whether the new state is a terminal state or not.
        """
        pass

    @abstractmethod
    def update_exploration(self):
        """
        Updates the exploration parameter.
        """
        pass

    @abstractmethod
    def reset_exploration(self, value):
        """
        Resets the exploration parameter to the specified value.

        Args:
            value: Value to reset the parameter to.
        """
        pass
