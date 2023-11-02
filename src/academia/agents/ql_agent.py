from typing import Any

import numpy as np

from .base import TabularAgent


class QLAgent(TabularAgent):
    """
    QLAgent class implements a Q-learning agent for tabular environments.

    This agent learns to make decisions in an environment with discrete states and actions
    by maintaining a Q-table, which represents the quality of taking a certain action
    in a specific state.

    Args:
        n_actions: Number of possible actions in the environment.
        alpha: Learning rate. Defaults to 0.1.
        gamma: Discount factor. Defaults to 0.99.
        epsilon: Exploration-exploitation trade-off parameter. Defaults to 1.
        epsilon_decay: Decay rate for epsilon. Defaults to 0.999.
        min_epsilon: Minimum value for epsilon during exploration. Defaults to 0.01.
        random_state: Seed for the random number generator.

    Raises:
        ValueError: If the given state is not supported.

    Attributes:
        epsilon (float): Exploration-exploitation trade-off parameter.
        min_epsilon (float): Minimum value for epsilon during exploration.
        epsilon_decay (float): Decay rate for epsilon.
        n_actions (int): Number of possible actions in the environment.
        gamma (float): Discount factor.
        alpha (float): Learning rate.
        q_table (dict): Q-table for the agent.
    """

    __slots__ = ['q_table', 'n_actions', 'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'min_epsilon', 
                 'random_state']

    def update(self, state: Any, action: int, reward: float, new_state: Any, is_terminal: bool):
        """
        Updates the Q-value for the given state-action pair based on the observed reward and new state
        according to update strategy defined in Q-learning algorithm.

        Args:
            state: Current state in the environment.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            new_state: New state observed after taking the action.
            is_terminal: Whether the new state is a terminal state or not.
        """
        self.q_table[state][action] =\
            (1-self.alpha)*self.q_table[state][action] \
            + self.alpha*(reward + self.gamma * np.max(self.q_table[new_state]))
