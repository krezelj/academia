import os
from collections import defaultdict
from typing import Optional, Any
import numbers
import json

import numpy as np
import numpy.typing as npt

from . import EpsilonGreedyAgent


class TabularAgent(EpsilonGreedyAgent):
    """
    TabularAgent class implements a reinforcement learning agent for simple environments
    where a Q-table can be effectively used.

    This class serves as the base class for tabular agents such as :class:`academia.agents.QLAgent` 
    and :class:`academia.agents.SarsaAgent`. This agent learns to make decisions in an 
    environment with discrete states and actions by maintaining a Q-table, which represents the quality of 
    taking a certain actionin a specific state.
    
    Args:
        n_actions: Number of possible actions in the environment.
        alpha: Learning rate. Defaults to 0.1.
        gamma: Discount factor. Defaults to 0.99.
        epsilon: Exploration-exploitation trade-off parameter. Defaults to 1.
        epsilon_decay: Decay rate for epsilon. Defaults to 0.999.
        min_epsilon: Minimum value for epsilon during exploration. Defaults to 0.01.
        random_state: Seed for the random number generator. Defaults to ``None``.

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
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1, epsilon_decay=0.999,
                 min_epsilon=0.01, random_state: Optional[int] = None) -> None:
        super().__init__(epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, 
                         n_actions=n_actions, gamma=gamma, random_state=random_state)
        self.alpha = alpha
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state: Any, legal_mask: npt.NDArray[np.int32] = None, greedy: bool = False) -> int:
        """
        Gets an action for the given state using epsilon-greedy policy.

        Args:
            state: Current state in the environment.
            legal_mask: A mask representing legal actions in the current state.
            greedy: Whether to choose the greedy action. Defaults to False.

        Returns:
            int: Action to be taken in the given state.
        """
        qs = self.q_table[state]
        if legal_mask is not None:
            qs = (qs - np.min(qs)) * legal_mask + legal_mask
            # add legal mask in case best action shares value with an illegal action
        if greedy or self._rng.uniform() > self.epsilon:
            return np.argmax(qs)
        elif legal_mask is not None:
            return self._rng.choice(np.arange(0, self.n_actions), size=1, p=legal_mask/legal_mask.sum())[0]
        else:
            return self._rng.integers(0, self.n_actions)

    def save(self, path: str) -> str:
        """
        Saves the agent's state to a JSON file.

        Args:
            path: Path to save the JSON file.

        Returns:
            An absolute path to the saved file.
        """
        q_table_keys = list(self.q_table.keys())
        if len(q_table_keys) > 0 and not self._validate_state(q_table_keys[0]):
            raise ValueError(f'Tabular agents only support numerical and string states')
        q_table_values = [val.tolist() for val in self.q_table.values()]
        learner_state_dict = {
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'q_table_keys': q_table_keys,
            'q_table_values': q_table_values,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'random_state': self._rng.bit_generator.state
        }
        if not path.endswith('.json'):
            path += '.agent.json'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(learner_state_dict, file)
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> 'TabularAgent':
        """
        Loads the agent's state from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A loaded agent with the saved state.
        """
        if not path.endswith('.json'):
            path += '.agent.json'
        with open(path, 'r') as file:
            learner_state_dict = json.load(file)

        q_table_keys = learner_state_dict.pop('q_table_keys')
        if len(q_table_keys) > 0 and not TabularAgent._validate_state(q_table_keys[0]):
            raise ValueError(f'Tabular agents only support numerical and string states')
        q_table_values = [np.array(val) for val in learner_state_dict.pop('q_table_values')]
        q_table = dict(zip(q_table_keys, q_table_values))

        rng_state = learner_state_dict.pop('random_state')
        agent = cls(**learner_state_dict)
        agent.q_table = q_table
        agent._rng.bit_generator.state = rng_state
        return agent

    @staticmethod
    def _validate_state(state: Any) -> bool:
        """
        Checks whether a state is compatible with TabularAgent.

        Args:
            state: a state to validate.

        Returns:
            True if this type of state is supported, False otherwise.
        """
        return isinstance(state, numbers.Number) or isinstance(state, str)
