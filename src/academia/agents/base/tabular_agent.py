import os
from collections import defaultdict
from typing import Optional, Any
import numbers

import yaml
import numpy as np

from .agent import Agent


class TabularAgent(Agent):

    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1, epsilon_decay=0.999,
                 min_epsilon=0.01, random_state: Optional[int] = None) -> None:
        super().__init__(epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, 
                         n_actions=n_actions, gamma=gamma, random_state=random_state)
        self.alpha = alpha
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, legal_mask=None, greedy=False):
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
        if not path.endswith('.yml'):
            path += '.agent.yml'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            yaml.dump(learner_state_dict, file)
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> 'TabularAgent':
        if not path.endswith('.yml'):
            path += '.agent.yml'
        with open(path, 'r') as file:
            learner_state_dict = yaml.safe_load(file)

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
        Checks whether a state is compatible with TabularAgent

        Args:
            state: a state to validate

        Returns:
            bool: ``True`` if this type of state is supported or ``False`` otherwise
        """
        return isinstance(state, numbers.Number) or isinstance(state, str)
