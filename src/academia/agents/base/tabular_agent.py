import os
from collections import defaultdict
from typing import Optional

import yaml
import numpy as np

from .agent import Agent


class TabularAgent(Agent):

    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1, epsilon_decay=0.999,
                 min_epsilon=0.01, q_table=None, random_state: Optional[int] = None) -> None:
        super().__init__(epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, 
                         n_actions=n_actions, gamma=gamma, random_state=random_state)
        self.alpha = alpha
        if q_table is None:
            self.q_table = defaultdict(lambda: np.zeros(n_actions))
        else:
            self.q_table = q_table

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

    def save(self, path: str):
        learner_state_dict = {
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'q_table': {str(key): value.tolist() for key, value in self.q_table.items()},
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'random_state': self._rng.bit_generator.state
        }
        if not path.endswith('.yml'):
            path += '.agent.yml'
        with open(path, 'w') as file:
            yaml.dump(dict(learner_state_dict), file)
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> 'TabularAgent':
        if not path.endswith('.yml'):
            path += '.agent.yml'

        with open(path, 'r') as file:
            learner_state_dict = yaml.safe_load(file)

        q_table = defaultdict(lambda: np.zeros(learner_state_dict['n_actions']))
        for key, value in learner_state_dict['q_table'].items():
            q_table[eval(key)] = np.array(value)
        del learner_state_dict['q_table']
        rng_state = learner_state_dict.pop('random_state')
        agent = cls(q_table=q_table, **learner_state_dict)
        agent._rng.bit_generator.state = rng_state
        return agent

    
