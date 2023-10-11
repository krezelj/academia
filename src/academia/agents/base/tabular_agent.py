import json
import os
from abc import ABC
from collections import defaultdict

import numpy as np

from .agent import Agent


class TabularAgent(Agent, ABC):

    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1, epsilon_decay=0.999,
                 min_epsilon=0.01) -> None:
        super().__init__(epsilon, epsilon_decay, min_epsilon)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, legal_mask=None, greedy=False):
        qs = self.q_table[state]
        if legal_mask is not None:
            qs = (qs - np.min(qs)) * legal_mask + legal_mask
            # add legal mask in case best action shares value with an illegal action
        if greedy or np.random.uniform() > self.epsilon:
            return np.argmax(qs)
        elif legal_mask is not None:
            return np.random.choice(np.arange(0, self.n_actions), size=1, p=legal_mask/legal_mask.sum())[0]
        else:
            return np.random.randint(0, self.n_actions)

    def to_json(self, name: str):
        learner_state_dict = {
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
        }
        save_dir = './saved_agents/'
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/TabularAgent-{name}.json', 'w') as file:
            json.dump(learner_state_dict, file)

    def load_json(self, name: str):
        save_dir = './saved_agents/'
        with open(f'{save_dir}/TabularAgent-{name}.json', 'r') as file:
            learner_state_dict = json.load(file)
        self.n_actions = learner_state_dict['n_actions']
        self.alpha = learner_state_dict['alpha']
        self.gamma = learner_state_dict['gamma']
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), learner_state_dict['q_table'])
        self.epsilon = learner_state_dict['epsilon']
        self.epsilon_decay = learner_state_dict['epsilon_decay']
        self.min_epsilon = learner_state_dict['min_epsilon']
