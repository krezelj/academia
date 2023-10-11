import numpy as np

from .base import TabularAgent


class QLAgent(TabularAgent):

    __slots__ = ['q_table', 'n_actions', 'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'min_epsilon']

    def update(self, state, action, reward, new_state, is_terminal):
        self.q_table[state][action] =\
            (1-self.alpha)*self.q_table[state][action] +\
            self.alpha*(reward + self.gamma * np.max(self.q_table[new_state]))
