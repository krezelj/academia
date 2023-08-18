from collections import defaultdict
import numpy as np


class QLAgent():

    __slots__ = ['q_table', 'n_actions', 'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'min_epsilon']

    def __init__(self, n_actions, alpha = 0.1, gamma=0.99, 
                 epsilon=1, epsilon_decay=0.999, min_epsilon=0.01) -> None:
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda : np.zeros(n_actions))

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


    def update(self, state, action, reward, new_state, is_terminal):
        self.q_table[state][action] =\
            (1-self.alpha)*self.q_table[state][action] +\
            self.alpha*(reward + self.gamma * np.max(self.q_table[new_state]))

    def decay_epsilon(self):
        self.epsilon = np.maximum(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, epsilon=1):
        self.epsilon = epsilon