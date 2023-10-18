from .base import TabularAgent


class SarsaAgent(TabularAgent):

    __slots__ = ['q_table', 'n_actions', 'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'min_epsilon', 'random_state']

    def update(self, state, action, reward, new_state, is_terminal):
        policy_next_action = self.get_action(state)
        self.q_table[state][action] = \
            (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * self.q_table[new_state][policy_next_action])
