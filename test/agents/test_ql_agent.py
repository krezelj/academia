import unittest

import numpy as np
from academia.agents import QLAgent

class TestQLAgent(unittest.TestCase):

    def test_update(self):
        # arrange
        alpha = 0.1
        gamma = 0.9
        agent = QLAgent(n_actions=3, alpha=alpha, gamma=gamma, random_state=0)

        mock_state = "mock_state"
        mock_new_state = "mock_new_state"
        init_q_values = {
            mock_state: np.array([0.0, 0.0, 1.0]),
            mock_new_state: np.array([1.0, 2.0, 1.0])
        }

        agent.q_table[mock_state] = init_q_values[mock_state].copy()
        agent.q_table[mock_new_state] = init_q_values[mock_new_state].copy()

        action = 0
        reward = 5

        # act
        agent.update(
            state=mock_state, 
            action=action, 
            reward=reward, 
            new_state=mock_new_state,
            is_terminal=False)
        
        # assert
        # Q(s,a) <- (1-alpha)*Q(s,a)+alpha*(r+gamma*max_a Q(s',a))
        expected_q_value = (1 - alpha) * init_q_values[mock_state][action] \
            + alpha * (reward + gamma * np.max(init_q_values[mock_new_state]))

        # almost equal because of floating point operations
        self.assertAlmostEqual(agent.q_table[mock_state][action], expected_q_value, 5)


if __name__ == '__main__':
    unittest.main()