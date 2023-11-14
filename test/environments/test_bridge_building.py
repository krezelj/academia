import unittest

import numpy as np

from academia.environments import BridgeBuilding


class TestBridgeBuilding(unittest.TestCase):

    def setUp(self):
        self.max_steps = 100
        self.n_boulders_placed = 3
        self.start_with_boulder = False
        self.environment = BridgeBuilding(
            max_steps=self.max_steps,
            n_boulders_placed=self.n_boulders_placed,
            start_with_boulder=self.start_with_boulder
        )

    def test_reset(self):
        initial_state = self.environment.reset()
        self.assertEqual(self.environment.episode_steps, 0)
        self.assertIsNotNone(initial_state)

    def test_step(self):
        initial_state = self.environment.reset()
        new_state, reward, is_terminal = self.environment.step(0)
        self.assertIsInstance(reward, int)
        self.assertEqual(reward, self.environment.STEP_PENALTY)

        self.assertIsInstance(is_terminal, bool)
        self.assertEqual(is_terminal, False)

    def test_walk_valid(self):
        initial_state = self.environment.reset()
        initial_position = (0, 0)
        self.environment.player_position = initial_position
        initial_state = self.environment.observe()
        action = 0  # move left
        new_state, reward, is_terminal = self.environment.step(action)
        new_position = self.environment.player_position
        self.assertNotEqual(initial_position, new_position)
        self.assertNotEqual(initial_state, new_state)

    def test_walk_invalid(self):
        self.environment.reset()
        initial_position = (0, 2)
        self.environment.player_position = initial_position
        initial_state = self.environment.observe()
        action = 0  # move left
        new_state, reward, is_terminal = self.environment.step(action)
        new_position = self.environment.player_position
        self.assertEqual(initial_state, new_state)
        self.assertEqual(initial_position, new_position)

    def test_goal_reward(self):
        self.environment.reset()
        initial_player_position = (2, 2)
        initial_boulders_positions = [(3, 2), (4, 2), (5, 2)]
        self.environment.player_position = initial_player_position
        self.environment.boulder_positions = initial_boulders_positions
        action = 1  # move up
        self.environment.step(action)  # first step on bridge
        self.environment.step(action)  # second step on bridge
        self.environment.step(action)  # third step on bridge
        new_state, reward, is_terminal = self.environment.step(action)  # gain reward
        self.assertEqual(reward, 99)
        self.assertEqual(is_terminal, True)

    def test_drown_reward(self):
        initial_state = self.environment.reset()
        initial_player_position = (2, 2)
        initial_boulders_positions = [(3, 2), (4, 2), (5, 2)]
        self.environment.player_position = initial_player_position
        self.environment.boulder_positions = initial_boulders_positions
        move_up = 1
        move_right = 2
        self.environment.step(move_up)  # first step on bridge
        self.environment.step(move_up)  # second step on bridge
        new_state, reward, is_terminal = self.environment.step(move_right)  # agent falls to water
        self.assertEqual(reward, -101)
        self.assertEqual(is_terminal, True)

    def test_observe(self):
        initial_state = self.environment.reset()
        observed_state = self.environment.observe()
        self.assertIsInstance(observed_state, tuple)
        self.assertEqual(len(observed_state), 8)

    def test_get_legal_mask(self):
        legal_mask = self.environment.get_legal_mask()
        self.assertIsInstance(legal_mask, np.ndarray)
        self.assertEqual(len(legal_mask), self.environment.N_ACTIONS)


if __name__ == '__main__':
    unittest.main()
