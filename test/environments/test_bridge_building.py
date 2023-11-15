import unittest

import numpy as np

from academia.environments import BridgeBuilding


class TestBridgeBuilding(unittest.TestCase):

    def setUp(self):
        self.max_steps = 100
        self.n_boulders_placed = 3
        self.start_with_boulder = False
        self.sut = BridgeBuilding(
            max_steps=self.max_steps,
            n_boulders_placed=self.n_boulders_placed,
            start_with_boulder=self.start_with_boulder
        )

    def test_reset(self):
        initial_state = self.sut.reset()
        self.assertEqual(0, self.sut.episode_steps)
        self.assertIsNotNone(initial_state)

    def test_step(self):
        initial_state = self.sut.reset()
        new_state, reward, is_terminal = self.sut.step(0)
        self.assertIsInstance(reward, int)
        self.assertEqual(self.sut.STEP_PENALTY, reward)

        self.assertIsInstance(is_terminal, bool)
        self.assertEqual(False, is_terminal)

    def test_walk_valid(self):
        initial_state = self.sut.reset()
        initial_position = (0, 0)
        self.sut.player_position = initial_position
        initial_state = self.sut.observe()
        action = 0  # move left
        new_state, reward, is_terminal = self.sut.step(action)
        new_position = self.sut.player_position
        self.assertNotEqual(initial_position, new_position)
        self.assertNotEqual(initial_state, new_state)

    def test_walk_invalid(self):
        self.sut.reset()
        initial_position = (0, 2)
        self.sut.player_position = initial_position
        initial_state = self.sut.observe()
        action = 0  # move left
        new_state, reward, is_terminal = self.sut.step(action)
        new_position = self.sut.player_position
        self.assertEqual(initial_state, new_state)
        self.assertEqual(initial_position, new_position)

    def test_goal_reward(self):
        self.sut.reset()
        initial_player_position = (2, 2)
        initial_boulders_positions = [(3, 2), (4, 2), (5, 2)]
        self.sut.player_position = initial_player_position
        self.sut.boulder_positions = initial_boulders_positions
        action = 1  # move up
        self.sut.step(action)  # first step on bridge
        self.sut.step(action)  # second step on bridge
        self.sut.step(action)  # third step on bridge
        new_state, reward, is_terminal = self.sut.step(action)  # gain reward
        self.assertEqual(99, reward)
        self.assertEqual(True, is_terminal)

    def test_drown_reward(self):
        initial_state = self.sut.reset()
        initial_player_position = (2, 2)
        initial_boulders_positions = [(3, 2), (4, 2), (5, 2)]
        self.sut.player_position = initial_player_position
        self.sut.boulder_positions = initial_boulders_positions
        move_up = 1
        move_right = 2
        self.sut.step(move_up)  # first step on bridge
        self.sut.step(move_up)  # second step on bridge
        new_state, reward, is_terminal = self.sut.step(move_right)  # agent falls to water
        self.assertEqual(-101, reward)
        self.assertEqual(True, is_terminal)

    def test_observe(self):
        initial_state = self.sut.reset()
        observed_state = self.sut.observe()
        self.assertIsInstance(observed_state, tuple)
        self.assertEqual(8, len(observed_state))
        self.assertEqual(initial_state, observed_state)

    def test_get_legal_mask(self):
        legal_mask = self.sut.get_legal_mask()
        self.assertIsInstance(legal_mask, np.ndarray)
        self.assertEqual(self.sut.N_ACTIONS, len(legal_mask))


if __name__ == '__main__':
    unittest.main()
