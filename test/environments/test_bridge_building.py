import unittest
from unittest import mock

import numpy as np

from academia.environments import BridgeBuilding


class TestBridgeBuilding(unittest.TestCase):
    
    def setUp(self):
        self.max_steps = 100
        self.difficulty = 3
        self.sut = BridgeBuilding(
            difficulty = self.difficulty,
            max_steps=self.max_steps
        )

    def test_string_obs_type(self):
        sut = BridgeBuilding(3, obs_type="string")
        state = sut.reset()
        self.assertIsInstance(state, str)
        self.assertEqual(10, len(state))

    def test_array_obs_type(self):
        sut = BridgeBuilding(3, obs_type="array")
        state = sut.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(np.float32, state.dtype)
        self.assertEqual(10, len(state))

    def test_step(self):
        self.sut.reset()
        _, reward, is_terminal = self.sut.step(0)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_terminal, bool)
        self.assertEqual(False, is_terminal)

    def test_reset(self):
        initial_state = self.sut.reset()
        self.assertEqual(0, self.sut.step_count)
        self.assertIsNotNone(initial_state)

    def test_observe(self):
        initial_state = self.sut.reset()
        observed_state = self.sut.observe()
        self.assertEqual(10, len(observed_state))
        self.assertTrue(np.all(initial_state == observed_state))

    def test_append_step_count(self):
        sut = BridgeBuilding(3, append_step_count=True)
        state = sut.reset()
        self.assertEqual(11, len(state))

    def test_n_frames_stacked(self):
        sut = BridgeBuilding(3, n_frames_stacked=2)
        state = sut.reset()
        self.assertEqual(20, len(state))

    def test_invalid_difficulty(self):
        with self.assertRaises(ValueError):
            with mock.patch.object(BridgeBuilding, '_BridgeBuilding__RIVER_WIDTH', 3):
                BridgeBuilding(difficulty=4)
                BridgeBuilding(difficulty=-1)

    def test_get_legal_mask(self):
        legal_mask = self.sut.get_legal_mask()
        self.assertIsInstance(legal_mask, np.ndarray)
        self.assertEqual(self.sut.N_ACTIONS, len(legal_mask))


if __name__ == '__main__':
    unittest.main()
