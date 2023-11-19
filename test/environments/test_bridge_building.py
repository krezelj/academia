import unittest
from unittest import mock

import numpy as np

from academia.environments import BridgeBuilding


class TestBridgeBuilding(unittest.TestCase):
    
    def __force_state(self, env, state, held_boulder_index=-1):
        def direction_from_offset(target):
            if target[1] == 1: return 0
            if target[0] == 1: return 1
            if target[1] == -1: return 2
            if target[0] == -1: return 3

        player_position = state[[0, 1]]
        player_target = state[[2, 3]]
        player_direction = direction_from_offset(player_target - player_position)
        setattr(env, '_BridgeBuilding__player_position', player_position)
        setattr(env, '_BridgeBuilding__player_direction', player_direction)
        setattr(env, '_BridgeBuilding__boulder_positions', state[4:].reshape(3, 2))
        setattr(env, '_BridgeBuilding__held_boulder_index', held_boulder_index)


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
        state = state.replace('-', '') # ignore negative signs
        self.assertIsInstance(state, str)
        self.assertEqual(10, len(state), msg=state)

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

    def test_legal_masks(self):
        # agent facing void, cannot move forward/pickup/drop
        init_state = np.array([0, 0, -1, 0, 3, 0, 4, 0, 5, 0])
        self.__force_state(self.sut, init_state)
        lm = self.sut.get_legal_mask()
        self.assertEqual(0, lm[2])
        self.assertEqual(0, lm[3])

        # agent facing clear terrain with boulder
        init_state = np.array([0, 0, 1, 0, -1, -1, 4, 0, 5, 0])
        self.__force_state(self.sut, init_state, held_boulder_index=0)
        lm = self.sut.get_legal_mask()
        self.assertEqual(1, lm[2])
        self.assertEqual(1, lm[3])

        # agent facing boulder without boulder, cannot move forward can pickup
        init_state = np.array([0, 0, 1, 0, 1, 0, 4, 0, 5, 0])
        self.__force_state(self.sut, init_state)
        lm = self.sut.get_legal_mask()
        self.assertEqual(0, lm[2])
        self.assertEqual(1, lm[3])

        # agent facing boulder with boulder, cannot move forward cannot drop
        init_state = np.array([0, 0, 1, 0, 1, 0, -1, -1, 5, 0])
        self.__force_state(self.sut, init_state, held_boulder_index=1)
        lm = self.sut.get_legal_mask()
        self.assertEqual(0, lm[2])
        self.assertEqual(0, lm[3])


    def test_random_state(self):
        sut_1 = BridgeBuilding(3, random_state=42)
        sut_2 = BridgeBuilding(3, random_state=42)

        self.assertTrue(np.all(sut_1._state == sut_2._state))
        # we are not testing if two seeds result in a different environment since 
        # there is a relatively small number of possible initial states
        # 2016 possible initial states for difficulty 3
        # 3024 possible initial states for difficulty 2
        # 864  possible initial states for difficulty 1
        # 108  possible initial states for difficulty 0
        # total 6012 (so unlikely but not impossible)

    def test_goal(self):
        max_steps = 2
        sut = BridgeBuilding(0, max_steps=max_steps)
        init_state = np.array([5, 0, 6, 0, 3, 0, 4, 0, 5, 0])
        self.__force_state(sut, init_state)
        _, reward, done = sut.step(2)
        self.assertEqual(0.5, reward)
        self.assertTrue(done)

    def test_drown(self):
        init_state = np.array([2, 2, 3, 2, 3, 0, 4, 0, 5, 0])
        self.__force_state(self.sut, init_state)
        _, reward, done = self.sut.step(2)
        self.assertEqual(0, reward)
        self.assertTrue(done)

    def test_max_steps(self):
        max_steps = 2
        sut = BridgeBuilding(0, max_steps=max_steps)
        init_state = np.array([2, 2, 1, 2, 3, 0, 4, 0, 5, 0])
        self.__force_state(sut, init_state)

        _, reward, done = sut.step(2)
        self.assertEqual(0, reward)
        self.assertFalse(done)

        _, reward, done = sut.step(2)
        self.assertEqual(0, reward)
        self.assertTrue(done)


if __name__ == '__main__':
    unittest.main()
