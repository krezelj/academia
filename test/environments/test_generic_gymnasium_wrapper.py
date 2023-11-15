import unittest
from unittest.mock import patch

import numpy as np

from academia.environments import LavaCrossing
from academia.environments.base import GenericGymnasiumWrapper


class TestGenericGymnasiumWrapper(unittest.TestCase):

    def setUp(self):
        # Set up the subclass of GenericGymnasiumWrapper environment with mock values for testing
        self.sut = LavaCrossing(difficulty=1)
        self.sut_stacked = LavaCrossing(difficulty=1,
                                        n_frames_stacked=2)
        self.sut_step_count = LavaCrossing(difficulty=1,
                                           append_step_count=True)

    def test_step(self):
        initial_state = self.sut.observe()
        action = 0
        new_state, reward, is_episode_end = self.sut.step(action)

        self.assertEqual(self.sut.step_count, 1)

        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_episode_end, bool)

        # Ensure the environment state is updated
        self.assertNotEqual(initial_state.tolist(), new_state.tolist())

    def test_observe(self):
        state = self.sut.observe()
        self.assertIsInstance(state, np.ndarray)

    def test_reset(self):
        initial_state = self.sut.reset()

        self.assertEqual(0, self.sut.step_count)

        self.assertNotEqual(initial_state.tolist(), np.zeros(self.sut.STATE_SHAPE).tolist())

    def test_get_legal_mask(self):
        # Test the get_legal_mask method
        legal_mask = self.sut.get_legal_mask()
        self.assertIsInstance(legal_mask, np.ndarray)
        self.assertEqual([1] * self.sut.N_ACTIONS, legal_mask.tolist())

    def test_state_representation(self):
        # Implement a concrete subclass of GenericGymnasiumWrapper
        class TestWrapper(GenericGymnasiumWrapper):
            def _transform_state(self, raw_state):
                return np.array(raw_state) * 2

        # as environment_id passed sth from gymnasium library
        sut_test_wrapper = TestWrapper(difficulty=1, environment_id='CartPole-v1')

        example_observation = np.ones(4, dtype=np.float32)
        mocked_step_return = (example_observation, 1.0, False, False, {})
        # Mock the base environment's step method
        with patch.object(sut_test_wrapper._base_env, 'step', return_value=mocked_step_return):
            sut_test_wrapper.reset()
            transformed_state, reward, is_done = sut_test_wrapper.step(0)
        self.assertIsInstance(transformed_state, np.ndarray)
        self.assertEqual([2, 2, 2, 2], transformed_state.tolist())

    def test_frame_stacking(self):
        self.sut_stacked.reset()
        initial_state = self.sut_stacked.observe()
        action = 0
        new_state, reward, is_episode_end = self.sut_stacked.step(action)

        self.assertEqual(1, self.sut_stacked.step_count)

        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_episode_end, bool)

        # Ensure the environment state is updated
        self.assertNotEqual( initial_state.tolist(), new_state.tolist())

        # Ensure the environment state is stacked
        self.assertEqual(((7 * 7 + 1) * 2,), new_state.shape)

    def test_step_append(self):
        self.sut_step_count.reset()
        initial_state = self.sut_step_count.observe()
        action = 0
        new_state, reward, is_episode_end = self.sut_step_count.step(action)

        self.assertEqual(1, self.sut_step_count.step_count)

        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_episode_end, bool)

        # Ensure the environment state is updated
        self.assertNotEqual(initial_state.tolist(), new_state.tolist())

        # Ensure the environment state is stacked
        self.assertEqual(((7 * 7 + 1) * 1 + 1,), new_state.shape)


if __name__ == '__main__':
    unittest.main()
