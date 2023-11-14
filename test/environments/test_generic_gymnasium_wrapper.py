import unittest
from unittest.mock import patch

import numpy as np

from academia.environments import LavaCrossing
from academia.environments.base import GenericGymnasiumWrapper


class TestGenericGymnasiumWrapper(unittest.TestCase):

    def setUp(self):
        # Set up the subclass of GenericGymnasiumWrapper environment with mock values for testing
        self.env = LavaCrossing(difficulty=1)
        self.env_stacked = LavaCrossing(difficulty=1,
                                        n_frames_stacked=2)
        self.env_step_count = LavaCrossing(difficulty=1,
                                           append_step_count=True)

    def test_step(self):
        initial_state = self.env.observe()
        action = 0
        new_state, reward, is_episode_end = self.env.step(action)

        self.assertEqual(self.env.step_count, 1)

        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_episode_end, bool)

        # Ensure the environment state is updated
        self.assertNotEqual(initial_state.tolist(), new_state.tolist())

    def test_observe(self):
        state = self.env.observe()
        self.assertIsInstance(state, np.ndarray)

    def test_reset(self):
        initial_state = self.env.reset()

        self.assertEqual(0, self.env.step_count)

        self.assertNotEqual(initial_state.tolist(), np.zeros(self.env.STATE_SHAPE).tolist())

    def test_get_legal_mask(self):
        # Test the get_legal_mask method
        legal_mask = self.env.get_legal_mask()
        self.assertIsInstance(legal_mask, np.ndarray)
        self.assertEqual([1] * self.env.N_ACTIONS, legal_mask.tolist())

    def test_state_representation(self):
        # Implement a concrete subclass of GenericGymnasiumWrapper
        class TestWrapper(GenericGymnasiumWrapper):
            def _transform_state(self, raw_state):
                return np.array(raw_state) * 2

        # as environment_id passed sth from gymnasium library
        test_wrapper = TestWrapper(difficulty=1, environment_id='CartPole-v1')

        example_observation = np.ones(4, dtype=np.float32)
        mocked_step_return = (example_observation, 1.0, False, False, {})
        # Mock the base environment's step method
        with patch.object(test_wrapper._base_env, 'step', return_value=mocked_step_return):
            test_wrapper.reset()
            transformed_state, reward, is_done = test_wrapper.step(0)
        self.assertIsInstance(transformed_state, np.ndarray)
        self.assertEqual([2, 2, 2, 2], transformed_state.tolist())

    def test_frame_stacking(self):
        self.env_stacked.reset()
        initial_state = self.env_stacked.observe()
        action = 0
        new_state, reward, is_episode_end = self.env_stacked.step(action)

        self.assertEqual(1, self.env_stacked.step_count)

        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_episode_end, bool)

        # Ensure the environment state is updated
        self.assertNotEqual( initial_state.tolist(), new_state.tolist())

        # Ensure the environment state is stacked
        self.assertEqual(((7 * 7 + 1) * 2,), new_state.shape)

    def test_step_append(self):
        self.env_step_count.reset()
        initial_state = self.env_step_count.observe()
        action = 0
        new_state, reward, is_episode_end = self.env_step_count.step(action)

        self.assertEqual(1, self.env_step_count.step_count)

        self.assertIsInstance(new_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_episode_end, bool)

        # Ensure the environment state is updated
        self.assertNotEqual(initial_state.tolist(), new_state.tolist())

        # Ensure the environment state is stacked
        self.assertEqual(((7 * 7 + 1) * 1 + 1,), new_state.shape)


if __name__ == '__main__':
    unittest.main()
