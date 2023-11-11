import unittest
from unittest.mock import Mock, patch
import numpy as np
import gymnasium
from academia.environments.base import GenericGymnasiumWrapper

class TestGenericGymnasiumWrapper(unittest.TestCase):

    def setUp(self):
        # Set up the environment with mock values for testing
        self.env = GenericGymnasiumWrapper(difficulty=1, environment_id='MiniGrid-LavaCrossingS9N1-v0')
        self.env_stacked = GenericGymnasiumWrapper(difficulty=1, 
                                                   environment_id='MiniGrid-LavaCrossingS9N1-v0', 
                                                   n_frames_stacked=2)
        self.env_step_count = GenericGymnasiumWrapper(difficulty=1,
                                                        environment_id='MiniGrid-LavaCrossingS9N1-v0',
                                                        append_step_count=True)

    def test_step(self):
        example_image_observation = np.ones((7, 7, 3), dtype=np.uint8)   
        mocked_step_return = ({
            'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
            'image': example_image_observation,
            'mission': 'example_mission'  # Example of mission, not important for testing
        }, 0., False, False, {})  # Sample return values, modify based on your actual environment
        # Mock the base environment's step method
        with patch.object(self.env._base_env, 'step', return_value=mocked_step_return):
            initial_state = self.env.observe()
            action = 0
            new_state, reward, is_episode_end = self.env.step(action)

            self.assertEqual(self.env.step_count, 1)

            self.assertIsInstance(new_state, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(is_episode_end, bool)

            # Ensure the environment state is updated
            self.assertNotEqual(new_state.tolist(), initial_state.tolist())

    def test_observe(self):
        # Test the observe method
        state = self.env.observe()
        self.assertIsInstance(state, np.ndarray)

    def test_reset(self):
        example_image_observation = np.ones((7, 7, 3), dtype=np.uint8) * 128  
        mocked_step_return = ({
            'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
            'image': example_image_observation,
            'mission': 'example_mission'  # Example of mission, not important for testing
        }, 0., False, False, {})  # Sample return values, modify based on your actual environment
        # Mock the base environment's reset method
        with patch.object(self.env._base_env, 'reset', return_value=mocked_step_return):
            initial_state = self.env.reset()

            self.assertEqual(self.env.step_count, 0)

            self.assertNotEqual(initial_state.tolist(), np.zeros(self.env.STATE_SHAPE).tolist())

    def test_render(self):
        self.env.render()

    def test_get_legal_mask(self):
        # Test the get_legal_mask method
        legal_mask = self.env.get_legal_mask()
        self.assertIsInstance(legal_mask, np.ndarray)
        self.assertEqual(legal_mask.tolist(), [1] * self.env.N_ACTIONS)

    def test_transform_state(self):
        # Implement a concrete subclass of GenericGymnasiumWrapper and test its _transform_state method
        class TestWrapper(GenericGymnasiumWrapper):
            def _transform_state(self, raw_state):
                return np.array(raw_state) * 2

        test_wrapper = TestWrapper(difficulty=1, environment_id='test_env')
        transformed_state = test_wrapper._transform_state([1, 2, 3])

        self.assertIsInstance(transformed_state, np.ndarray)
        self.assertEqual(transformed_state.tolist(), [2, 4, 6])

    def test_frame_stacking(self):
        # Test frame stacking
        example_image_observation = np.ones((7, 7, 3), dtype=np.uint8)   
        mocked_step_return = ({
            'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
            'image': example_image_observation,
            'mission': 'example_mission'  # Example of mission, not important for testing
        }, 0., False, False, {})  # Sample return values, modify based on your actual environment
        # Mock the base environment's step method
        with patch.object(self.env_stacked._base_env, 'step', return_value=mocked_step_return):
            self.env_stacked.reset()
            initial_state = self.env_stacked.observe()
            action = 0
            new_state, reward, is_episode_end = self.env_stacked.step(action)

            self.assertEqual(self.env_stacked.step_count, 1)

            self.assertIsInstance(new_state, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(is_episode_end, bool)

            # Ensure the environment state is updated
            self.assertNotEqual(new_state.tolist(), initial_state.tolist())

            # Ensure the environment state is stacked
            self.assertEqual(new_state.shape, ((7 * 7 + 1) * 2,))
    
    def test_step_append(self):
        example_image_observation = np.ones((7, 7, 3), dtype=np.uint8)   
        mocked_step_return = ({
            'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
            'image': example_image_observation,
            'mission': 'example_mission'  # Example of mission, not important for testing
        }, 0., False, False, {})
        # Mock the base environment's step method
        with patch.object(self.env_step_count._base_env, 'step', return_value=mocked_step_return):
            self.env_step_count.reset()
            initial_state = self.env_step_count.observe()
            action = 0
            new_state, reward, is_episode_end = self.env_step_count.step(action)

            self.assertEqual(self.env_step_count.step_count, 1)

            self.assertIsInstance(new_state, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(is_episode_end, bool)

            # Ensure the environment state is updated
            self.assertNotEqual(new_state.tolist(), initial_state.tolist())

            # Ensure the environment state is stacked
            self.assertEqual(new_state.shape, ((7 * 7 + 1) * 1 + 1,))
        
if __name__ == '__main__':
    unittest.main()
