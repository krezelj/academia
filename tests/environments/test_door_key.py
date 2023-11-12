import unittest
from unittest.mock import patch
import gymnasium
import numpy as np
from academia.environments import DoorKey


class TestDoorKey(unittest.TestCase):

    def setUp(self):
        # Set up the environment with mock values for testing
        self.env = DoorKey(difficulty=1)

    def test_transform_action(self):
        # Test the _transform_action method
        state = self.env.reset()
        # If mapping is correct, action 4 should be mapped to action 5 in the underlying environment
        # and agent should change his state, otherwise performing not allowed action state should be the same
        new_state, reward, is_episode_end = self.env.step(1)
        self.assertNotEqual(state.tolist(), new_state.tolist())

        state = self.env.reset()
        new_state, reward, is_episode_end = self.env.step(6)  # Not allowed action
        self.assertEqual(state.tolist(), new_state.tolist())

    def test_observe(self):
        transformed_state = self.env.reset()

        # Ensure the transformed state has the correct shape
        self.assertEqual(transformed_state.shape, (7 * 7 + 2,))

        # Ensure the door status is properly updated
        self.assertEqual(self.env._door_status, 2)  # Initial state

        # Imitate the agent opening the door
        example_image_observation = np.ones((7, 7, 3), dtype=np.uint8)
        mocked_step_return = ({
                                  'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
                                  'image': example_image_observation,
                                  'mission': 'example_mission'  # Example of mission, not important for testing
                              }, 0., False, False, {})
        with patch.object(self.env._base_env, 'step', return_value=mocked_step_return):
            self.env.step(0)
            self.assertEqual(self.env._door_status, 1)  # Door opened

        example_image_observation = np.zeros((7, 7, 3), dtype=np.uint8)
        mocked_step_return = ({
                                  'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
                                  'image': example_image_observation,
                                  'mission': 'example_mission'  # Example of mission, not important for testing
                              }, 0., False, False, {})
        with patch.object(self.env._base_env, 'step', return_value=mocked_step_return):
            self.env.step(0)
            self.assertEqual(self.env._door_status, 0)

    def test_invalid_difficulty(self):
        # Test that an invalid difficulty level raises a ValueError
        with self.assertRaises(ValueError):
            DoorKey(difficulty=4)


if __name__ == '__main__':
    unittest.main()
