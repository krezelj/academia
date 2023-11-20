import unittest
from unittest.mock import patch
import gymnasium
import numpy as np
from academia.environments import DoorKey


class TestDoorKey(unittest.TestCase):

    def setUp(self):
        # Set up the environment with mock values for testing
        self.sut = DoorKey(difficulty=1)

    def test_transform_action(self):
        # Test the _transform_action method
        state = self.sut.reset()
        # If mapping is correct, action 4 should be mapped to action 5 in the underlying environment
        # and agent should change his state, otherwise performing not allowed action state should be the same
        new_state, reward, is_episode_end = self.sut.step(1)
        self.assertNotEqual(state.tolist(), new_state.tolist())

        state = self.sut.reset()
        new_state, reward, is_episode_end = self.sut.step(6)  # Not allowed action
        self.assertEqual(state.tolist(), new_state.tolist())

    def test_observe(self):
        transformed_state = self.sut.reset()

        # Ensure the transformed state has the correct shape
        self.assertEqual((7 * 7 + 2,), transformed_state.shape)

        # Ensure the door status is properly updated
        self.assertEqual(2, self.sut._door_status)  # Initial state

        # Imitate the agent opening the door
        example_image_observation = np.ones((7, 7, 3), dtype=np.uint8)
        mocked_step_return = ({
                                  'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
                                  'image': example_image_observation,
                                  'mission': 'example_mission'  # Example of mission, not important for testing
                              }, 0., False, False, {})
        with patch.object(self.sut._base_env, 'step', return_value=mocked_step_return):
            self.sut.step(0)
            self.assertEqual(1, self.sut._door_status)  # Door opened

        example_image_observation = np.zeros((7, 7, 3), dtype=np.uint8)
        mocked_step_return = ({
                                  'direction': gymnasium.spaces.Discrete(4).sample(),  # Example discrete direction
                                  'image': example_image_observation,
                                  'mission': 'example_mission'  # Example of mission, not important for testing
                              }, 0., False, False, {})
        with patch.object(self.sut._base_env, 'step', return_value=mocked_step_return):
            self.sut.step(0)
            self.assertEqual(0, self.sut._door_status)

    def test_invalid_difficulty(self):
        # Test that an invalid difficulty level raises a ValueError
        with self.assertRaises(ValueError):
            DoorKey(difficulty=4)
        with self.assertRaises(ValueError):
            DoorKey(difficulty=-1)


if __name__ == '__main__':
    unittest.main()
