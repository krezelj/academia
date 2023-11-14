import unittest

from academia.environments import LavaCrossing


class TestLavaCrossing(unittest.TestCase):
    def setUp(self):
        # Set up the environment with mock values for testing
        self.env = LavaCrossing(difficulty=1)

    def test_observe(self):
        transformed_state = self.env.reset()

        # Ensure the transformed state has the correct shape
        self.assertEqual(transformed_state.shape, (7 * 7 + 1,))

    def test_invalid_difficulty(self):
        # Test that an invalid difficulty level raises a ValueError
        with self.assertRaises(ValueError):
            LavaCrossing(difficulty=4)
            LavaCrossing(difficulty=-1)

if __name__ == '__main__':
    unittest.main()
