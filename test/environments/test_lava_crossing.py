import unittest

from academia.environments import LavaCrossing


class TestLavaCrossing(unittest.TestCase):
    def setUp(self):
        # Set up the environment with mock values for testing
        self.sut = LavaCrossing(difficulty=1)

    def test_observe(self):
        transformed_state = self.sut.reset()

        # Ensure the transformed state has the correct shape
        self.assertEqual((7 * 7 + 1,), transformed_state.shape)

    def test_invalid_difficulty(self):
        # Test that an invalid difficulty level raises a ValueError
        with self.assertRaises(ValueError):
            LavaCrossing(difficulty=4)
        with self.assertRaises(ValueError):
            LavaCrossing(difficulty=-1)

if __name__ == '__main__':
    unittest.main()
