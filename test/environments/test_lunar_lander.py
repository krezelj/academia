import unittest

from academia.environments import LunarLander


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Set up the environment with mock values for testing
        self.env = LunarLander(difficulty=1)

    def test_observe(self):
        state = self.env.reset()
        self.assertEqual(state.shape, (8,))

    def test_invalid_difficulty(self):
        # Test that an invalid difficulty level raises a ValueError
        with self.assertRaises(ValueError):
            LunarLander(difficulty=6)
            LunarLander(difficulty=-1)


if __name__ == '__main__':
    unittest.main()
