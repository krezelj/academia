import unittest

from academia.environments import MsPacman


class TestMsPacman(unittest.TestCase):
    def test_invalid_difficulty(self):
        # Test that an invalid difficulty level raises a ValueError
        with self.assertRaises(ValueError):
            MsPacman(difficulty=4)
            MsPacman(difficulty=-1)


if __name__ == '__main__':
    unittest.main()
