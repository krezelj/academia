import unittest
from unittest import mock
from academia.agents.base import Agent

@mock.patch.multiple(Agent, __abstractmethods__=frozenset())
class TestAgent(unittest.TestCase):
    
    
    def test_epsilon_decay(self):
        # arrange
        sut = Agent(n_actions=5, epsilon=1, epsilon_decay=0.5)
        # act
        sut.decay_epsilon()
        sut.decay_epsilon()
        # assert
        expected_epsilon = 0.25
        
        # allow for some small error since it's a floating point operation
        self.assertAlmostEqual(expected_epsilon, sut.epsilon, 5)

    def test_min_epsilon(self):
        # arrange
        sut = Agent(n_actions=5, epsilon=1, epsilon_decay=0.5, min_epsilon=0.5)
        # act
        sut.decay_epsilon()
        sut.decay_epsilon()
        # assert
        expected_epsilon = 0.5

        # allow for some small error since it's a floating point operation
        self.assertAlmostEqual(expected_epsilon, sut.epsilon, 5)
        # since it's minimum we want to assert that it's not less
        self.assertGreaterEqual(expected_epsilon, sut.epsilon)

    def test_epsilon_reset(self):
        # arrange
        sut = Agent(n_actions=5, epsilon=1, epsilon_decay=0.5)
        # act
        sut.decay_epsilon()
        sut.decay_epsilon()
        sut.reset_epsilon(0.8)
        # assert
        expected_epsilon = 0.8

        # allow for some small error since it's a floating point operation
        self.assertAlmostEqual(expected_epsilon, sut.epsilon, 5)


if __name__ == '__main__':
    unittest.main()