import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from academia.agents.base import TabularAgent

class TestTabularAgent(unittest.TestCase):

    def __assert_agents_equal(self, expected: TabularAgent, returned: TabularAgent):
        self.assertEqual(len(returned.q_table), len(expected.q_table))
        self.assertIn(self.mock_state, returned.q_table)
        self.assertTrue(
            np.all(returned.q_table[self.mock_state] == expected.q_table[self.mock_state]))
        
        ignored_attributes = ['q_table', '_rng']
        for attribute_name in expected.__dict__.keys():
            if attribute_name not in ignored_attributes:
                self.assertEqual(
                    getattr(expected, attribute_name), 
                    getattr(returned, attribute_name),
                    msg=f"Attribute '{attribute_name}' not equal")

    def setUp(self) -> None:
        # arrange
        with mock.patch.object(TabularAgent, "__abstractmethods__", frozenset()):
            # instantiate an object with non default values to test saving loading
            self.tabular_agent = TabularAgent(
                n_actions=3,
                alpha=0.05,
                gamma = 0.8,
                epsilon=0.9,
                epsilon_decay=0.9,
                min_epsilon=0.03,
                random_state=0
            )
            self.mock_state = "mock_state"
            self.tabular_agent.q_table[self.mock_state] = np.array([1, -1, 2])

    def test_get_greedy_action(self):
        # act
        action = self.tabular_agent.get_action(self.mock_state, greedy=True)
        #assert
        expected_action = 2
        self.assertEqual(action, expected_action)

    def test_legal_mask(self):
        # act
        action = self.tabular_agent.get_action(self.mock_state, legal_mask=np.array([0,1,0]), greedy=True)
        # assert
        expected_action = 1
        self.assertEqual(action, expected_action)

    def test_file_path_suffixed(self):
        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.json', delete=False)
        returned_path = self.tabular_agent.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(returned_path, tmpfile.name)

    def test_file_path_unsuffixed(self):
        # act 
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        returned_path = self.tabular_agent.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(returned_path, tmpfile.name + '.agent.json')

    def test_saving_loading(self):
        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.json', delete=False)
        self.tabular_agent.save(tmpfile.name)
        with mock.patch.object(TabularAgent, "__abstractmethods__", frozenset()):
            loaded_agent = TabularAgent.load(tmpfile.name)
        tmpfile.close()

        # assert
        self.__assert_agents_equal(self.tabular_agent, loaded_agent)


if __name__ == '__main__':
    unittest.main()