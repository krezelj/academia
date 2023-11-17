from typing import Any, Callable
import unittest
from unittest import mock
from collections import deque

import numpy as np

from academia.tools import AgentDebugger
from academia.environments import BridgeBuilding
from academia.agents import QLAgent


class MockEnvironment:
    """
    Mock environment that always terminates if action ``BridgeBuilding.N_ACTIONS`` 
    was taken (to avoid agent accidentally taking this action) and never terminates otherwise.
    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.state = ""

    def step(self, action):
        return self.state, 0, action == BridgeBuilding.N_ACTIONS
    
    def get_legal_mask(self):
        return np.ones(shape=BridgeBuilding.N_ACTIONS)


class InjectedAsserter:

    def __init__(self, 
                 tester: unittest.TestCase, 
                 debugger: AgentDebugger,
                 assertions: list[Callable],
                 input_sequence: list[str]):
        self.tester = tester
        self.debugger = debugger
        self.assertions = deque(assertions)
        self.input_sequence = deque(input_sequence)

    def __call__(self, *args, **kwargs):
        self.assertions.popleft()(self.tester, self.debugger)
        if len(self.input_sequence) > 0:
            return self.input_sequence.popleft(), False
        return '\x1b', True


class TestAgentDebugger(unittest.TestCase):

    @staticmethod
    def assert_nothing(teset, agent):
        """
        dummy asserter that always passes
        """
        pass

    @staticmethod
    def assert_attribute_equal_factory(attribute_name: str, expected: Any):
        def dynamic_assert(tester: unittest.TestCase, debugger: AgentDebugger):
            tester.assertEqual(expected, getattr(debugger, attribute_name),
                               msg=f"wrong value of {attribute_name}")
        return dynamic_assert

    def setUp(self) -> None:
        self.agent = QLAgent(n_actions=BridgeBuilding.N_ACTIONS)
        self.env = BridgeBuilding()
    
    def test_pausing(self):
        # arrange
        sut = AgentDebugger(self.agent, self.env)
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("paused", False),
            self.assert_attribute_equal_factory("paused", True),
            self.assert_attribute_equal_factory("paused", False),
        ], input_sequence=['p', 'p', '\x1b'])

        # assert
        self.assertFalse(sut.paused)
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut.run()

    def test_start_paused(self):
        # arrange
        sut = AgentDebugger(self.agent, self.env, start_paused=True)
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("paused", True),
        ], input_sequence=['\x1b'])

        # assert
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut.run()

    def test_toggle_greedy(self):
        # arrange
        sut = AgentDebugger(self.agent, self.env)
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("greedy", False),
            self.assert_attribute_equal_factory("greedy", True),
            self.assert_attribute_equal_factory("greedy", False),
        ], input_sequence=['g', 'g', '\x1b'])

        # assert
        self.assertFalse(sut.paused)
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut.run()

    def test_start_greedy(self):
        # arrange
        sut = AgentDebugger(self.agent, self.env, start_greedy=True)
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("greedy", True),
        ], input_sequence=['\x1b'])

        # assert
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut.run()

    def test_step(self):
        sut = AgentDebugger(self.agent, self.env, start_paused=True)
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("steps", 1),
            self.assert_attribute_equal_factory("steps", 2),
        ], input_sequence=[' ', '\x1b'])

        # assert
        # prevent episode from terminating
        with mock.patch.object(BridgeBuilding, 'step', return_value=("", 0.0, False)):
            with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
                sut.run()

    def test_terminate(self):
        sut = AgentDebugger(self.agent, self.env, start_paused=True)
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("episodes", 1),
            self.assert_attribute_equal_factory("episodes", 2),
        ], input_sequence=['t', '\x1b'])

        # assert
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut.run()

    def test_quit(self):
        sut = AgentDebugger(self.agent, self.env, start_paused=True)
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("running", True),
        ], input_sequence=['\x1b'])

        # assert
        self.assertFalse(sut.running)
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut.run()
        self.assertFalse(sut.running)

    def test_autorun(self):
        injected_asserter = InjectedAsserter(self, None, [
            self.assert_nothing
        ], input_sequence=['\x1b'])

        # assert
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut = AgentDebugger(self.agent, self.env, start_paused=True, run=True)
        self.assertGreaterEqual(sut.steps, 0)

    def test_key_action_map(self):
        env = MockEnvironment()
        sut = AgentDebugger(self.agent, env, start_paused=True, 
            key_action_map={'a': BridgeBuilding.N_ACTIONS, 'b': 1}
        )
        injected_asserter = InjectedAsserter(self, sut, [
            self.assert_attribute_equal_factory("episodes", 1),  #a terminate
            self.assert_attribute_equal_factory("episodes", 2),  #b step     
            self.assert_attribute_equal_factory("episodes", 2),  #b step
            self.assert_attribute_equal_factory("steps", 3),     #c (maps to None) step
            self.assert_attribute_equal_factory("steps", 4),     #a terminate
            self.assert_attribute_equal_factory("episodes", 3),  #0 terminate
            self.assert_attribute_equal_factory("episodes", 4),  #1 step
            self.assert_attribute_equal_factory("episodes", 4),  #c (maps to None) step
            self.assert_attribute_equal_factory("episodes", 4),  #quit
        ], input_sequence=['a', 'b', 'b', 'c', 'a', f'{BridgeBuilding.N_ACTIONS}', '1', 'c', '\x1b'])

        # assert
        with mock.patch('academia.tools.agent_debugger.timedKey', new=injected_asserter):
            sut.run()

    def test_invalid_key_action_map(self):
        # check that each key fails separately
        for reserved_key in AgentDebugger.reserved_keys:
            with self.assertRaises(ValueError):
                AgentDebugger(self.agent, self.env, start_paused=True, key_action_map={
                    reserved_key: 0
                })

if __name__ == '__main__':
    unittest.main()
    