import threading

import unittest
from unittest import mock
from academia.tools import AgentDebugger

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.
    Source: https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
    """

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

class async_debugger:
    """
    Asynchronous agent debugger
    """

    def __init__(self, debugger: AgentDebugger):
        self.thread = StoppableThread(target=debugger.run, args=(5,))

    def __enter__(self):
        self.thread.start()

    def __exit__(self, *exc):
        self.thread.stop()
        self.thread.join()

class TestAgentDebugger(unittest.TestCase):
    
    def test_paused(self):
        pass

    def test_start_paused(self):
        pass

    def test_greedy(self):
        pass

    def test_start_greedy(self):
        pass

    def test_key_action_map(self):
        pass

    def test_invalid_key_action_map(self):
        pass

    def test_step(self):
        pass

    def test_terminate(self):
        pass

    def test_quit(self):
        pass

    def test_thought_handler(self):
        pass

if __name__ == '__main__':
    unittest.main()