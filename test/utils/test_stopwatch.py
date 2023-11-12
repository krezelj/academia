import unittest

import numpy as np

from academia.utils import Stopwatch


class TestStopwatch(unittest.TestCase):

    def test_is_running(self):
        sut = Stopwatch(start=False)
        self.assertFalse(sut.is_running, "Stopwatch should not be running before it starts")
        sut.start()
        self.assertTrue(sut.is_running, "Stopwatch should be running after it has started")
        sut.lap()
        self.assertTrue(sut.is_running, "Stopwatch should be running after lap is done")
        sut.stop()
        self.assertFalse(sut.is_running, "Stopwatch should not be running after it has stopped")
        sut.start()
        self.assertTrue(sut.is_running, "Stopwatch should be running after it has been restarted")

    def test_autostart(self):
        sut = Stopwatch(start=True)
        self.assertTrue(sut.is_running)

    def test_time_is_measured(self):
        sut = Stopwatch(start=True)
        wall_time, cpu_time = sut.stop()
        self.assertGreaterEqual(wall_time, 0)
        self.assertGreaterEqual(cpu_time, 0)

    def test_peek_methods(self):
        sut = Stopwatch(start=True)
        wall_time, cpu_time = sut.peek_time()
        wall_lap_time, cpu_lap_time = sut.peek_lap_time()
        self.assertGreaterEqual(wall_time, 0)
        self.assertGreaterEqual(cpu_time, 0)
        self.assertGreaterEqual(wall_lap_time, 0)
        self.assertGreaterEqual(cpu_lap_time, 0)

    def test_restart_resets_lap_times(self):
        # arrange
        sut = Stopwatch(start=False)
        # act
        sut.start()
        sut.lap()
        sut.start()
        # assert
        self.assertEqual(len(sut.wall_lap_times), 0)
        self.assertEqual(len(sut.cpu_lap_times), 0)

    def test_errors_when_no_start(self):
        sut = Stopwatch(start=False)
        with self.assertRaises(RuntimeError):
            sut.peek_time()
        with self.assertRaises(RuntimeError):
            sut.peek_lap_time()
        with self.assertRaises(RuntimeError):
            sut.lap()
        with self.assertRaises(RuntimeError):
            sut.stop()

    def test_lap_adds_lap_time(self):
        sut = Stopwatch(start=True)
        sut.lap()
        self.assertEqual(1, len(sut.wall_lap_times))
        self.assertEqual(1, len(sut.cpu_lap_times))

    def test_stop_with_lap_adds_lap_time(self):
        sut = Stopwatch(start=True)
        sut.stop(lap=True)
        self.assertEqual(1, len(sut.wall_lap_times))
        self.assertEqual(1, len(sut.cpu_lap_times))

    def test_stop_without_lap_doesnt_add_lap_time(self):
        sut = Stopwatch(start=True)
        sut.stop(lap=False)
        self.assertEqual(0, len(sut.wall_lap_times))
        self.assertEqual(0, len(sut.cpu_lap_times))

    def test_lap_times_sum_to_total_time(self):
        sut = Stopwatch(start=True)
        total_wall_time, total_cpu_time = sut.stop(lap=True)
        self.assertEqual(total_wall_time, np.sum(sut.wall_lap_times))
        self.assertEqual(total_cpu_time, np.sum(sut.cpu_lap_times))


if __name__ == '__main__':
    unittest.main()
