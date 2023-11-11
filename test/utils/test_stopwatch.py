import unittest

import numpy as np

from academia.utils import Stopwatch


class TestStopwatch(unittest.TestCase):

    def test_is_running(self):
        spt = Stopwatch(start=False)
        self.assertFalse(spt.is_running)
        spt.start()
        self.assertTrue(spt.is_running)
        spt.lap()
        self.assertTrue(spt.is_running)
        spt.stop()
        self.assertFalse(spt.is_running)
        spt.start()
        self.assertTrue(spt.is_running)
        spt.start()
        self.assertTrue(spt.is_running)

    def test_autostart(self):
        spt = Stopwatch(start=True)
        self.assertTrue(spt.is_running)

    def test_time_is_measured(self):
        spt = Stopwatch(start=True)
        wall_time, cpu_time = spt.stop()
        self.assertGreaterEqual(wall_time, 0)
        self.assertGreaterEqual(cpu_time, 0)

    def test_peek_methods(self):
        spt = Stopwatch(start=True)
        wall_time, cpu_time = spt.peek_time()
        wall_lap_time, cpu_lap_time = spt.peek_lap_time()
        self.assertGreaterEqual(wall_time, 0)
        self.assertGreaterEqual(cpu_time, 0)
        self.assertGreaterEqual(wall_lap_time, 0)
        self.assertGreaterEqual(cpu_lap_time, 0)

    def test_restart_resets_lap_times(self):
        # arrange
        spt = Stopwatch(start=False)
        # act
        spt.start()
        spt.lap()
        spt.start()
        # assert
        self.assertEqual(len(spt.wall_lap_times), 0)
        self.assertEqual(len(spt.cpu_lap_times), 0)

    def test_errors_when_no_start(self):
        spt = Stopwatch(start=False)
        with self.assertRaises(RuntimeError):
            spt.peek_time()
        with self.assertRaises(RuntimeError):
            spt.peek_lap_time()
        with self.assertRaises(RuntimeError):
            spt.lap()
        with self.assertRaises(RuntimeError):
            spt.stop()

    def test_lap_adds_lap_time(self):
        spt = Stopwatch(start=True)
        spt.lap()
        self.assertEqual(len(spt.wall_lap_times), 1)
        self.assertEqual(len(spt.cpu_lap_times), 1)

    def test_stop_with_lap_adds_lap_time(self):
        spt = Stopwatch(start=True)
        spt.stop(lap=True)
        self.assertEqual(len(spt.wall_lap_times), 1)
        self.assertEqual(len(spt.cpu_lap_times), 1)

    def test_stop_without_lap_doesnt_add_lap_time(self):
        spt = Stopwatch(start=True)
        spt.stop(lap=False)
        self.assertEqual(len(spt.wall_lap_times), 0)
        self.assertEqual(len(spt.cpu_lap_times), 0)

    def test_lap_times_sum_to_total_time(self):
        spt = Stopwatch(start=True)
        total_wall_time, total_cpu_time = spt.stop(lap=True)
        self.assertEqual(np.sum(spt.wall_lap_times), total_wall_time)
        self.assertEqual(np.sum(spt.cpu_lap_times), total_cpu_time)


if __name__ == '__main__':
    unittest.main()
