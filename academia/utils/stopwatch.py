from typing import Callable, Optional
import time

import numpy as np


class Stopwatch:
    """A utility class for measuring and storing consecutive CPU/wall times.
    All times are stored in seconds"""

    __slots__ = ['__wall_stopwatch', '__cpu_stopwatch']

    def __init__(self, start=True):
        """
        :param start: whether or not to start a stopwatch immidiately after initialising it
        """
        self.__wall_stopwatch = _GenericStopwatch(timestamp_func=time.perf_counter, start=start)
        self.__cpu_stopwatch = _GenericStopwatch(timestamp_func=time.process_time, start=start)

    def start(self) -> None:
        self.__wall_stopwatch.start()
        self.__cpu_stopwatch.start()

    def lap(self) -> tuple[float, float]:
        """
        :return: Wall and CPU lap times
        """
        return self.__wall_stopwatch.lap(), self.__cpu_stopwatch.lap()

    def stop(self, lap=False) -> tuple[float, float]:
        """
        :param lap: whether or not end and save the final lap
        :return: Wall and CPU total times
        """
        wall_total = self.__wall_stopwatch.stop(lap=lap)
        cpu_total = self.__cpu_stopwatch.stop(lap=lap)
        return wall_total, cpu_total

    @property
    def wall_lap_times(self) -> list[float]:
        return self.__wall_stopwatch.lap_times

    @property
    def cpu_lap_times(self) -> list[float]:
        return self.__cpu_stopwatch.lap_times


class _GenericStopwatch:

    __slots__ = ['lap_times', '__lap_start', '__timestamp_func']

    def __init__(self, timestamp_func: Callable[[], float], start=True):
        """
        :param start: whether or not to start a stopwatch immidiately after initialising it
        """
        self.__timestamp_func = timestamp_func
        self.lap_times: list[float] = []
        self.__lap_start: Optional[float] = None
        if start:
            self.start()

    def start(self) -> None:
        self.__lap_start = self.__timestamp_func()

    def lap(self) -> float:
        """
        :return: Lap time
        """
        curr_time = self.__timestamp_func()
        lap_time = curr_time - self.__lap_start
        self.lap_times.append(lap_time)
        self.__lap_start = curr_time
        return lap_time

    def stop(self, lap=False) -> float:
        """
        :param lap: whether or not end and save the final lap
        :return: Total elapsed time
        """
        if lap:
            self.lap()
            total_time = np.sum(self.lap_times)
        else:
            curr_time = self.__timestamp_func()
            total_time = np.sum(self.lap_times) + (curr_time - self.__lap_start)
        return total_time
