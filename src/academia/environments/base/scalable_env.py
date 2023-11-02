from abc import ABC, abstractmethod
from typing import Any, Union

import numpy.typing as npt


class ScalableEnvironment(ABC):
    """Base class for all scalable environments used in this package"""

    N_ACTIONS: int
    """Number of available actions"""
    STATE_SIZE: int
    """A constant denoting the dimension of the state representation"""

    @abstractmethod
    def __init__(self, difficulty: int, n_frames_stacked: int = 1, **kwargs):
        """
        :param difficulty: Difficulty level. Higher values indicate more
                           difficult environments
        """
        self.difficulty = difficulty
        self.n_frames_stacked = n_frames_stacked

    @abstractmethod
    def step(self, action: int) -> tuple[Any, float, bool]:
        """Takes the given action in the environment

        :param action: an action to take
        :return: A tuple consisting of a new state, reward and a flag indicating
        whether the state is terminal"""
        pass

    @abstractmethod
    def reset(self) -> Any:
        """Resets the environment

        :return: A starting state"""
        pass

    @abstractmethod
    def observe(self) -> Any:
        """:return: A current state"""
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        pass
