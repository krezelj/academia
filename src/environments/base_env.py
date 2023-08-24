from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):

    N_ACTIONS: int

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool]:
        """Takes the given action in the environment

        :param action: an action to take
        :return: A tuple consisting of a new state, reward and a flag indicating
        whether the state is terminal"""
        pass

    @abstractmethod
    def reset(self):
        """Resets the environment"""
        pass

    @abstractmethod
    def observe(self) -> Any:
        """:return: A current state"""
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def get_legal_mask(self):
        pass
