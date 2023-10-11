from abc import ABC, abstractmethod

from .base_env import BaseEnvironment


class ScalableEnvironment(BaseEnvironment, ABC):

    @abstractmethod
    def __init__(self, difficulty: int):
        """
        :param difficulty: Difficulty level. Higher values indicate more
                           difficult environments
        """
        self.difficulty = difficulty
