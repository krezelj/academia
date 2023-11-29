from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class ScalableEnvironment(ABC):
    """
    Base class for all environments used in this package. Scalability ensures environments can be
    used for `Curriculum Learning`.

    Args:
        difficulty: Difficulty level. Higher values indicate more difficult environments.
        n_frames_stacked: How many most recent states should be stacked together to form a final state
            representation. Defaults to 1.

    Attributes:
        difficulty (int): Difficulty level. Higher values indicate more difficult environments.
        n_frames_stacked (int): How many most recent states should be stacked together to form a final state
            representation.
    """

    N_ACTIONS: int
    """Number of available actions."""
    STATE_SHAPE: tuple[int, ...]
    """Shape of the state representation. Can vary for each instance"""

    @abstractmethod
    def __init__(self, difficulty: int, n_frames_stacked: int = 1, **kwargs):
        self.difficulty = difficulty
        self.n_frames_stacked = n_frames_stacked

    @abstractmethod
    def step(self, action: int) -> tuple[Any, float, bool]:
        """
        Takes the given action in the environment

        Args:
            action: An action to take.

        Returns:
            A tuple consisting of a new state, reward and a flag indicating whether the state is terminal.
        """
        pass

    @abstractmethod
    def reset(self) -> Any:
        """
        Resets the environment.

        Returns:
             A starting state.
        """
        pass

    @abstractmethod
    def observe(self) -> Any:
        """
        Returns:
            A current state.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Renders the environment.
        """
        pass

    @abstractmethod
    def get_legal_mask(self) -> npt.NDArray[np.int32]:
        """
        Returns:
            A binary mask with 0s in place for illegal actions (actions that
            have no effect) and 1s for legal actions.
        """
        pass
