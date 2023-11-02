from typing import Any
from abc import abstractmethod

import numpy as np
import numpy.typing as npt

from . import GenericGymnasiumWrapper


class GenericMiniGridWrapper(GenericGymnasiumWrapper):
    """
    A wrapper for MiniGrid environments that makes them scalable.
    """

    def __init__(self, difficulty: int, difficulty_envid_map: dict, n_frames_stacked: int = 1, **kwargs):
        """
        :param difficulty:  Difficulty level from 0 to 3, where 0 is the easiest
                            and 3 is the hardest
        """
        
        self._difficulty_envid_map = difficulty_envid_map
        try:
            env_id = self._difficulty_envid_map[difficulty]
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 3")
            raise ValueError(msg)
        super().__init__(
            difficulty=difficulty,
            environment_id=env_id,
            n_frames_stacked=n_frames_stacked,
            **kwargs
        )

    @abstractmethod
    def _transform_state(self, raw_state: Any) -> npt.NDArray[np.float32]:
        """
        Transforms a state returned by the underlying environment to a numpy array of float32, which is a
        format commonly used throughout this package
        """
        pass
