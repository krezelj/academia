from typing import Any, Optional
from abc import abstractmethod

import numpy as np
import numpy.typing as npt

from . import GenericGymnasiumWrapper


class GenericMiniGridWrapper(GenericGymnasiumWrapper):
    """
    A wrapper for *MiniGrid* environments that makes them scalable.

    Args:
        difficulty: Difficulty level from 0 to 3, where 0 is the easiest
            and 3 is the hardest.
        difficulty_envid_map: A dict that maps numerical difficulty level to gymnasium environment ID.
        n_frames_stacked: How many most recent states should be stacked together to form a final state
            representation. Defaults to 1.
        append_step_count: Whether or not append the current step count to each state. Defaults to ``False``.
        random_state: Optional seed that controls the randomness of the environment. Defaults to ``None``.
        kwargs: Arguments passed down to ``gymnasium.make``.

    Raises:
        ValueError: If the specified difficulty level is invalid.

    Attributes:
        step_count (int): Current step count since the last reset.
        difficulty (int): Difficulty level. Higher values indicate more difficult environments.
        n_frames_stacked (int): How many most recent states should be stacked together to form a final state
            representation.
        append_step_count (bool): Whether or not append the current step count to each state.
    """

    def __init__(self, difficulty: int, difficulty_envid_map: dict, n_frames_stacked: int = 1,
                 append_step_count: bool = False, random_state: Optional[int] = None, **kwargs):
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
            append_step_count=append_step_count,
            random_state=random_state,
            **kwargs
        )

    @abstractmethod
    def _transform_state(self, raw_state: Any) -> npt.NDArray[np.float32]:
        """
        Transforms a state returned by the underlying environment to a numpy array of float32, which is a
        format commonly used throughout this package
        """
        pass
