from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .base import GenericGymnasiumWrapper


class LunarLander(GenericGymnasiumWrapper):
    """
    This class is a wrapper for *Gymnasium*'s Lunar Lander environment, which itself is
    a variant of the classic Lunar Lander game.

    The goal is to land a spacecraft on the moon's surface by controlling its thrusters.
    The environment has a state size of 8 and 4 possible actions.
    The difficulty ranges from 0 to 5, with higher values indicating more challenging conditions.
    The environment can be rendered in different modes.

    Possible actions:

    +-------+--------------------+
    | Num   | Action             |
    +=======+====================+
    | 0     | Do nothing         |
    +-------+--------------------+
    | 1     | Fire left engine   |
    +-------+--------------------+
    | 2     | Fire down engine   |
    +-------+--------------------+
    | 3     | Fire right engine  |
    +-------+--------------------+

    Difficulty levels:

    +------------+-----------------------------------------------+
    | Difficulty | Description                                   |
    +============+===============================================+
    | 0          | No wind, no turbulence                        |
    +------------+-----------------------------------------------+
    | 1          | Weak wind, no turbulence                      |
    +------------+-----------------------------------------------+
    | 2          | Moderate wind, weak turbulence                |
    +------------+-----------------------------------------------+
    | 3          | Medium strong wind, moderate turbulence       |
    +------------+-----------------------------------------------+
    | 4          | Strong wind, medium strong turbulence         |
    +------------+-----------------------------------------------+
    | 5          | Very strong wind, strong turbulence           |
    +------------+-----------------------------------------------+

    See Also:
        *Gymnasium*'s Lunar Lander environment: https://gymnasium.farama.org/environments/box2d/lunar_lander/

    Args:
        difficulty: The difficulty level of the environment (0 to 5).
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

    N_ACTIONS: int = 4

    __difficulty_params_map = {
        0: {'enable_wind': False, 'wind_power': 0.0, 'turbulence_power': 0.0},
        1: {'enable_wind': True, 'wind_power': 5.0, 'turbulence_power': 0.0},
        2: {'enable_wind': True, 'wind_power': 10.0, 'turbulence_power': 0.5},
        3: {'enable_wind': True, 'wind_power': 15.0, 'turbulence_power': 1.0},
        4: {'enable_wind': True, 'wind_power': 20.0, 'turbulence_power': 1.5},
        5: {'enable_wind': True, 'wind_power': 25.0, 'turbulence_power': 2.0}, 
    }

    def __init__(self, difficulty: int, n_frames_stacked: int = 1, append_step_count: bool = False,
                 random_state: Optional[int] = None, **kwargs):
        try:
            difficulty_params = LunarLander.__difficulty_params_map[difficulty]
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 5")
            raise ValueError(msg)
        super().__init__(
            difficulty=difficulty,
            environment_id='LunarLander-v2',
            n_frames_stacked=n_frames_stacked,
            append_step_count=append_step_count,
            random_state=random_state,
            **difficulty_params,
            **kwargs,
        )

    def _transform_state(self, raw_state: Any) -> npt.NDArray[np.float32]:
        # raw state returned by lunar lander is already a numpy array
        raw_state: np.ndarray
        return raw_state.astype(np.float32)
