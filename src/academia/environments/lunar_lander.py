from typing import Any, Union, Optional

import numpy as np
import numpy.typing as npt
import gymnasium

from.base import ScalableEnvironment


class LunarLander(ScalableEnvironment):
    """
    A class representing the Lunar Lander environment, a variant of the classic Lunar Lander game.
    
    The goal is to land a spacecraft on the moon's surface by controlling its thrusters.
    The environment has a state size of 8 and 4 possible actions.
    The difficulty ranges from 0 to 5, with higher values indicating more challenging conditions.
    The environment can be rendered in different modes.

    Possible actions:

    | Num   | Action             |
    |-------|--------------------|
    | 0     | Do nothing         |
    | 1     | Fire left engine   |
    | 2     | Fire down engine   |
    | 3     | Fire right engine  |

    Difficulty levels:

    | Difficulty | Description                                   |
    |------------|-----------------------------------------------|
    | 0          | No wind, no turbulence                        |
    | 1          | Weak wind, no turbulence                      |
    | 2          | Moderate wind, weak turbulence                |
    | 3          | Medium strong wind, moderate turbulence       |
    | 4          | Strong wind, medium strong turbulence         |
    | 5          | Very strong wind, strong turbulence           |
    
    """

    N_ACTIONS: int = 4
    """The number of possible actions (4)."""
    STATE_SIZE: int = 8
    """The size of the state space (8)."""

    __difficulty_params_map = {
        0: {'enable_wind': False, 'wind_power': 0.0, 'turbulence_power': 0.0},
        1: {'enable_wind': True, 'wind_power': 5.0, 'turbulence_power': 0.0},
        2: {'enable_wind': True, 'wind_power': 10.0, 'turbulence_power': 0.5},
        3: {'enable_wind': True, 'wind_power': 15.0, 'turbulence_power': 1.0},
        4: {'enable_wind': True, 'wind_power': 20.0, 'turbulence_power': 1.5},
        5: {'enable_wind': True, 'wind_power': 25.0, 'turbulence_power': 2.0}, 
    }

    def __init__(self, difficulty: int, render_mode: Optional[str] = None, **kwargs):
        """
        Initializes a new instance of the LunarLander class with the specified difficulty and render mode.
        
        Args:
            difficulty: The difficulty level of the environment (0 to 5).
            render_mode: The render mode ('human', 'rgb_array', or None).

        Raises:
            ValueError: If the specified difficulty level is invalid.
        """
        super().__init__(difficulty, **kwargs)
        try:
            self.difficulty_params = LunarLander.__difficulty_params_map[difficulty]
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 5")
            raise ValueError(msg)
        self._base_env = gymnasium.make('LunarLander-v2', render_mode=render_mode, **self.difficulty_params, **kwargs)
        self._state = None
        self.reset()
        
    def step(self, action: int) -> tuple[Any, float, bool]:
        """
        Advances the environment by one step given the specified action.
        
        Args:
            action: The action to take (0 to 3).

        Returns:
            A tuple containing the new state, reward, and a flag indicating episode termination.
        """
        new_state, reward, terminated, truncated, _ = self._base_env.step(action)
        self._state = new_state
        is_episode_end = terminated or truncated
        return self.observe(), float(reward), is_episode_end
    
    def observe(self) -> Any:
        """
        Returns the current state of the environment.

        Returns:
            The current state of the environment.
        """
        return self._state
    
    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        """
        Returns:
            A binary mask indicating legal actions.
        """
        return np.array([1 for _ in range(self.N_ACTIONS)])
    
    def reset(self) -> Any:
        """
        Resets the environment to its initial state.

        Returns:
            The new state after resetting the environment.
        """
        self._state = self._base_env.reset()[0]
        return self.observe()
    
    def render(self):
        """
        Renders the environment in the current render mode.
        """
        self._base_env.render()
    
