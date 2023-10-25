from typing import Any, Union

import numpy as np
import numpy.typing as npt
import gymnasium

from.base import ScalableEnvironment


class LunarLander(ScalableEnvironment):

    N_ACTIONS = 4
    STATE_SIZE = 8

    __difficulty_params_map = {
        0: {'enable_wind': False, 'wind_power': 0.0, 'turbulence_power': 0.0},
        1: {'enable_wind': True, 'wind_power': 5.0, 'turbulence_power': 0.0},
        2: {'enable_wind': True, 'wind_power': 10.0, 'turbulence_power': 0.5},
        3: {'enable_wind': True, 'wind_power': 15.0, 'turbulence_power': 1.0},
        4: {'enable_wind': True, 'wind_power': 20.0, 'turbulence_power': 1.5},
        5: {'enable_wind': True, 'wind_power': 25.0, 'turbulence_power': 2.0}, 
    }

    def __init__(self, difficulty: int):
        super().__init__(difficulty)
        try:
            self.params = LunarLander.__difficulty_params_map.get(difficulty, {})
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 5")
            raise ValueError(msg)
        self._base_env = gymnasium.make('LunarLander-v2', **self.params)
        self._state = None
        self.reset()
        
    def step(self, action: int) -> tuple[Any, float, bool]:
        new_state, reward, terminated, truncated, _ = self._base_env.step(action)
        self._state = new_state
        is_episode_end = terminated or truncated
        return self.observe(), float(reward), is_episode_end
    
    def observe(self) -> Any:
        return self._state
    
    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        return np.array([1 for _ in range(self.N_ACTIONS)])
    
    def reset(self) -> Any:
        self._state = self._base_env.reset()[0]
        return self.observe()
    
    def render(self):
        self._base_env.render()
    
