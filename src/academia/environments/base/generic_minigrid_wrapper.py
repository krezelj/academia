from typing import Union, Any, Optional
from abc import abstractmethod

import gymnasium
import numpy as np
import numpy.typing as npt

from .scalable_env import ScalableEnvironment

class GenericMiniGridWrapper(ScalableEnvironment):
    """
    A wrapper for MiniGrid environments that makes them scalable.
    """

    def __init__(self, difficulty: int, difficulty_envid_map: dict, render_mode: Optional[str] = None):
        """
        :param difficulty:  Difficulty level from 0 to 3, where 0 is the easiest
                            and 3 is the hardest
        :param render_mode: render_mode value passed to gymnasium.make
        """
        super().__init__(difficulty)
        self.__difficulty_envid_map = difficulty_envid_map
        try:
            env_id = self.__difficulty_envid_map[difficulty]
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 3")
            raise ValueError(msg)
        self.__base_env = gymnasium.make(env_id, render_mode=render_mode)
        self.__state_raw = None  # will be set inside self.reset()
        self.reset()
        self.STATE_SIZE = len(self._state)

    def step(self, action: int) -> tuple[Any, float, bool]:
        new_state, reward, terminated, truncated, _ = self.__base_env.step(action)
        self.__state_raw = new_state
        is_episode_end = terminated or truncated
        return self.observe(), float(reward), is_episode_end

    def observe(self) -> Any:
        return self._state
    

    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        pass

    def reset(self) -> Any:
        self.__state_raw = self.__base_env.reset()[0]
        return self.observe()

    def render(self):
        self.__base_env.render()

    @property
    @abstractmethod
    def _state(self) -> tuple[int, ...]:
        raise NotImplementedError