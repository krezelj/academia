from typing import Union, Any, Optional
from abc import abstractmethod
from collections import deque

import gymnasium
import numpy as np
import numpy.typing as npt

from .scalable_env import ScalableEnvironment


class GenericMiniGridWrapper(ScalableEnvironment):
    """
    A wrapper for MiniGrid environments that makes them scalable.
    """

    def __init__(self, difficulty: int, difficulty_envid_map: dict, n_frames_stacked: int = 1, **kwargs):
        """
        :param difficulty:  Difficulty level from 0 to 3, where 0 is the easiest
                            and 3 is the hardest
        """
        
        super().__init__(difficulty, n_frames_stacked, **kwargs)
        self._difficulty_envid_map = difficulty_envid_map
        try:
            env_id = self._difficulty_envid_map[difficulty]
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 3")
            raise ValueError(msg)
        self._base_env = gymnasium.make(env_id, **kwargs)
        self._state_raw = None  # will be set inside self.reset()
        self._past_n_states = deque()  # will be properly set inside self.reset()
        self.reset()
        self.STATE_SIZE = len(self._state)

    def step(self, action: int) -> tuple[Any, float, bool]:
        new_state, reward, terminated, truncated, _ = self._base_env.step(action)
        self._state_raw = new_state

        # frame stacking
        self._past_n_states.append(self._state)
        if len(self._past_n_states) > self.n_frames_stacked:
            self._past_n_states.popleft()

        is_episode_end = terminated or truncated
        return self.observe(), float(reward), is_episode_end

    def observe(self) -> Any:
        stacked_state = np.concatenate(list(self._past_n_states))
        return stacked_state
    
    @abstractmethod
    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        pass

    def reset(self) -> Any:
        self._state_raw = self._base_env.reset()[0]
        self._past_n_states = deque([self._state])
        return self._state

    def render(self):
        self._base_env.render()

    @property
    @abstractmethod
    def _state(self) -> npt.NDArray[int]:
        pass
