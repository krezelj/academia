from typing import Union, Any, Optional

import gymnasium
import numpy as np
import numpy.typing as npt

from .base import ScalableEnvironment

class DoorKey(ScalableEnvironment):

    N_ACTIONS = 5

    __difficulty_envid_map = {
        0: 'MiniGrid-DoorKey-5x5-v0',
        1: 'MiniGrid-DoorKey-6x6-v0',
        2: 'MiniGrid-DoorKey-8x8-v0',
        3: 'MiniGrid-DoorKey-16x16-v0'
    }
    """A dictionary that maps difficulty levels to environment ids"""

    def __init__(self, difficulty: int, render_mode: Optional[str] = None):
        """
        :param difficulty:  Difficulty level from 0 to 3, where 0 is the easiest
                            and 3 is the hardest
        :param render_mode: render_mode value passed to gymnasium.make
        """
        super().__init__(difficulty)
        try:
            env_id = DoorKey.__difficulty_envid_map[difficulty]
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 3")
            raise ValueError(msg)
        self.__base_env = gymnasium.make(env_id, render_mode=render_mode)
        self.__state_raw = None  # will be set inside self.reset()
        self.reset()
        self.STATE_SIZE = len(self.__state)


    def step(self, action: int) -> tuple[Any, float, bool]:
        new_state, reward, terminated, truncated, _ = self.__base_env.step(action)
        self.__state_raw = new_state
        is_episode_end = terminated or truncated
        return self.observe(), float(reward), is_episode_end

    def observe(self) -> Any:
        return self.__state

    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        return np.array([1 for _ in range(self.N_ACTIONS)])

    def reset(self) -> Any:
        self.__door_status = 2
        self.__state_raw = self.__base_env.reset()[0]
        return self.observe()
    
    def render(self):
        return self.__base_env.render()
    
    @property
    def __state(self) -> tuple[int, ...]:
        cells_obj_types = self.__state_raw['image'][:,:,0]
        cells_flattened = cells_obj_types.flatten()
        direction = self.__state_raw['direction']
        door_array = self.__state_raw['image'][:,:,2].flatten()

        if self.__door_status == 2 and 1 in door_array:
            self.__door_status = 1
        elif self.__door_status == 1 and 0 in door_array:
            self.__door_status = 0

        return *cells_flattened, direction, self.__door_status
    





