# TODO pip install ale-py, shimmy



from typing import Any, Optional, Union


import gymnasium
import numpy as np
import numpy.typing as npt


from .base import ScalableEnvironment


class MsPacman(ScalableEnvironment):
    
    N_ACTIONS = 9

    def __init__(self, 
                 difficulty: int, 
                 render_mode: Optional[str] = None,
                 flatten_state: bool = False,
                 n_frames_stacked: int = 1,
                 skip_game_start: bool = True,
                 **kwargs) -> None:
        super().__init__(difficulty, **kwargs)

        if 'mode' in kwargs:
            del kwargs['mode']
        
        # See atariage.com manual for ms pacman for details
        kwargs['mode'] = 0 if difficulty == 3 else 1 + difficulty

        self._base_env = gymnasium.make("ALE/MsPacman-v5", render_mode=render_mode, **kwargs)
        self._state_raw = None  # will be set inside self.reset()
        self.flatten_state = flatten_state
        self.skip_game_start = skip_game_start

        self.reset()
        # self.STATE_SIZE = len(self._state)

    
    def step(self, action: int) -> tuple[Any, float, bool]:
        new_state, reward, terminated, truncated, _ = self._base_env.step(action)
        self._state_raw = new_state
        is_episode_end = terminated or truncated
        return self.observe(), float(reward), is_episode_end

    def observe(self) -> Any:
        return self._state
    
    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        return np.array([1 for _ in range(self.N_ACTIONS)])

    def reset(self) -> Any:
        self._state_raw = self._base_env.reset()[0]
        if self.skip_game_start:
            self.__skip_game_start()
        return self.observe()

    def render(self):
        self._base_env.render()

    @property
    def _state(self) -> npt.NDArray[int]:
        if self.flatten_state:
            return np.moveaxis(self._state_raw.flatten(), -1, 0) / 255
        else:
            return np.moveaxis(self._state_raw, -1, 0) / 255


    def __skip_game_start(self):
        # TODO make sure it's 65
        # The first 65 frames of the game are static. Each action is the same as NOOP (action 0)
        # It might be beneficial to completly skip those frames
        for _ in range(65):
            self.step(0)

