# TODO pip install ale-py, shimmy
from typing import Any


import numpy as np
import numpy.typing as npt


from .base import GenericAtariWrapper


class MsPacman(GenericAtariWrapper):
    
    N_ACTIONS = 9

    def __init__(self, 
                 difficulty: int, 
                 n_frames_stacked: int = 1,
                 append_step_count: bool = False,
                 skip_game_start: bool = True,
                 flatten_state: bool = False,
                 **kwargs) -> None:
        
        self.skip_game_start = skip_game_start
        
        # See atariage.com manual for ms pacman for details
        kwargs['mode'] = 0 if difficulty == 3 else 1 + difficulty
        super(MsPacman, self).__init__(
            difficulty, "ALE/MsPacman-v5", n_frames_stacked, append_step_count, flatten_state, **kwargs)
        

    def reset(self) -> Any:
        super().reset()
        if self.skip_game_start:
            self.__skip_game_start()
        return self.observe()

    def __skip_game_start(self):
        # TODO make sure it's 65
        # The first 65 frames of the game are static. Each action is the same as NOOP (action 0)
        # It might be beneficial to completly skip those frames
        for _ in range(65):
            self.step(0)

