# TODO pip install ale-py, shimmy
from typing import Any


import numpy as np
import numpy.typing as npt


from . import GenericGymnasiumWrapper


class GenericAtariWrapper(GenericGymnasiumWrapper):
    
    def __init__(self, 
                 difficulty: int, 
                 environment_id: str,
                 n_frames_stacked: int = 1,
                 append_step_count: bool = False,
                 flatten_state: bool = False,
                 **kwargs) -> None:
        
        self.flatten_state = flatten_state
        super().__init__(difficulty, environment_id, n_frames_stacked, append_step_count, **kwargs)
        

    def _transform_state(self, raw_state: Any) -> npt.NDArray[int]:
        if self.flatten_state:
            return np.moveaxis(raw_state.flatten(), -1, 0) / 255
        else:
            return np.moveaxis(raw_state, -1, 0) / 255

