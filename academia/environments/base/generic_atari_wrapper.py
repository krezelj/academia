from typing import Optional

import numpy as np
import numpy.typing as npt

from . import GenericGymnasiumWrapper


class GenericAtariWrapper(GenericGymnasiumWrapper):
    """
    A wrapper for Atari environments that makes them scalable.

    Args:
        difficulty: Difficulty level from 0 to 3, where 0 is the easiest and 3 is the hardest.
        n_frames_stacked: How many most recent states should be stacked together to form a final state
            representation. Defaults to 1.
        append_step_count: Whether or not append the current step count to each state. Defaults to ``False``.
        flatten_state: Wheter ot not to flatten the state if represented by and RGB or grayscale image.
            If ``obs_type`` is set to ``"ram"`` this parameter does nothing. Defaults to ``False``.
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
        flatten_state (bool): Wheter ot not to flatten the state if represented by and RGB or grayscale image.
    """
    
    def __init__(self, 
                 difficulty: int, 
                 environment_id: str,
                 n_frames_stacked: int = 1,
                 append_step_count: bool = False,
                 flatten_state: bool = False,
                 random_state: Optional[int] = None,
                 **kwargs) -> None:
        
        self.flatten_state = flatten_state
        super().__init__(difficulty, environment_id, n_frames_stacked, append_step_count,
                         random_state, **kwargs)

    def _transform_state(self, raw_state: npt.NDArray) -> npt.NDArray[np.float32]:
        """
        This method takes the raw state representation returned
        by the base environment and transforms it so that it is compatible
        with the agent API provided by this package.

        The raw state representation is either an RGB image, grayscale image 
        or 128 bytes of console RAM.

        To obtain the final result the data is scaled to fit inside [0,1] range
        and if :attr:`flatten_state` is set to ``True`` the data is also flattened to a 1D array.

        Returns:
            An array representing a scaled and potentially flattended image or scaled RAM content.
        """
        if self._base_env.spec.kwargs['obs_type'] == 'grayscale':
            raw_state = np.reshape(raw_state, (210, 160, 1))
        if self.flatten_state:
            return np.moveaxis(raw_state.flatten(), -1, 0) / 255
        else:
            return np.moveaxis(raw_state, -1, 0) / 255
