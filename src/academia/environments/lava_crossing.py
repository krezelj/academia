from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .base import GenericMiniGridWrapper


class LavaCrossing(GenericMiniGridWrapper):
    """
    A grid environment where an agent has to avoid patches of lava in order to
    reach the destination. The higher the difficulty, the more lava patches are
    generated on the grid.

    Possible actions:

    | Num | Name    | Action             |
    |-----|---------|--------------------|
    | 0   | left    | Turn left          |
    | 1   | right   | Turn right         |
    | 2   | forward | Move forward       |
    | 3   | pickup  | Unused             |
    | 4   | drop    | Unused             |
    | 5   | toggle  | Unused             |
    | 6   | done    | Unused             |

    Possible difficulty levels:
    0: 9x9 grid size with 1 lava patch
    1: 9x9 grid size with 2 lava patches
    2: 9x9 grid size with 3 lava patches
    3: 11x11 grid size with 5 lava patches
    """

    N_ACTIONS = 3

    __difficulty_envid_map = {
        0: 'MiniGrid-LavaCrossingS9N1-v0',
        1: 'MiniGrid-LavaCrossingS9N2-v0',
        2: 'MiniGrid-LavaCrossingS9N3-v0',
        3: 'MiniGrid-LavaCrossingS11N5-v0'
    }
    """A dictionary that maps difficulty levels to environment ids"""

    def __init__(self, difficulty: int, n_frames_stacked: int = 1, render_mode: Optional[str] = None,
                 **kwargs):
        """
        :param difficulty:  Difficulty level from 0 to 3, where 0 is the easiest
                            and 3 is the hardest
        :param render_mode: render_mode value passed to gymnasium.make
        """
        
        super().__init__(
            difficulty=difficulty,
            difficulty_envid_map=LavaCrossing.__difficulty_envid_map,
            n_frames_stacked=n_frames_stacked,
            render_mode=render_mode,
            **kwargs,
        )

    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        return np.array([1 for _ in range(self.N_ACTIONS)])
        
    @property
    def _state(self) -> tuple[int, ...]:
        """
        This property takes the raw state representation (self._state_raw) returned
        by the base environment and transforms it so that it is compatible
        with the agent API provided by this package.

        The raw state representation is a dictionary with three items:

        - "image": an array of shape (width, height, 3) which encodes information on each cell
          in the agent's field of view,
        - "direction": an integer indicating the direction which agent is
          facing,
        - "mission": a static string that denotes the objective of the agent.

        Every cell on the grid is represented by a three-element array which values
        indicate respectively:

        - an integer encoding the object type (unseen, empty, lava, wall, etc.),
        - an integer encoding cell colour,
        - an integer encoding the door state (open, closed, locked).

        Details on each of these encodings can be found here:
        https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/constants.py

        The colour is not relevant to the agent and the door state is not used
        in this particular environment.

        To obtain the final result, object types of each cell and the direction
        the agent is facing are used.

        **Note:** the position of the agent is not marked on the 2D "image"
        array that comes in the input state. Agent's cell might be in one of
        four positions in this array - in the center of any of the array's
        sides. This position could be different in every generated environment,
        but once the environment is initialised this position will not change
        no matter the direction the agent is facing or its location on the grid.

        :return: an array of object types of every grid cell concatenated with
                 the direction which the agent is facing.
        """

        cells_obj_types: np.ndarray = self._state_raw['image'][:, :, 0]
        cells_flattened = cells_obj_types.flatten()
        direction = self._state_raw['direction']

        return np.array([*cells_flattened, direction])
    
