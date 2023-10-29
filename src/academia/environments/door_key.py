from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .base import GenericMiniGridWrapper


class DoorKey(GenericMiniGridWrapper):
    """
    A grid environment where an agent has to find a key and then open a door to reach the destination.
    The higher the difficulty, the bigger the grid so it is more complicated to find the key, next the door 
    and next the destination.
    Possible actions:

    | Num | Name     | Action             |
    |-----|----------|--------------------|
    | 0   | left     | Turn left          |
    | 1   | right    | Turn right         |
    | 2   | forward  | Move forward       |
    | 3   | pickup   | Pick up an object  |
    | 4   | drop     | Unused             |
    | 5   | toggle   | Toggle/activate an object |
    | 6   | done     | Unused             |

    Possible difficulty levels:
    0: 5x5 grid size with 1 key and 1 door
    1: 6x6 grid size with 1 key and 1 door
    2: 8x8 grid size with 1 key and 1 door
    3: 16x16 grid size with 1 key and 1 door
    """

    N_ACTIONS = 6

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

        self._door_status = 2
        super().__init__(difficulty, DoorKey.__difficulty_envid_map, render_mode=render_mode)
    
    def get_legal_mask(self) -> npt.NDArray[Union[bool, int]]:
        mask = np.array([1 for _ in range(self.N_ACTIONS)])
        mask[4] = 0 #drop action is unused
        return mask
    
    @property
    def _state(self) -> tuple[int, ...]:
        """
        This property takes the raw state representation (self.__state) returned
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

        :return: a tuple of object types of every grid cell concatenated with
                 the direction which the agent is facing and with door state.
        """

        cells_obj_types: np.ndarray = self._state_raw['image'][:, :, 0]
        cells_flattened = cells_obj_types.flatten()
        direction = self._state_raw['direction']
        door_array = self._state_raw['image'][:,:,2].flatten()

        if self._door_status == 2 and 1 in door_array:
            self._door_status = 1
        elif self._door_status == 1 and 0 in door_array:
            self._door_status = 0
            
        return np.array([*cells_flattened, direction, self._door_status])