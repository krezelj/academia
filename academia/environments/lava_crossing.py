from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .base import GenericMiniGridWrapper


class LavaCrossing(GenericMiniGridWrapper):
    """
    This class is a wrapper for *MiniGrid*'s Lava Crossing environments.

    A grid environment where an agent has to avoid patches of lava in order to
    reach the destination. The higher the difficulty, the more lava patches are
    generated on the grid.

    Possible actions:

    +-----+---------+---------------+
    | Num | Name    | Action        |
    +=====+=========+===============+
    | 0   | left    | Turn left     |
    +-----+---------+---------------+
    | 1   | right   | Turn right    |
    +-----+---------+---------------+
    | 2   | forward | Move forward  |
    +-----+---------+---------------+

    Difficulty levels:

    +------------+----------------------------------------------+
    | Difficulty | Description                                  |
    +============+==============================================+
    | 0          | 9x9 grid size with 1 lava patch              |
    +------------+----------------------------------------------+
    | 1          | 9x9 grid size with 2 lava patches            |
    +------------+----------------------------------------------+
    | 2          | 9x9 grid size with 3 lava patches            |
    +------------+----------------------------------------------+
    | 3          | 11x11 grid size with 5 lava patches          |
    +------------+----------------------------------------------+

    See Also:
        *MiniGrid*'s Lava Crossing environments: https://minigrid.farama.org/environments/minigrid/CrossingEnv/

    Args:
        difficulty: Difficulty level from 0 to 3, where 0 is the easiest
            and 3 is the hardest.
        n_frames_stacked: How many most recent states should be stacked together to form a final state
            representation. Defaults to 1.
        append_step_count: Whether or not append the current step count to each state. Defaults to ``False``.
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
    """

    N_ACTIONS = 3

    __difficulty_envid_map = {
        0: 'MiniGrid-LavaCrossingS9N1-v0',
        1: 'MiniGrid-LavaCrossingS9N2-v0',
        2: 'MiniGrid-LavaCrossingS9N3-v0',
        3: 'MiniGrid-LavaCrossingS11N5-v0'
    }
    """A dictionary that maps difficulty levels to gymnasium environment ids"""

    def __init__(self, difficulty: int, n_frames_stacked: int = 1, append_step_count: bool = False,
                 random_state: Optional[int] = None, **kwargs):
        super().__init__(
            difficulty=difficulty,
            difficulty_envid_map=LavaCrossing.__difficulty_envid_map,
            n_frames_stacked=n_frames_stacked,
            append_step_count=append_step_count,
            random_state=random_state,
            **kwargs,
        )
        
    def _transform_state(self, raw_state: Any) -> npt.NDArray[np.float32]:
        """
        This method takes the raw state representation returned
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

        Notes:
            The position of the agent is not marked on the 2D "image"
            array that comes in the input state. Agent's cell might be in one of
            four positions in this array - in the center of any of the array's
            sides. This position could be different in every generated environment,
            but once the environment is initialised this position will not change
            no matter the direction the agent is facing or its location on the grid.

        Returns:
            An array of object types of every grid cell concatenated with the direction which the agent
            is facing.
        """
        cells_obj_types: np.ndarray = raw_state['image'][:, :, 0]
        cells_flattened = cells_obj_types.flatten()
        direction = raw_state['direction']
        return np.array([*cells_flattened, direction], dtype=np.float32)

