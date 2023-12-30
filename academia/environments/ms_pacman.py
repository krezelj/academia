from typing import Optional

import numpy as np
import numpy.typing as npt

from .base import GenericAtariWrapper


class MsPacman(GenericAtariWrapper):
    """
    This class is a wrapper for *Gymnasium*'s Ms Pacman environment.

    MsPacman is an Atari 2600 environment where the agent has to navigate a maze, eat pellets
    and avoid ghosts. The higher the difficulty, the more ghosts to avoid.

    Possible actions:

    +-----+-----------+----------------+
    | Num | Name      | Action         |
    +=====+===========+================+
    | 0   | NOOP      | Do nothing     |
    +-----+-----------+----------------+
    | 1   | UP        | Move up        |
    +-----+-----------+----------------+
    | 2   | RIGHT     | Move right     |
    +-----+-----------+----------------+
    | 3   | DOWN      | Move down      |
    +-----+-----------+----------------+
    | 4   | LEFT      | Move left      |
    +-----+-----------+----------------+
    | 5   | UPRIGHT   | Move upright   |
    +-----+-----------+----------------+
    | 6   | UPLEFT    | Move upleft    |
    +-----+-----------+----------------+
    | 7   | DOWNRIGHT | Move downright |
    +-----+-----------+----------------+
    | 8   | DOWNLEFT  | Move downleft  |
    +-----+-----------+----------------+

    Difficulty levels:

    +------------+----------------------------------------------+
    | Difficulty | Description                                  |
    +============+==============================================+
    | 0          | 1 ghost is chasing the player                |
    +------------+----------------------------------------------+
    | 1          | 2 ghosts are chasing the player              |
    +------------+----------------------------------------------+
    | 2          | 3 ghosts are chasing the player              |
    +------------+----------------------------------------------+
    | 3          | 4 ghosts are chasing the player              |
    +------------+----------------------------------------------+

    Note:
        For this environment the keyword argument ``mode`` is not used. This is because Ms Pacman did not
        use the difficulty settings available in Atari but did use mode settings to control the number of
        ghosts on the map. Because of this the :attr:`difficulty` parameter is mapped to ``mode``.

    See Also:
        *Gymnasium*'s Ms Pacman environment: https://www.gymlibrary.dev/environments/atari/ms_pacman/

    Args:
        difficulty: Difficulty level from 0 to 3, where 0 is the easiest and 3 is the hardest.
        n_frames_stacked: How many most recent states should be stacked together to form a final state
            representation. Defaults to 1.
        append_step_count: Whether or not append the current step count to each state. Defaults to ``False``.
        flatten_state: Wheter ot not to flatten the state if represented by and RGB or grayscale image.
            If ``obs_type`` is set to ``"ram"`` this parameter does nothing. Defaults to ``False``.
        skip_game_start: Whether or not skip the game start. After every reset the game is an "noop" state
            for 65 frames which can hinder the training process. If true the game skips this stage
            by applying 65 NOOP actions before returning the first observed state. Defaults to ``True``.
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
        skip_game_start (bool): Whether or not skip the game start.
    """
    
    N_ACTIONS = 9

    def __init__(self, 
                 difficulty: int, 
                 n_frames_stacked: int = 1,
                 append_step_count: bool = False,
                 flatten_state: bool = False,
                 skip_game_start: bool = True,
                 random_state: Optional[int] = None,
                 **kwargs) -> None:
        
        self.skip_game_start = skip_game_start
        if 0 > difficulty or 3 < difficulty:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 3")
            raise ValueError(msg)
        
        # See atariage.com manual for ms pacman for details
        kwargs['mode'] = 0 if difficulty == 3 else 1 + difficulty
        super(MsPacman, self).__init__(
            difficulty, "ALE/MsPacman-v5", n_frames_stacked,
            append_step_count, flatten_state, random_state, **kwargs)

    def reset(self) -> npt.NDArray[np.float32]:
        """
        Resets the environment to its initial state.

        Returns:
            The new state after resetting the environment.

        Note:
            if :attr:`skip_game_start` is set to ``True`` this method also performs 65 NOOP
            actions before returning the first observed state.
        """
        super().reset()
        if self.skip_game_start:
            self.__skip_game_start()
        return self.observe()

    def __skip_game_start(self):
        """
        Skips the initial part of each game by performing 65 NOOP actions
        """
        for _ in range(65):
            self.step(0)
