from collections import deque
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from .base import ScalableEnvironment


class BridgeBuilding(ScalableEnvironment):
    """
    A grid environment where an agent has to use boulders scattered around the map to
    build a bridge across a river and get from one side to the other. 
    The higher the difficulty the more work the agent has to put into building the bridge.
    At lower difficulties the bridge is partly (or even fully) built and the 
    agent only has to learn how to finish it and/or navigate it.

    The reward system is presented in the table below. Note that the last two rewards
    can only be obtained if :attr:`reward_density` is set to ``"dense"``:

    +-----------------------------------+------------------------------------------------------------------------------+
    | Event                             | Reward                                                                       |
    +===================================+==============================================================================+
    | Running out of time               | 0                                                                            |
    +-----------------------------------+------------------------------------------------------------------------------+
    | Drowning in the river             | 0                                                                            |
    +-----------------------------------+------------------------------------------------------------------------------+
    | Reaching the goal                 | 1 - ``step_count``/:attr:`max_steps` + ``bridge_length``/:attr:`river_width` |
    +-----------------------------------+------------------------------------------------------------------------------+
    | (Dense) Constructing the bridge   | 0.5 * (``length_after_step`` - ``length_before_step``)                       |
    +-----------------------------------+------------------------------------------------------------------------------+
    | (Dense) Deconstructing the bridge | 0.5 * (``length_after_step`` - ``length_before_step``)                       |
    +-----------------------------------+------------------------------------------------------------------------------+

    The main reward function (reaching the goal) is meant to mimic *Minigrid*'s reward
    function but its last component also forces the agent to fully build the bridge.

    Possible actions:

    +-----+-------------+-----------------------+
    | Num | Name        | Action                |
    +=====+=============+=======================+
    | 0   | left        | Turn left             |
    +-----+-------------+-----------------------+
    | 1   | right       | Turn right            |
    +-----+-------------+-----------------------+
    | 2   | forward     | Move forward          |
    +-----+-------------+-----------------------+
    | 3   | pickup/drop | Pick up/Drop boulder  |
    +-----+-------------+-----------------------+

    Difficulty levels:

    +------------+---------------------------------------------+
    | Difficulty | Description                                 |
    +============+=============================================+
    | n          | The bridge is missing n boulders            |
    +------------+---------------------------------------------+

    Args:
        difficulty: Difficulty level from 0 to :attr:`river_width`, where 0 is the easiest
            and :attr:`river_width` is the hardest.
        river_width: The width of the river.
        max_steps: The maximum number of steps an agent can spend in the environment.
            If the agent doesn't reach the goal in that time the episode terminates. Defaults to ``100``.
        render_mode: How the environment should be rendered. If set to ``"human"`` the environment
            will be rendered in a way interpretable by a human. Defaults to ``None``.
        obs_type: How should the state be observed. If ``"string"`` a string representing the state
            will be returned. If ``"array"`` an array representing the state will be returned.
            Defaults to ``"array"``.
        reward_density: The density of the reward function. Possible values are ``"sparse"`` and ``"dense"``.
            If ``"sparse"`` is passed the agent will only get the reward at the end of the episode.
            If ``"dense"`` is passed the agent will additionally obtain rewards (and penalties) for
            constructing (or deconstructing) parts of the bridge. Defaults to ``"sparse"``.
        n_frames_stacked: How many most recent states should be stacked together to form a final state
            representation. Defaults to 1.
        append_step_count: Whether or not append the current step count to each state. Defaults to ``False``.
        random_state: Optional seed that controls the randomness of the environment. Defaults to ``None``.

    Raises:
        ValueError: If the specified river width level is invalid.
        ValueError: If the specified difficulty level is invalid.

    Attributes:
        step_count (int): Current step count since the last reset.
        difficulty (int): Difficulty level. Higher values indicate more difficult environments.
        river_width (int): The width of the river.
        n_frames_stacked (int): How many most recent states should be stacked together to form a final state
            representation.
        append_step_count (bool): Whether or not append the current step count to each state.
        max_steps (int): The maximum number of steps an agent can spend in the environment.
        render_mode Optiona[Literal["human"]]: How the environment should be rendered.
        obs_type Literal["string", "array"]: How should the state be observed.
        reward_density: The density of the reward function.

    """
    
    N_ACTIONS = 4

    @property
    def __player_target(self):
        offset = None
        if self.__player_direction == 0: # up
            offset = np.array([0, 1])
        elif self.__player_direction == 1: # right
            offset = np.array([1, 0])
        elif self.__player_direction == 2: # down
            offset = np.array([0, -1])
        elif self.__player_direction == 3: # left
            offset = np.array([-1, 0])
        return self.__player_position + offset

    @property
    def __player_holds_boulder(self) -> bool:
        return self.__held_boulder_index >= 0
    
    @property
    def __bridge_length(self) -> int:
        """
        Returns the length of the bridge as the number of horizontal, 
        consecutive boulders inside the river touching the left bank.

        Examples:
        . # . ~ ~ ~ .
        # . . ~ ~ ~ .
        . . # ~ ~ ~ .
        bridge length is 0

        . # . # ~ ~ .
        . . . # ~ ~ .
        . . . ~ ~ ~ .
        bridge length is 1

        . . . # ~ ~ .
        . . . # # ~ .
        . . . ~ ~ ~ .
        bridge length is 2

        . . . # ~ ~ .
        . . . # . # .
        . . . ~ ~ ~ .
        bridge length is 1

        . . . ~ # ~ .
        . . . ~ # # .
        . . . ~ ~ ~ .
        bridge length is 0

        """
        # positions of boulders inside the river touching the left bank
        bridge_entrances = self.__boulder_positions[
            np.where(self.__boulder_positions[:,0] == self.__LEFT_BANK_WIDTH)[0],:]
        max_bridge_length = 0
        for entrance in bridge_entrances:
            current_bridge_length = 1
            right_most_boulder = entrance
            while self.__is_on_bridge(right_most_boulder + (1, 0)):
                right_most_boulder = right_most_boulder + (1, 0)
                current_bridge_length += 1
            max_bridge_length = max(max_bridge_length, current_bridge_length)
        return max_bridge_length
        
    def __init__(self, 
                 difficulty: int,
                 river_width: int = 2,
                 max_steps: int = 100, 
                 render_mode: Optional[Literal["human"]] = None,
                 obs_type: Literal["string", "array"] = "array",
                 reward_density: Literal["sparse", "dense"] = "sparse",
                 n_frames_stacked: int = 1,
                 append_step_count: bool = False, 
                 random_state: Optional[int] = None) -> None:
        if not isinstance(river_width, int) or not river_width > 0:
            raise ValueError("Incorrect river width. Must be a positive integer")
        if not isinstance(difficulty, int) or not (0 <= difficulty <= river_width):
            raise ValueError(f"Incorrect difficulty. Must be an integer in range [0, {river_width}]")
        self.difficulty = difficulty
        self.river_width = river_width
        self.__init_map_constants()
        self.__init_bridge_length = self.river_width - self.difficulty
        self.__boulder_positions = np.zeros(shape=(self.__N_BOULDERS,2))
        self.__player_position = np.zeros(shape=(2))
        self.__player_direction = 0
        self.__held_boulder_index = -1

        self.reward_density = reward_density
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.n_frames_stacked = n_frames_stacked
        self.append_step_count = append_step_count
        self.step_count = 0
        self._past_n_states = deque()  # properly set in self.reset()
        self._rng = np.random.default_rng(random_state)

        # using a separate rng in case other parts of the code depend
        # on self._rng which would result in inconsistent subsequent restarts
        # even with the same `random_state` especially if self._rng was used in self.step
        # (which can be called a variable number of times between resets).
        # It's probably not going to matter but I think it's good to future proof
        # in case we want to add some randomness to the environment
        self.__init_state_rng = np.random.default_rng(self._rng.integers(0, 999999999999))

        self._state = None
        self.reset()
        observed_state = self.observe()
        if self.obs_type == "string":
            self.STATE_SHAPE = (len(observed_state),)
        elif self.obs_type == "array":
            self.STATE_SHAPE = observed_state.shape

    def __init_map_constants(self):
        self.__N_BOULDERS = self.river_width
        self.__RIVER_HEIGHT = 3
        self.__LEFT_BANK_WIDTH = 3
        self.__TOTAL_WIDTH = self.__LEFT_BANK_WIDTH + self.river_width + 1

    def reset(self) -> Union[str, npt.NDArray[np.float32]]:
        """
        Resets the environment to its initial state.

        Returns:
            The new state after resetting the environment.
        """
        self.step_count = 0
        self.__generate_initial_state()
        self._past_n_states = deque([self._state for _ in range(self.n_frames_stacked)])
        self.render()
        return self.observe()

    def observe(self) -> Union[str, npt.NDArray[np.float32]]:
        """
        Returns the current state of the environment. Performs state stacking if :attr:`n_frames_stacked` is
        greater than 1.

        Returns:
            The current state of the environment.
        """
        stacked_state = np.concatenate(list(self._past_n_states))
        if self.append_step_count:
            stacked_state = np.append(stacked_state, self.step_count / self.max_steps)
        if self.obs_type == "string":
            str_state = ""
            for element in stacked_state: 
                str_state += str(int(element))
            return str_state
        return stacked_state

    def step(self, action) -> tuple[Union[str, npt.NDArray[np.float32]], float, bool]:
        """
        Advances the environment by one step given the specified action.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the new state, reward, and a flag indicating episode termination.
        """
        self.step_count += 1
        is_terminal = self.step_count >= self.max_steps
        reward = 0.0

        bridge_length_before = self.__bridge_length
        self.__handle_action(action)
        self.__compose_state()
        bridge_length_after = self.__bridge_length
        if self.reward_density == "dense":
            reward += (bridge_length_after - bridge_length_before) / self.river_width

        self._past_n_states.append(self._state)
        self._past_n_states.popleft()

        if self.__is_on_goal(self.__player_position):
            reward = 1 - self.step_count / self.max_steps
            reward += self.__bridge_length / self.river_width
            is_terminal = True
        elif self.__is_on_unbridged_river(self.__player_position):
            reward = 0
            is_terminal = True

        self.render()
        return self.observe(), reward, is_terminal

    def render(self) -> None:
        """
        Renders the environment in the current render mode.
        """

        if self.render_mode != 'human':
            return
        str_representation = ""
        for y in range(self.__RIVER_HEIGHT - 1, -1, -1):
            for x in range(self.__TOTAL_WIDTH):
                position = x, y
                if self.__are_positions_equal(position, self.__player_position):
                    str_representation += 'B' if self.__player_holds_boulder else 'P'
                elif self.__are_positions_equal(position, self.__player_target):
                    str_representation += 'X'
                elif self.__contains_boulder(position):
                    str_representation += '#'
                elif self.__is_on_river(position):
                    str_representation += '~'
                else:
                    str_representation += '.'
            str_representation += '\n'
        print(str_representation)

    def get_legal_mask(self) -> npt.NDArray[np.int32]:
        """
        Returns a binary legal action mask.

        Returns:
            A binary mask with 0s in place for illegal actions (actions that
            have no effect) and 1s for legal actions.
        """

        # turning left/right is always legal
        legal_mask = np.array([1 for _ in range(self.N_ACTIONS)])

        if not self.__is_valid(self.__player_target):
            # can't move forward or pickup since it's an invalid target
            legal_mask[2:] = 0
            return legal_mask
        
        # from now on we can assume target is valid
        if not self.__is_walkable(self.__player_target):
            legal_mask[2] = 0
        if self.__player_holds_boulder and self.__contains_boulder(self.__player_target):
            legal_mask[3] = 0
        if not self.__player_holds_boulder and not self.__contains_boulder(self.__player_target):
            legal_mask[3] = 0

        return legal_mask

    def __compose_state(self):
        """
        Composes a new value of _state using environment features.
        """
        self._state = np.concatenate([
            self.__player_position, self.__player_target, self.__boulder_positions.flatten()
        ]).astype(np.float32)

    def __handle_action(self, action):
        """
        Updates the agent and the environment based on the specified action
        """
        if action == 0:
            self.__turn_left()
            return
        if action == 1:
            self.__turn_right()
            return
        
        if not self.__is_valid(self.__player_target):
            return
        
        if action == 2:
            self.__move_forward()
            return
        if action == 3:
            if self.__player_holds_boulder:
                self.__drop()
            else:
                self.__pickup()

    def __turn_left(self):
        """
        Turns the agent left
        """
        self.__player_direction = (self.__player_direction - 1) % 4

    def __turn_right(self):
        """
        Turns the agent right
        """
        self.__player_direction = (self.__player_direction + 1) % 4

    def __pickup(self):
        """
        Picks up a boulder in front of the agent if there is one. Otherwise does nothing.
        """
        # assuming valid target guaranteed by `__handle_action`
        self.__held_boulder_index = self.__boulder_index_at_target()
        if self.__held_boulder_index == -1:
            return
        self.__boulder_positions[self.__held_boulder_index, :] = (-1, -1)

    def __drop(self):
        """
        Drops the boulder the agent is holding in front of the agent if there isn't a boulder there already.
        Otherwise does nothing.
        """
        # assuming valid target guaranteed by `__handle_action`
        if self.__contains_boulder(self.__player_target):
            return
        self.__boulder_positions[self.__held_boulder_index, :] = self.__player_target
        self.__held_boulder_index = -1
        
    def __move_forward(self):
        """
        Moves the agent forward if the target position is walkable (isn't occupied by a boulder on land)
        """
        # assuming valid target guaranteed by `__handle_action`
        if self.__is_walkable(self.__player_target):
            self.__player_position = self.__player_target

    def __generate_initial_state(self) -> None:
        """
        Generates the initial state of the environment by creating a bridge (if necessary)
        and scattering the boulders and the player randomly on the left bank of the river.
        """

        # temporarily shuffle boulder indices to preventt the agent learning
        # to build the bridge in a particular boulder order when using curriculum
        idx_shuffle = np.arange(self.__N_BOULDERS)
        self.__init_state_rng.shuffle(idx_shuffle)

        # generate initial bridge
        bridge_y = self.__init_state_rng.integers(0, self.__RIVER_HEIGHT - 1)
        for i in range(self.__init_bridge_length):
            self.__boulder_positions[idx_shuffle[i], :] = self.__LEFT_BANK_WIDTH + i, bridge_y

        # generate boulders and player positions
        positions = self.__init_state_rng.choice(np.arange(self.__RIVER_HEIGHT * self.__LEFT_BANK_WIDTH),
                                     size=1 + self.difficulty, replace=False)
        self.__player_position[:] = \
            np.unravel_index(positions[0], (self.__RIVER_HEIGHT, self.__LEFT_BANK_WIDTH))
        self.__player_direction = self.__init_state_rng.integers(0, 3)

        for position_idx, boulder_idx in enumerate(range(self.__init_bridge_length, self.__N_BOULDERS)):
            x, y = np.unravel_index(positions[position_idx + 1], (self.__RIVER_HEIGHT, self.__LEFT_BANK_WIDTH))
            self.__boulder_positions[idx_shuffle[boulder_idx], :] = x, y

        self.__compose_state()

    def __is_valid(self, position):
        """
        Checks if the position is valid (inside map borders).
        """
        x, y = position
        if x < 0 or x >= self.__TOTAL_WIDTH:
            return False
        if y < 0 or y >= self.__RIVER_HEIGHT:
            return False
        return True

    def __is_on_river(self, position):
        """
        Checks if the position is on the river (including a bridge).
        """
        # assuming valid position
        return self.__LEFT_BANK_WIDTH <= position[0] < self.__LEFT_BANK_WIDTH + self.river_width

    def __is_on_bridge(self, position):
        """
        Checks if the position is on the bridge.
        """
        # assuming valid position
        return self.__is_on_river(position) and self.__contains_boulder(position)

    def __is_on_unbridged_river(self, position):
        """
        Checks if the position is on the river but not on the bridge.
        """
        # assuming valid position
        return self.__is_on_river(position) and not self.__is_on_bridge(position)

    def __is_walkable(self, position):
        """
        Checks if the position is walkable by the agent.
        """
        # assuming valid position
        return not self.__contains_boulder(position) or self.__is_on_bridge(position)

    def __is_on_goal(self, position):
        """
        Checks if the position is on the goal.
        """
        # assuming valid position
        return position[0] == self.__TOTAL_WIDTH - 1

    def __contains_boulder(self, position):
        """
        Checks if the position contains a boulder.
        """
        # assuming valid position
        return np.any(np.all(position == self.__boulder_positions, axis=1))
    
    def __boulder_index_at_target(self):
        """
        Returns the index of the boulder at the player target position.
        Returns -1 if there is no boulder there.
        """
        # assuming valid position
        index = np.where(np.all(self.__player_target == self.__boulder_positions, axis=1))[0]
        if len(index) == 0:
            return -1
        return index[0]
    
    @staticmethod
    def __are_positions_equal(p1, p2):
        """
        Checks if two given positions are the same.
        """
        # since positions are numpy arrays they need np.all to assert equality
        # this method is a human readable representation of that fact
        return np.all(p1 == p2)
    