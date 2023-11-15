from collections import deque
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from .base import ScalableEnvironment


class BridgeBuilding(ScalableEnvironment):
    
    N_ACTIONS = 8

    __RIVER_WIDTH = 3
    __N_BOULDERS = __RIVER_WIDTH
    __RIVER_HEIGHT = 3
    __LEFT_BANK_WIDTH = 3
    __TOTAL_WIDTH = __LEFT_BANK_WIDTH + __RIVER_WIDTH + 1

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
    def __player_holds_boulder(self):
        return self.__held_boulder_index >= 0

    @property
    def _state(self):
        return np.concatenate([
            self.__player_position, self.__player_target, self.__boulder_positions.flatten()
        ])

    def __init__(self, 
                 difficulty: int, 
                 max_steps: int = 100, 
                 render_mode: Optional[Literal["human"]] = None,
                 obs_type: Literal["string", "array"] = "string",
                 n_frames_stacked: int = 1,
                 append_step_count: bool = False, 
                 random_state: Optional[int] = None) -> None:
        self.difficulty = difficulty
        self.__init_bridge_length = self.__RIVER_WIDTH - difficulty
        self.__boulder_positions = np.zeros(shape=(self.__N_BOULDERS,2))
        self.__player_position = np.zeros(shape=(2))
        self.__player_direction = 0
        self.__held_boulder_index = -1

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.n_frames_stacked = n_frames_stacked
        self.append_step_count = append_step_count
        self.step_count = 0
        # self._state = None  # properly set in self.reset()
        # """note: self._state IS NOT STACKED. To obtain a stacked state use self.observe()"""
        self._past_n_states = deque()  # properly set in self.reset()
        self._rng = np.random.default_rng(random_state)

        # using a separate rng in case other parts of the code depend
        # on self._rng which would result in inconsistent subsequent restarts
        # even with the same `random_state` especially if self._rng was used in self.step
        # (which can be called a variable number of times between resets).
        # It's probably not going to matter but I think it's good to future proof
        # in case we want to add some randomness to the environment
        self.__init_state_rng = np.random.default_rng(self._rng.integers(0, 999999999999))

        self.reset()
        observed_state = self.observe()
        if self.obs_type == "string":
            self.STATE_SHAPE = (len(observed_state),)
        elif self.obs_type == "array":
            self.STATE_SHAPE = observed_state.shape

    def reset(self) -> Union[str, npt.NDArray[np.float32]]:
        self.step_count = 0
        self.__generate_initial_state()
        self._past_n_states = deque([self._state for _ in range(self.n_frames_stacked)])
        return self.observe()

    def observe(self) -> Union[str, npt.NDArray[np.float32]]:
        stacked_state = np.concatenate(list(self._past_n_states))
        if self.append_step_count:
            stacked_state = np.append(stacked_state, self.step_count)
        if self.obs_type == "string":
            str_state = ""
            for element in stacked_state: 
                str_state += str(element)
            return str_state
        return stacked_state

    def step(self, action) -> tuple[Union[str, npt.NDArray[np.float32]], float, bool]:
        self.step_count += 1
        is_terminal = self.step_count >= self.max_steps

        self.__handle_action(action)
        self._past_n_states.append(self._state)
        self._past_n_states.popleft()

        if self.__is_on_goal(self.__player_position):
            reward = 1 - self.step_count / self.max_steps
            if self.__is_boulder_left():
                reward = np.maximum(0, reward - 0.5)
            return self.observe(), reward, True
        elif self.__is_on_unbridged_river(self.__player_position):
            is_terminal = True
        
        return self.observe(), 0, is_terminal

    def render(self) -> None:
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
        # turning left/right is always legal
        legal_mask = np.array([1 for _ in range(self.N_ACTIONS)])

        if not self.__is_valid(self.__player_target):
            # can't move forward or pickup since it's an invalid target
            legal_mask[2:] = 0
            return
        
        # from now on we can assume target is valid
        if not self.__is_walkable(self.__player_target):
            legal_mask[2] = 0
        if self.__player_holds_boulder and self.__contains_boulder(self.__player_target):
            legal_mask[3] = 0
        if not self.__player_holds_boulder and not self.__contains_boulder(self.__player_target):
            legal_mask[3] = 0

    def __handle_action(self, action):
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
        self.__player_direction = (self.__player_direction - 1) % 4

    def __turn_right(self):
        self.__player_direction = (self.__player_direction + 1) % 4

    def __pickup(self):
        # assuming valid target guaranteed by `__handle_action`
        self.__held_boulder_index = self.__boulder_index_at_target()
        if self.__held_boulder_index == -1:
            return
        self.__boulder_positions[self.__held_boulder_index, :] = (-1, -1)

    def __drop(self):
        # assuming valid target guaranteed by `__handle_action`
        if self.__contains_boulder(self.__player_target):
            return
        self.__boulder_positions[self.__held_boulder_index, :] = self.__player_target
        self.__held_boulder_index = -1
        
    def __move_forward(self):
        # assuming valid target guaranteed by `__handle_action`
        if self.__is_walkable(self.__player_target):
            self.__player_position = self.__player_target

    def __generate_initial_state(self) -> None:
        # generate initial bridge
        bridge_y = self.__init_state_rng.integers(0, self.__RIVER_HEIGHT - 1)
        for i in range(self.__init_bridge_length):
            self.__boulder_positions[i, :] = self.__LEFT_BANK_WIDTH + i, bridge_y

        # generate boulders and player postions
        positions = self.__init_state_rng.choice(np.arange(self.__RIVER_HEIGHT * self.__LEFT_BANK_WIDTH),
                                     size=1 + self.difficulty, replace=False)
        self.__player_position[:] = \
            np.unravel_index(positions[0], (self.__RIVER_HEIGHT, self.__LEFT_BANK_WIDTH))
        self.__player_direction = self.__init_state_rng.integers(0, 3)

        for position_idx, boulder_idx in enumerate(range(self.__init_bridge_length, self.__N_BOULDERS)):
            x, y = np.unravel_index(positions[position_idx], (self.__RIVER_HEIGHT, self.__LEFT_BANK_WIDTH))
            self.__boulder_positions[boulder_idx, :] = x, y

    def __is_valid(self, position):
        x, y = position
        if x < 0 or x >= self.__TOTAL_WIDTH:
            return False
        if y < 0 or y >= self.__RIVER_HEIGHT:
            return False
        return True

    def __is_on_river(self, position):
        # assuming valid position
        return self.__LEFT_BANK_WIDTH <= position[0] < self.__LEFT_BANK_WIDTH + self.__RIVER_WIDTH

    def __is_on_bridge(self, position):
        # assuming valid position
        return self.__is_on_river(position) and self.__contains_boulder(position)

    def __is_on_unbridged_river(self, position):
        # assuming valid position
        return self.__is_on_river(position) and not self.__is_on_bridge(position)

    def __is_walkable(self, position):
        # assuming valid position
        return not self.__contains_boulder(position) or self.__is_on_bridge(position)

    def __is_on_goal(self, position):
        # assuming valid position
        return position[0] == self.__TOTAL_WIDTH - 1

    def __contains_boulder(self, position):
        # assuming valid position
        return np.any(np.all(position == self.__boulder_positions, axis=1))
    
    def __boulder_index_at_target(self):
        # assuming valid position
        index = np.where(np.all(self.__player_target == self.__boulder_positions, axis=1))[0]
        if len(index) == 0:
            return -1
        return index[0]
    
    def __is_boulder_left(self):
        for position in self.__boulder_positions:
            if position[0] < self.__LEFT_BANK_WIDTH:
                return True
        return False

    def __are_positions_equal(self, p1, p2):
        # since positions are numpy arrays they need np.all to assert equality
        # this method is a human readable representation of that fact
        return np.all(p1 == p2)