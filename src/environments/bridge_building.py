import numpy as np

from .base import BaseEnvironment


class BridgeBuilding(BaseEnvironment):

    N_ACTIONS=8

    STEP_PENALTY = -1
    DROWN_PENALTY = -100
    BRIDGE_PENALTY = -50
    BOULDER_LEFT_PENALTY = -20
    GOAL_REWARD = 100

    RIVER_WIDTH = 3
    N_BOULDERS = RIVER_WIDTH
    RIVER_HEIGHT = 3
    LEFT_BANK_WIDTH = 3
    TOTAL_WIDTH = LEFT_BANK_WIDTH + RIVER_WIDTH + 1
    BRIDGE_ENTRANCE = LEFT_BANK_WIDTH - 1, -1
    CAN_USE_BRIDGE = False

    

    __slots__ = ['episode_steps', 'max_steps', 'n_boulders_placed', 'start_with_boulder',
                 'player_position', 'player_has_boulder', 'boulder_positions', 'active_boulder_index']

    def __init__(self, max_steps=100, n_boulders_placed=0, start_with_boulder=False) -> None:
        self.max_steps = max_steps
        self.n_boulders_placed = n_boulders_placed
        self.start_with_boulder = start_with_boulder
        self.reset()

    def reset(self):
        self.episode_steps = 0
        self.boulder_positions = [None for _ in range(self.N_BOULDERS)]
        self.__generate_initial_state()
        return self.observe()

    def step(self, action):

        self.episode_steps += 1
        reward = self.STEP_PENALTY
        is_terminal = self.episode_steps >= self.max_steps

        is_walk_action = (action & 4) == 0
        action_direction = action & 3
        position_offset = self.__get_offset_from_direction(action_direction)
        target = self.player_position[0] + position_offset[0], self.player_position[1] + position_offset[1]

        if not self.__is_target_valid(target):
            return self.observe(), reward, is_terminal
        if is_walk_action:
            if target == self.BRIDGE_ENTRANCE:
                self.player_position = self.TOTAL_WIDTH - 1, 0
                reward += self.BRIDGE_PENALTY
                is_terminal = True
                return self.observe(), reward, is_terminal
            if self.__is_on_river(target) or not self.__collides_with_boulder(target):
                self.player_position = target
            if self.__is_on_river(target) and not self.__collides_with_boulder(target):
                reward += self.DROWN_PENALTY
                is_terminal = True
            if target[0] == self.TOTAL_WIDTH - 1: # GOAL POSITION
                if self.__is_boulder_left():
                    reward += self.BOULDER_LEFT_PENALTY
                reward += self.GOAL_REWARD
                is_terminal = True
        else:
            if self.player_has_boulder:
                if not self.__collides_with_boulder(target):
                    self.boulder_positions[self.active_boulder_index] = target
                    self.active_boulder_index = None
                    self.player_has_boulder = False
            else:
                self.active_boulder_index = self.__get_target_boulder_index(target)
                if self.active_boulder_index is not None:
                    self.boulder_positions[self.active_boulder_index] = (-1, -1)
                    self.player_has_boulder = True

        return self.observe(), reward, is_terminal

    def observe(self):
        state = self.player_position
        for i in range(self.N_BOULDERS): 
            state += self.boulder_positions[i]
        return state
    
    def render(self):
        pass

    def get_legal_mask(self):
        mask = np.zeros(self.N_ACTIONS)
        offsets = [self.__get_offset_from_direction(d) for d in range(4)]
        for i, offset in enumerate(offsets):
            target = self.player_position[0] + offset[0], self.player_position[1] + offset[1]
            if self.__is_target_valid(target):
                collides_with_boulder = self.__collides_with_boulder(target)
                if not collides_with_boulder or self.__is_on_river(target):
                    mask[i] = 1
                if collides_with_boulder ^ self.player_has_boulder:
                    mask[i+4] = 1
        return mask

    def __is_boulder_left(self):
        for position in self.boulder_positions:
            if position[0] < self.LEFT_BANK_WIDTH:
                return True
        return False

    def __is_on_river(self, target):
        # assuming target is valid
        return self.LEFT_BANK_WIDTH <= target[0] < self.LEFT_BANK_WIDTH + self.RIVER_WIDTH
    
    def __collides_with_boulder(self, target):
        return target in self.boulder_positions
    
    def __get_target_boulder_index(self, target):
        for i in range(self.N_BOULDERS):
            if target == self.boulder_positions[i]:
                return i
        return None
    
    def __is_target_valid(self, target):
        return ((0 <= target[0] < self.TOTAL_WIDTH) and \
                (0 <= target[1] < self.RIVER_HEIGHT)) or \
                (self.CAN_USE_BRIDGE and \
                target==self.BRIDGE_ENTRANCE)

    def __get_offset_from_direction(self, direction_num):
        if direction_num == 0:
            return (0, 1)
        elif direction_num == 1:
            return (1, 0)
        elif direction_num == 2:
            return (0, -1)
        elif direction_num == 3:
            return (-1, 0)
        return None

    def __generate_initial_state(self):
        def generate_non_overlapping_position(end_index):
            # generates a postion on the left bank that does not overlap with any boulder 
            # on the left bank up until `end_index`
            while True:
                x_position = np.random.randint(0, self.LEFT_BANK_WIDTH)
                y_position = np.random.randint(0, self.RIVER_HEIGHT)
                for j in range(self.n_boulders_placed, end_index):
                    if self.boulder_positions[j][0] == x_position and self.boulder_positions[j][1] == y_position:
                        break
                else:
                    return x_position, y_position

        # generate boulders in the river
        bridge_y_position = np.random.randint(0, self.RIVER_HEIGHT)
        for i in range(self.n_boulders_placed):
            self.boulder_positions[i] = (i + self.LEFT_BANK_WIDTH, bridge_y_position)

        # generate boulders on the bank
        for i in range(self.n_boulders_placed, self.N_BOULDERS):
            self.boulder_positions[i] = generate_non_overlapping_position(i)

        # generate player position
        self.player_position = generate_non_overlapping_position(self.N_BOULDERS)

        self.player_has_boulder=self.start_with_boulder
        if self.player_has_boulder:
            self.active_boulder_index = -1
            self.boulder_positions[-1] = -1, -1

    def __str__(self):
        str_representation = ""
        for y in range(self.RIVER_HEIGHT - 1, -1, -1):
            for x in range(self.TOTAL_WIDTH):
                if (x, y) == self.player_position:
                    str_representation += 'B' if self.player_has_boulder else 'P'
                elif (x, y) in self.boulder_positions:
                    str_representation += '#'
                elif self.LEFT_BANK_WIDTH <= x < self.LEFT_BANK_WIDTH + self.RIVER_WIDTH:
                    str_representation += '~'
                else:
                    str_representation += '.'
            str_representation += '\n'
        return str_representation



def main():
    bb = BridgeBuilding(n_boulders_placed=2)
    print(bb)

if __name__ == '__main__':
    main()