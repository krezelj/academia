import numpy as np

class BridgeBuilding():

    STEP_PENALTY = -1
    DROWN_PENALTY = -100
    BRIDGE_PENALTY = -50

    RIVER_WIDTH = 3
    RIVER_HEIGHT = 3
    LEFT_BANK_WIDTH = 3
    TOTAL_WIDTH = LEFT_BANK_WIDTH + RIVER_WIDTH + 1
    # RIGHT_BANK_WIDTH = 1

    # GOAL_POSITION = (LEFT_BANK_WIDTH+RIVER_WIDTH+RIVER_WIDTH-1, RIVER_HEIGHT//2)

    __slots__ = ['episode_steps', 'max_steps', 'n_boulders_placed', 'start_with_boulder',
                 'player_position', 'player_has_boulder', 'boulder_positions', 'active_boulder_index']

    def __init__(self, max_steps=100, n_boulders_placed=0, start_with_boulder=False) -> None:
        self.max_steps = max_steps
        self.n_boulders_placed = n_boulders_placed
        self.start_with_boulder = start_with_boulder
        self.reset()

    def reset(self):
        self.episode_steps = 0
        self.boulder_positions = [None for _ in range(self.RIVER_WIDTH)]
        self.__generate_initial_state()
        return self.observe()

    def step(self, action):
        def is_on_river(target):
            return self.LEFT_BANK_WIDTH <= target[0] < self.LEFT_BANK_WIDTH + self.RIVER_WIDTH
        def collides_with_boulder(target):
            return target in self.boulder_positions
        def get_target_boulder_index(target):
            return None

        self.episode_steps += 1
        reward = self.STEP_PENALTY
        is_terminal = False

        # parse action
        is_walk_action = (action & 1) == 0
        action_direction = action >> 1
        position_offset = self.__get_direction_tuple(action_direction)
        target = self.player_position[0] + position_offset[0], self.player_position[1] + position_offset[1]
        is_target_valid = (0 <= target[0] < self.TOTAL_WIDTH) and (0 <= target[1] < self.RIVER_HEIGHT)

        if not is_target_valid:
            if is_walk_action and target[0] == self.LEFT_BANK_WIDTH - 1 and target[1] == -1:
                self.player_position = self.TOTAL_WIDTH - 1, 0
                is_terminal = True
                return self.observe(), reward, is_terminal
            else:
                return self.observe(), reward, is_terminal    
        
        if is_walk_action:
            if is_on_river(target) or not collides_with_boulder(target):
                self.player_position = target
            if is_on_river(target) and not collides_with_boulder(target):
                reward += self.DROWN_PENALTY
                is_terminal = True
            if target == self.GOAL_POSITION:
                is_terminal = True
        else:
            if self.player_has_boulder:
                if not collides_with_boulder(target):
                    self.boulder_positions[self.active_boulder_index] = target
                    self.active_boulder_index = None
                    self.player_has_boulder = False
            else:
                self.active_boulder_index = get_target_boulder_index(target)
                if self.active_boulder_index:
                    self.boulder_positions[self.active_boulder_index] = (-1, -1)
                    self.player_has_boulder = True

        return self.observe(), reward, is_terminal

    def observe(self):
        state = self.player_position
        for i in range(self.RIVER_WIDTH): 
            state += self.boulder_positions[i]
        # state += (1,) if self.player_has_boulder else (0,)
        return state
    
    def render(self):
        pass

    def get_illegal_mask(self):
        pass

    def __get_direction_tuple(self, direction_num):
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
        for i in range(self.n_boulders_placed, self.RIVER_WIDTH):
            self.boulder_positions[i] = generate_non_overlapping_position(i)

        # generate player position
        self.player_position = generate_non_overlapping_position(self.RIVER_WIDTH)

        self.player_has_boulder=self.start_with_boulder
        if self.player_has_boulder:
            self.active_boulder_index = -1 # last one, use one from the bank
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
        # for x in range(self.LEFT_BANK_WIDTH-1):
        #     str_representation += ' '
        # for x in range(self.RIVER_WIDTH + 2):
        #     str_representation += '='
        # str_representation += '\n'
        return str_representation



def main():
    bb = BridgeBuilding(n_boulders_placed=2)
    print(bb)

if __name__ == '__main__':
    main()