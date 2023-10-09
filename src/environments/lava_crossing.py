from typing import Any, Hashable

import gymnasium

from .base import ScalableEnvironment


class LavaCrossing(ScalableEnvironment):
    """
    An environment where an agent has to go around a patch of lava to get to
    reach the destination. The higher the difficulty is, the more lava patches there are
    """

    N_ACTIONS = 3
    STATE_SIZE = NotImplemented

    _difficulty_envid_map = {
        0: 'MiniGrid-LavaCrossingS9N1-v0',
        1: 'MiniGrid-LavaCrossingS9N2-v0',
        2: 'MiniGrid-LavaCrossingS9N3-v0',
        3: 'MiniGrid-LavaCrossingS11N5-v0'
    }
    """A dictionary that maps difficulty levels to environment ids"""

    def __init__(self, difficulty: int):
        """Difficulty from 0 to 3, where 0 is the easiest and 3 is the hardest"""
        super().__init__(difficulty)
        try:
            env_id = LavaCrossing._difficulty_envid_map[difficulty]
        except KeyError:
            msg = (f"Difficulty value of {difficulty} is invalid for this environment. "
                   "Difficulty level should be an integer between 0 and 3")
            raise ValueError(msg)
        self.__base_env = gymnasium.make(env_id, render_mode="human")
        self.__state = NotImplemented
        self.reset()

    def step(self, action: int) -> tuple[Any, float, bool]:
        new_state, reward, terminated, _, _ = self.__base_env.step(action)
        self.__state = new_state
        return self.observe(), float(reward), terminated

    def observe(self) -> Any:
        return self._encode_state(self.__state)

    def get_legal_mask(self):
        pass

    def reset(self) -> Any:
        init_state = self.__base_env.reset()[0]
        return self._encode_state(init_state)

    def render(self):
        self.__base_env.render()

    @staticmethod
    def _encode_state(state) -> tuple:
        # TODO
        return str(state)
