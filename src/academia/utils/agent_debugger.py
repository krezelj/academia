import logging
from pytimedinput import timedKey
from typing import Any, Optional, Union


from academia.agents.base import Agent
from academia.environments.base import ScalableEnvironment

_logger = logging.getLogger('academia.curriculum')

class AgentDebugger:

    # static keycodes for code readability
    __KEY_TERMINATE = 't'      # letter t
    __KEY_QUIT = '\x1b'        # escape
    __KEY_STEP = ' '           # space
    __KEY_PAUSE = 'p'          # letter p
    __KEY_TOGGLE_GREEDY = 'g'  # letter g

    def __init__(self, 
                 agent: Agent, 
                 env: ScalableEnvironment,
                 start_greedy: bool = False,
                 start_paused: bool = False,
                 key_action_map: dict = {},
                 run: bool = False,
                 run_verbose: int = 1) -> None:
        self.agent = agent
        self.env = env
        self.greedy = start_greedy
        self.paused = start_paused
        self.key_action_map = key_action_map
        self.input_timeout = 1_000_000_000 if self.paused else 0.05
        if run:
            self.run(run_verbose)

    def __agent_thoughts_visitor(self, agent):
        if type(agent).__qualname__ == "PPOAgent":
            print('hello')

    def run(self, verbose: int = 0):
        self.running = True
        episodes = 0
        self.__agent_thoughts_visitor(self.agent)
        while self.running:
            episodes += 1
            state = self.env.reset()
            steps = 0
            episode_reward = 0
            done = False
            while not done and self.running:
                steps += 1
                while True:
                    key, timedout = timedKey("", self.input_timeout)
                    user_action = None if timedout else self.__handle_key_press(key)
                    if not self.paused or user_action not in [self.__KEY_TOGGLE_GREEDY]:
                        break
                if user_action in [self.__KEY_TERMINATE, self.__KEY_QUIT]:
                    break
                if user_action is not None and user_action != self.__KEY_STEP:
                    action = user_action
                else:
                    action = self.agent.get_action(state, self.env.get_legal_mask(), self.greedy)

                state, reward, done = self.env.step(action)
                episode_reward += reward
                if verbose > 1:
                    _logger.info(f"Step {steps} reward: {reward}")
            if verbose > 0:
                _logger.info(f"Episode {episodes} reward: {episode_reward}")


    def __handle_key_press(self, key) -> Optional[Union[Any, int]]:
        if key is None:
            return
        if key == self.__KEY_PAUSE:
            self.paused = not self.paused
            self.input_timeout = 1_000_000_000 if self.paused else 0.05
        elif key == self.__KEY_TOGGLE_GREEDY:
            self.greedy = not self.greedy
            return key
        elif key == self.__KEY_QUIT:
            self.running = False
            return key
        elif key == self.__KEY_TERMINATE or key == self.__KEY_STEP:
            return key
        else:
            return self.key_action_map.get(key, int(key) if key.isnumeric() else None)
        return None

        
        
            

        

