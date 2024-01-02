import logging
from pytimedinput import timedKey
from typing import Any, Optional, Union

import torch
import numpy as np

from academia.agents.base import Agent
from academia.agents import *
from academia.environments.base import ScalableEnvironment

_logger = logging.getLogger('academia.curriculum')


def _ppoagent_thoughts_handler(agent: PPOAgent, state: Any) -> str:
    state = torch.unsqueeze(torch.tensor(state), dim=0).float()
    probs = agent.actor(state).tolist()[0]
    state_value = agent.critic(state).tolist()[0]
    state = state[0]
    return f"Agent Thoughts\n"\
        + f"\tAction probs.: {[np.round(p, 2) for p in probs]}\n"\
        + f"\tState value:   {np.round(state_value, 3)}\n"\
        + f"\tBest action:   {agent.get_action(state, greedy=True)}"


def _dqnagent_thoughts_handler(agent: DQNAgent, state: Any) -> str:
    qvals = agent.network(torch.tensor(state, dtype=torch.float32)).tolist()
    return f"Agent Thoughts\n"\
        + f"\tQ-values: {[np.round(q, 2) for q in qvals]}\n"\
        + f"\tBest action: {agent.get_action(state, greedy=True)}\n"\
        + f"\tExploration parameter value: {agent.epsilon}"


def _qlagent_thoughts_handler(agent: QLAgent, state: Any) -> str:
    qvals = agent.q_table[state]
    return f"Agent Thoughts\n"\
        + f"\tQ-values: {[np.round(q, 2) for q in qvals]}\n"\
        + f"\tBest action: {agent.get_action(state, greedy=True)}\n"\
        + f"\tExploration parameter value: {agent.epsilon}"


def _sarsa_thoughts_handler(agent: SarsaAgent, state: Any) -> str:
    qvals = agent.q_table[state]
    return f"Agent Thoughts\n"\
        + f"\tQ-values: {[np.round(q, 2) for q in qvals]}\n"\
        + f"\tBest action: {agent.get_action(state, greedy=True)}\n"\
        + f"\tExploration parameter value: {agent.epsilon}"


class AgentDebugger:
    """
    Class allowing for easy agent debugging. Using this class the user can
    investigate agent's behavior step-by-step with ability to check what the agent
    thinks about the current state. The user can also toggle between greedy and non-greedy
    behavior mid-episode.
    
    Additionally the user can take over the agent
    at any moment by overriding the actions taken by the agent. This allows
    the user to put the agent in new, difficult or otherwise interesting situations
    and check how the agent behaves.

    The user can interact with the debugger using the following keys:\n
    \t- 't' - terminate the current episode (and start a new one)\n
    \t- 'p' - pause the environment\n
    \t- 'g' - toggle between greedy and non-greedy behavior\n
    \t- ' ' (space) - perform one step (only works when :attr:`paused` is set to ``True``)\n
    \t- esc ('\\x1b') - quit the debugger\n
    The user can also interact with the environment using a custom :attr:`key_action_map`.

    Args:
        agent: Agent object to be debugged.
        env: Environment object with which the agent will interact. The environment should be
            instantiated with ``render_mode`` set to ``"human"`` for the user to see it.
        start_greedy: Whether the agent should start with greedy behavior. Defaults to ``False``.
        start_paused: Whether the environment should start in a paused state. Defaults to ``False``.
        key_action_map: Dictionary between keyboard keys and environment actions. 
            It accepts one character per action. If a digit character is not present 
            in the dictionary it will be automatically converted to the corresponding action. 
            If any other character is not present in the dictionary it will be converted to 
            ``None`` and ignored. The dictionary does not accept :attr:`reserved_keys` as its keys.
            Defaults to an empty dictionary.
        run: Whether to run the debugger after initialization. Defaults to ``False``.
        run_verbose: Verbosity level with which to automatically run the debugger if ``run`` is ``True``.
            Defaults to 1.

    Attributes:
        agent (Agent): Agent that is being debugged.
        env (ScalableEnvironment): Environment with which the agent interacts.
        key_action_map (dict): Dictionary between keyboard keys and environment actions.
        greedy (bool): Whether the agent behaves in a greedy manner.
        paused (bool): Whether the environment is paused (allows for step-by-step execution).
        input_timeout (float): Time (in seconds) to wait for user input. If the user does not
            press any key in that time frame the execution continues (unless :attr:`paused` is ``True``).
        episodes (int): Number of episodes run in the environment.
        steps (int): Number of steps in the current episode.
        running (bool): Whether the debugger is currently running.
        
    Examples:
        Initialization:

        >>> from academia.tools import AgentDebugger
        >>> from academia.environments import LavaCrossing
        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> 
        >>> agent = DQNAgent(lava_crossing.MLPDQN, 3)
        >>> env = LavaCrossing(difficulty=0, render_mode='human')
        >>>
        >>> # auto running with keymap example
        >>> AgentDebugger(agent, env, run=True, key_action_map={
        >>>     'w': 2,
        >>>     'a': 0,
        >>>     'd': 1,  
        >>> })
        >>>
        >>> # manual running
        >>> ad = AgentDebugger(agent, env)
        >>> ad.run(verbose=5)
    """

    # static keycodes for code readability
    __KEY_TERMINATE = 't'      # letter t
    __KEY_QUIT = '\x1b'        # escape
    __KEY_STEP = ' '           # space
    __KEY_PAUSE = 'p'          # letter p
    __KEY_TOGGLE_GREEDY = 'g'  # letter g

    reserved_keys = [__KEY_TERMINATE, __KEY_QUIT, __KEY_STEP, __KEY_PAUSE, __KEY_TOGGLE_GREEDY]
    """
    A list of reserved keys that cannot be used by :attr:`key_action_map`
    """

    thoughts_handlers = {
        'PPOAgent': _ppoagent_thoughts_handler,
        'DQNAgent': _dqnagent_thoughts_handler,
        'QLAgent': _qlagent_thoughts_handler,
        'SarsaAgent': _sarsa_thoughts_handler,
    }
    """
    A class attribute that stores global list of available agent thought handlers.
    Thought handlers are functions that accept an agent object and an observed state
    and return a user defined "thought" e.g. q-values predicted by the agent.

    These functions are stored with the following signature::

        >>> def my_thought_handler(agent: Agent, state: Any) -> str:
        >>>     pass

    where ``agent`` is the agent object to handle and ``state`` is the observed state of the environment
    on which we want to get agent's thoughts.

    There are a few default thought handler corresponding to implemented agents:

    - ``'PPOAgent'`` - returns the predicted probabilites of actions when in discrete mode and mean action
      when in continuous mode as well as the state value as predicted by the critic.
    - ``'DQNAgent'`` - returns the predicted q-values of each action.
    - ``'QLAgent'`` - returns the predicted q-values of each action.
    - ``'SarsaAgent'`` - returns the predicted q-values of each action.
    
    Example:

        >>> from academia.agents.base import Agent
        >>>
        >>> # custom agent class
        >>> class MyAgent(Agent):
        >>>     pass
        >>>
        >>> def my_agent_handler(agent: Agent, state: Any):
        >>>     pass
        >>> # adds a new handler to the dicitonary
        >>> # the key should be a string containing the name of the class
        >>> AgentDebugger.thoughts_handlers['MyAgent'] = my_agent_handler
    """

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
        for key in key_action_map.keys():
            if key in self.reserved_keys:
                raise ValueError(f"Reserved key '{key}' present in the keymap.")
        self.key_action_map = key_action_map
        self.input_timeout = 1_000_000_000 if self.paused else 0.05
        self.episodes = 0
        self.steps = 0
        self.running = False
        if run:
            self.run(run_verbose)

    def run(self, verbose: int = 0) -> None:
        """
        Runs the agent debugger with the specified verbosity level.

        +-----------------+-------------------------------------------+
        | Verbosity level | What is logged                            |
        +=================+===========================================+
        | 0               | no logging (except for errors)            |
        +-----------------+-------------------------------------------+
        | 1               | Episode Rewards                           |
        +-----------------+-------------------------------------------+
        | 2               | Step Rewards                              |
        +-----------------+-------------------------------------------+
        | 3               | Agent Thoughts                            |
        +-----------------+-------------------------------------------+

        Args:
            verbose: Verbosity level. 
        """
        self.running = True
        self.episodes = 0
        while self.running:
            self.episodes += 1
            state = self.env.reset()
            self.steps = 0
            episode_reward = 0
            done = False
            while not done and self.running:
                self.steps += 1
                if verbose > 2:
                    _logger.info(self.thoughts_handlers[type(self.agent).__qualname__](self.agent, state))
                while True:
                    key, timedout = timedKey("", self.input_timeout)
                    user_action = None if timedout else self.__handle_key_press(key)
                    if not self.paused or user_action not in [self.__KEY_TOGGLE_GREEDY]:
                        break
                if user_action in [self.__KEY_TERMINATE, self.__KEY_QUIT]:
                    break
                if user_action is not None and user_action not in [self.__KEY_STEP, self.__KEY_TOGGLE_GREEDY]:
                    action = user_action
                else:
                    action = self.agent.get_action(state, self.env.get_legal_mask(), self.greedy)

                state, reward, done = self.env.step(action)
                episode_reward += reward
                if verbose > 1:
                    _logger.info(f"Step {self.steps} reward: {reward}")
            if verbose > 0:
                _logger.info(f"Episode {self.episodes} reward: {episode_reward}")

    def __handle_key_press(self, key) -> Optional[Union[Any, int]]:
        """
        Handles the key pressed by the user by altering the state of the
        agent debugger or mapping the key to an action specified by :attr:`key_action_map`
        """
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
