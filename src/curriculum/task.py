import numpy as np
from typing import Union

from src.environments import BaseEnvironment
from src.agents import Agent

class Task():

    __slots__ = ['env_type', 'env_args', 'env', 
                 'stop_condition', 'evaluation_interval', 
                 'episode_rewards', 'agent_evaluations',
                 'task_name']

    def __init__(self, env_type, env_args : dict, stop_conditions : dict, evaluation_interval : int = 100, task_name : Union[str,None] = None) -> None:
        self.env_type = env_type
        self.env_args = env_args

        # TODO assert at least one stop condition is present (i.e. dict not empty)
        self.stop_condition = stop_conditions
        self.evaluation_interval = evaluation_interval

        self.task_name = task_name

    def run_task(self, agent : Agent) -> None:
        self.__reset_task()

        episode = 0
        while not self.__is_task_finished():
            episode += 1

            episode_reward = self.__run_episode(agent)
            np.append(self.episode_rewards, episode_reward)

            if episode % self.evaluation_interval == 0:
                agent_evaluation = self.__run_episode(agent, evaluation_mode=True)
                np.append(self.agent_evaluations, agent_evaluation)

    def __run_episode(self, agent : Agent, evaluation_mode : bool = False) -> None:
        episode_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            action = agent.get_action(state, legal_mask=self.env.get_legal_mask(), greedy=evaluation_mode)
            new_state, reward, done = self.env.step(action)

            if not evaluation_mode:
                agent.update(state, action, reward, new_state, done)
                agent.decay_epsilon()

            state = new_state
            episode_reward += reward
        return episode_reward

    def __is_task_finished(self) -> bool:
        # using `if` instead of `elif` we will exit the task it *any* of the condition is true
        if 'max_episodes' in self.stop_condition:
            return len(self.episode_rewards) >= self.stop_condition['max_episodes']
        if 'predicate' in self.stop_condition:
            # custom predicate, value is a function that takes episode_rewards and agent_evaluations
            # as arguments and returns True or False deciding whether the episode should stop or not
            return self.stop_condition['predicate'](self.episode_rewards, self.agent_evaluations)

    def __reset_task(self) -> None:
        self.env : BaseEnvironment = self.environment_type(**self.env_args)
        self.episode_rewards = np.array([])
        self.agent_evaluations = np.array([])

    