from typing import Optional, Type
import os

import numpy as np
import yaml

from academia.environments.base import ScalableEnvironment
from academia.agents.base import Agent
from academia.utils import SavableLoadable


# TODO Add docstrings to all methods
# TODO Decide whether to pass env_type and env_args or an already instantiated environemnt

class Task(SavableLoadable):

    __slots__ = ['env_type', 'env_args', 'env',
                 'stop_conditions', 'evaluation_interval',
                 'episode_rewards', 'agent_evaluations',
                 'name']

    def __init__(self, env_type: Type[ScalableEnvironment], env_args: dict, stop_conditions: dict,
                 evaluation_interval: int = 100, name: Optional[str] = None) -> None:
        self.env_type = env_type
        self.env_args = env_args

        # TODO assert at least one stop condition is present (i.e. dict not empty)
        self.stop_conditions = stop_conditions
        self.evaluation_interval = evaluation_interval

        self.name = name

    def run(self, agent: Agent) -> None:
        self.__reset()

        episode = 0
        while not self.__is_finished():
            episode += 1

            episode_reward = self.__run_episode(agent)
            np.append(self.episode_rewards, episode_reward)

            if episode % self.evaluation_interval == 0:
                agent_evaluation = self.__run_episode(agent, evaluation_mode=True)
                np.append(self.agent_evaluations, agent_evaluation)

    def __run_episode(self, agent: Agent, evaluation_mode: bool = False) -> float:
        episode_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            action = agent.get_action(state, legal_mask=self.env.get_legal_mask(),
                                      greedy=evaluation_mode)
            new_state, reward, done = self.env.step(action)

            if not evaluation_mode:
                agent.update(state, action, reward, new_state, done)
                agent.decay_epsilon()

            state = new_state
            episode_reward += reward
        return episode_reward

    def __is_finished(self) -> bool:
        # using `if` instead of `elif` we will exit the task it *any* of the condition is true
        if 'max_episodes' in self.stop_conditions:
            return len(self.episode_rewards) >= self.stop_conditions['max_episodes']
        if 'predicate' in self.stop_conditions:
            # custom predicate, value is a function that takes episode_rewards and agent_evaluations
            # as arguments and returns True or False deciding whether the episode should stop or not
            return self.stop_conditions['predicate'](self.episode_rewards, self.agent_evaluations)

    def __reset(self) -> None:
        self.env: ScalableEnvironment = self.env_type(**self.env_args)
        self.episode_rewards = np.array([])
        self.agent_evaluations = np.array([])

    @classmethod
    def load(cls, path: str) -> 'Task':
        # add file extension (consistency with save() method)
        if not path.endswith('.yml'):
            path += '.task.yml'
        with open(path, 'r') as file:
            task_data: dict = yaml.safe_load(file)
        return cls.from_dict(task_data)

    def save(self, path: str) -> None:
        task_data = self.to_dict()
        # add file extension
        if not path.endswith('.yml'):
            path += '.task.yml'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            yaml.dump(task_data, file)

    @classmethod
    def from_dict(cls, task_data: dict) -> 'Task':
        env_type = cls.get_type(task_data['env_type'])
        # delete env_type because it will be passed to contructor separately
        del task_data['env_type']
        return cls(env_type=env_type, **task_data)

    def to_dict(self) -> dict:
        task_data = {
            'env_type': self.get_type_name_full(self.env_type),
            'env_args': self.env_args,
            'stop_conditions': self.stop_conditions,
            'evaluation_interval': self.evaluation_interval,
        }
        if self.name is not None:
            task_data['name'] = self.name
        return task_data

