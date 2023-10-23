from typing import Optional, Type
import os
import logging

import numpy as np
import yaml

from academia.environments.base import ScalableEnvironment
from academia.agents.base import Agent
from academia.utils import SavableLoadable


_logger = logging.getLogger('academia.curriculum')


class LearningTask(SavableLoadable):

    __slots__ = ['name', 'env_type', 'env_args', 'env',
                 'stop_conditions', 'evaluation_interval', 'evaluation_count',
                 'episode_rewards', 'agent_evaluations', 'step_counts']

    def __init__(self, env_type: Type[ScalableEnvironment], env_args: dict, stop_conditions: dict,
                 evaluation_interval: int = 100, evaluation_count: int = 5,
                 name: Optional[str] = None) -> None:
        self.env_type = env_type
        self.env_args = env_args

        if len(stop_conditions) == 0:
            msg = ('stop_conditions dict must not be empty. '
                   'Please provide at least one stop condition.')
            _logger.error(msg)
            raise ValueError(msg)
        self.stop_conditions = stop_conditions
        self.evaluation_interval = evaluation_interval
        self.evaluation_count = evaluation_count

        self.agent_evaluations = np.array([])
        self.episode_rewards = np.array([])
        self.step_counts = np.array([])
        self.episode_rewards_moving_avg = np.array([])
        self.step_counts_moving_avg = np.array([])

        self.name = name

    def run(self, agent: Agent, verbose=0, render=False) -> None:
        self.__reset()
        if render and self.env_args.get('render_mode') == 'human':
            self.env.render()
        elif render:
            _logger.warning("WARNING: Cannot render environment when render_mode is not 'human'. "
                            "Consider passing render_mode in env_args in the task configuration")
        episode = 0
        while not self.__is_finished():
            episode += 1

            episode_reward, steps_count = self.__run_episode(agent)

            self.episode_rewards = np.append(self.episode_rewards, episode_reward)
            self.step_counts = np.append(self.step_counts, steps_count)

            episode_rewards_mvavg = np.mean(self.episode_rewards[-5:])
            steps_count_mvavg = np.mean(self.step_counts[-5:])
            self.episode_rewards_moving_avg = np.append(
                self.episode_rewards, episode_rewards_mvavg)
            self.step_counts_moving_avg = np.append(
                self.step_counts, steps_count_mvavg)

            if verbose >= 2:
                _logger.info(f'Episode {episode} done.')
                _logger.info(f'Reward: {episode_reward:.2f}')
                _logger.info(f'Moving average of rewards: {episode_rewards_mvavg:.2f}')
                _logger.info(f'Steps count: {steps_count}')
                _logger.info(f'Moving average of step counts: {steps_count_mvavg:.1f}')

            if episode % self.evaluation_interval == 0:
                eval_rewards: list[float] = []
                for _ in range(self.evaluation_count):
                    eval_reward, _ = self.__run_episode(agent, evaluation_mode=True)
                    eval_rewards.append(eval_reward)
                self.agent_evaluations = np.append(self.agent_evaluations, np.mean(eval_rewards))

    def __run_episode(self, agent: Agent, evaluation_mode: bool = False) -> tuple[float, int]:
        """
        :return: episode reward and total number of steps
        """
        episode_reward = 0
        steps_count = 0
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
            steps_count += 1
        return episode_reward, steps_count

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
        self.step_counts = np.array([])

    @classmethod
    def load(cls, path: str) -> 'LearningTask':
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
    def from_dict(cls, task_data: dict) -> 'LearningTask':
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
