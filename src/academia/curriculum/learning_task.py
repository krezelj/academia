import sys
from typing import Optional, Type
import os
import logging

import numpy as np
import yaml

from academia.environments.base import ScalableEnvironment
from academia.agents.base import Agent
from academia.utils import SavableLoadable, Stopwatch


_logger = logging.getLogger('academia.curriculum')


class LearningTask(SavableLoadable):

    __slots__ = ['name', 'agent_save_path', 'env_type', 'env_args', 'env',
                 'stop_conditions', 'evaluation_interval', 'evaluation_count',
                 'episode_rewards', 'agent_evaluations', 'step_counts',
                 'episode_wall_times', 'episode_cpu_times']

    def __init__(self, env_type: Type[ScalableEnvironment], env_args: dict, stop_conditions: dict,
                 evaluation_interval: int = 100, evaluation_count: int = 5,
                 name: Optional[str] = None, agent_save_path: Optional[str] = None) -> None:
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
        self.episode_wall_times = np.array([])
        self.episode_cpu_times = np.array([])

        self.name = name
        self.agent_save_path = agent_save_path

    def run(self, agent: Agent, verbose=0, render=False) -> None:
        """
        Args:
            agent (Agent): An agent to train
            verbose (int): Verbosity level.
                - 0 - no logging (except for errors);
                - 1 - Task finished/Task interrupted + warnings;
                - 2 - Mean evaluation score at each iteration;
                - 3 - Each evaluation is logged;
                - 4 - Each episode is logged.
            render (bool): Whether or not to render the environment
        """
        self.__reset()
        if render and self.env_args.get('render_mode') == 'human':
            self.env.render()
        elif render and verbose >= 1:
            _logger.warning("Cannot render environment when render_mode is not 'human'. "
                            "Consider passing render_mode in env_args in the task configuration")
        try:
            self.__train_agent(agent, verbose)
        except KeyboardInterrupt:
            if verbose >= 1:
                _logger.info('Training interrupted.')
            self.__handle_task_terminated(agent, interrupted=True)
            sys.exit(130)
        except Exception as e:
            if verbose >= 1:
                _logger.info('Training interrupted.')
            _logger.exception(e)
            self.__handle_task_terminated(agent, interrupted=True)
            sys.exit(1)
        else:
            if verbose >= 1:
                _logger.info('Training finished.')
            self.__handle_task_terminated(agent)

    def __train_agent(self, agent: Agent, verbose=0) -> None:
        episode = 0
        while not self.__is_finished():
            episode += 1

            stopwatch = Stopwatch()
            episode_reward, steps_count = self.__run_episode(agent)
            wall_time, cpu_time = stopwatch.stop()
            self.__update_statistics(episode, episode_reward, steps_count, wall_time, cpu_time, verbose)

            if episode % self.evaluation_interval == 0:
                evaluation_rewards: list[float] = []
                for _ in range(self.evaluation_count):
                    evaluation_reward, _ = self.__run_episode(agent, evaluation_mode=True)
                    evaluation_rewards.append(evaluation_reward)
                self.agent_evaluations = np.append(self.agent_evaluations,
                                                   np.mean(evaluation_rewards))

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

    def __update_statistics(self, episode_no: int, episode_reward: float, steps_count: int,
                            wall_time: float, cpu_time: float, verbose=0) -> None:
        self.episode_wall_times = np.append(self.episode_wall_times, wall_time)
        self.episode_cpu_times = np.append(self.episode_cpu_times, cpu_time)

        self.episode_rewards = np.append(self.episode_rewards, episode_reward)
        self.step_counts = np.append(self.step_counts, steps_count)

        episode_rewards_mvavg = np.mean(self.episode_rewards[-5:])
        steps_count_mvavg = np.mean(self.step_counts[-5:])
        self.episode_rewards_moving_avg = np.append(
            self.episode_rewards_moving_avg, episode_rewards_mvavg)
        self.step_counts_moving_avg = np.append(
            self.step_counts_moving_avg, steps_count_mvavg)

        if verbose >= 4:
            _logger.info(f'Episode {episode_no} done.')
            _logger.info(f'Elapsed task wall time: {wall_time:.3f} sec')
            _logger.info(f'Elapsed task CPU time: {cpu_time:.3f} sec')
            _logger.info(f'Reward: {episode_reward:.2f}')
            _logger.info(f'Moving average of rewards: {episode_rewards_mvavg:.2f}')
            _logger.info(f'Steps count: {steps_count}')
            _logger.info(f'Moving average of step counts: {steps_count_mvavg:.1f}')

    def __handle_task_terminated(self, agent: Agent, interrupted=False, verbose=0) -> None:
        if self.agent_save_path is not None:
            dirname, filename = os.path.split(self.agent_save_path)
            if interrupted:
                # prefix to let user know that the training has not been completed
                filename = f'backup_{filename}'
            os.makedirs(dirname, exist_ok=True)
            full_path = os.path.join(dirname, filename)
            if verbose >= 1:
                _logger.info("Saving agent's state...")
            save_path = agent.save(full_path)
            if verbose >= 1:
                _logger.info(f"Agent's state saved to {save_path}")

    def __is_finished(self) -> bool:
        # using `if` instead of `elif` we will exit the task it *any* of the condition is true
        if 'max_episodes' in self.stop_conditions:
            return len(self.episode_rewards) >= self.stop_conditions['max_episodes']
        
        if 'max_steps' in self.stop_conditions:
            return np.sum(self.step_counts) >= self.stop_conditions['max_steps']
        
        if 'min_avg_reward' in self.stop_conditions and len(self.episode_rewards_moving_avg) > 5:
            # check if episode_rewards_moving_avg length is greater than because if not it is possibility
            # that agent scored max reward in first episode
            # and then it will stop training because it will think that it has reached min_avg_reward
            return self.episode_rewards_moving_avg[-1] >= self.stop_conditions['min_avg_reward']
        
        if 'min_reward_std_dev' in self.stop_conditions and len(self.episode_rewards) > 10:
            return np.std(self.episode_rewards[-10:]) <= self.stop_conditions['min_reward_std_dev']

        if 'evaluation_score' in self.stop_conditions:
            return np.mean(self.agent_evaluations[-5:]) >= self.stop_conditions['evaluation_score']
        
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

    def save(self, path: str) -> str:
        task_data = self.to_dict()
        # add file extension
        if not path.endswith('.yml'):
            path += '.task.yml'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            yaml.dump(task_data, file)
        return os.path.abspath(path)

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
