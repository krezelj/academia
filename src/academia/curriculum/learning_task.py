import sys
from typing import Optional, Type
import os
import logging
import json

import numpy as np
import yaml

from academia.environments.base import ScalableEnvironment
from academia.agents.base import Agent
from academia.utils import SavableLoadable, Stopwatch


_logger = logging.getLogger('academia.curriculum')


class LearningTask(SavableLoadable):
    """
    Controls agent's training.

    Args:
        env_type: A subclass of :class:`academia.environments.base.ScalableEnvironment` that the agent will
            be trained on. This should be a class, not an instantiated object.
        env_args: Arguments passed to the constructor of the environment class (passed as``env_type``
            argument).
        stop_conditions: Conditions deciding when to end the training process. Available conditions:
            ``'max_episodes'``, ``'max_steps'``, ``'min_avg_reward'``, ``'min_reward_std_dev'``,
            ``'evaluation_score'``.
        evaluation_interval: Controls how often evaluations are conducted. Defaults to 100.
        evaluation_count: Controls how many evaluation episodes are run during a single evaluation.
            Final agent evaluation will be the mean of these individual evaluations. Defaults to 5.
        name: Name of the task. This is unused when running a single task on its own.
            Hovewer, if specified it will appear in the logs and (optionally) in some file names if the
            task is run through the :class:`Curriculum` object.
        agent_save_path: A path to a file where the agent's state will be saved after the training is
            completed or if it is interrupted. If not set, agent's state will not be saved at any point.
        stats_save_path: A path to a file where the statistics gathered during training process will be
            saved after the training is completed or if it is interrupted. If not set, agent's state will
            not be saved at any point.

    Raises:
        ValueError: If `stop_conditions` is empty

    Attributes:
        env (ScalableEnvironment): An environment that an agent can interact with.
            It is of a type ``env_type``, initialised with parameters from ``env_args``.
        stats (LearningStats): Learning statistics. For more detailed description of their contents see
            :class:`LearningStats`.
        env_type (Type[ScalableEnvironment]): A subclass of
            :class:`academia.environments.base.ScalableEnvironment` that the agent will
            be trained on. This should be a class, not an instantiated object.
        env_args (dict): Arguments passed to the constructor of the environment class (passed as ``env_type``
            argument).
        stop_conditions (dict): Conditions deciding when to end the training process. Available conditions:
            ``'max_episodes'``, ``'max_steps'``, ``'min_avg_reward'``, ``'min_reward_std_dev'``,
            ``'evaluation_score'``.
        evaluation_interval (int): Controls how often evaluations are conducted.
        evaluation_count (int): Controls how many evaluation episodes are run during a single evaluation.
            Final agent evaluation will be the mean of these individual evaluations.
        name (str, optional): Name of the task. This is unused when running a single task
            on its own. Hovewer, if specified it will appear in the logs and (optionally) in some file names
            if the task is run through the :class:`Curriculum` object.
        agent_save_path (str, optional): A path to a file where the agent's state will be saved after the
            training is completed or if it is interrupted. If set to ``None``, agent's state will not be
            saved at any point.
        stats_save_path (str, optional): A path to a file where the statistics gathered during training
            process will be saved after the training is completed or if it is interrupted. If set to
            ``None``, agent's state will not be saved at any point.

    Examples:
        Initialisation using class contructor:

        >>> from academia.curriculum import LearningTask
        >>> from academia.environments import LavaCrossing
        >>> task = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 2, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>>     stats_save_path='./my_task_stats.json',
        >>> )

        Initialisaton using a config file:

        >>> from academia.curriculum import LearningTask
        >>> task = LearningTask.load('./my_config.task.yml')

        ``./my_config.task.yml``::

            env_type: academia.environments.LavaCrossing
            env_args:
                difficulty: 2
                render_mode: human
            stop_conditions:
                max_episodes: 1000
            stats_save_path: ./my_task_stats.json

        Running a task:

        >>> from academia.agents import DQNAgent
        >>> from academia.models import LavaCrossingMLP
        >>> agent = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> task.run(agent, verbose=4, render=True)
    """

    def __init__(self, env_type: Type[ScalableEnvironment], env_args: dict, stop_conditions: dict,
                 evaluation_interval: int = 100, evaluation_count: int = 5,
                 name: Optional[str] = None, agent_save_path: Optional[str] = None,
                 stats_save_path: Optional[str] = None) -> None:
        self.env_type = env_type
        self.env_args = env_args

        if len(stop_conditions) == 0:
            msg = ('stop_conditions dict must not be empty. '
                   'Please provide at least one stop condition.')
            _logger.error(msg)
            raise ValueError(msg)
        # temporary solution (see issue #67)
        available_stop_conditions = {'max_episodes', 'max_steps', 'min_avg_reward',
                                     'min_reward_std_dev', 'evaluation_score'}
        for sc in stop_conditions.keys():
            if sc not in available_stop_conditions:
                _logger.warning(f'"{sc}" is not a known stop condition. '
                                f'Available stop conditions: {available_stop_conditions}')

        self.stop_conditions = stop_conditions
        self.evaluation_interval = evaluation_interval
        self.evaluation_count = evaluation_count

        self.stats = LearningStats()

        self.name = name
        self.agent_save_path = agent_save_path
        self.stats_save_path = stats_save_path

    def run(self, agent: Agent, verbose=0, render=False) -> None:
        """
        Runs the training loop for the given agent on an environment specified during this task's
        initialisation. Training statistics will be saved to a JSON file if
        :attr:`stats_save_path` is not ``None``.

        Args:
            agent: An agent to train
            verbose: Verbosity level. Possible values: 0 - no logging (except for errors);
                1 - Task finished/Task interrupted + warnings; 2 - Mean evaluation score at each iteration;
                3 - Each evaluation is logged; 4 - Each episode is logged.
            render: Whether or not to render the environment
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
            self.__handle_task_terminated(agent, verbose, interrupted=True)
            sys.exit(130)
        except Exception as e:
            if verbose >= 1:
                _logger.info('Training interrupted.')
            _logger.exception(e)
            self.__handle_task_terminated(agent, verbose, interrupted=True)
            sys.exit(1)
        else:
            if verbose >= 1:
                _logger.info('Training finished.')
            self.__handle_task_terminated(agent, verbose)

    def __train_agent(self, agent: Agent, verbose=0) -> None:
        """
        Runs a training loop on a given agent until one or more stop conditions are met.
        """
        episode = 0
        while not self.__is_finished():
            episode += 1

            stopwatch = Stopwatch()
            episode_reward, steps_count = self.__run_episode(agent)
            wall_time, cpu_time = stopwatch.stop()
            self.stats.update(episode, episode_reward, steps_count, wall_time, cpu_time, verbose)

            if episode % self.evaluation_interval == 0:
                evaluation_rewards: list[float] = []
                for evaluation_no in range(self.evaluation_count):
                    evaluation_reward, _ = self.__run_episode(agent, evaluation_mode=True)
                    evaluation_rewards.append(evaluation_reward)
                    if verbose >= 3:
                        _logger.info(f'Evaluation {evaluation_no} after episode {episode}. '
                                     f'Reward: {evaluation_reward}')
                mean_evaluation = np.mean(evaluation_rewards)
                if verbose >= 2:
                    _logger.info(f'Evaluations after episode {episode} completed. '
                                 f'Mean reward: {mean_evaluation}')
                self.stats.agent_evaluations = np.append(
                    self.stats.agent_evaluations, mean_evaluation)

    def __run_episode(self, agent: Agent, evaluation_mode: bool = False) -> tuple[float, int]:
        """
        Runs a single episode on a given agent.

        Args:
            evaluation_mode: Whether or not to run the episode in an evaluation mode, i.e. in a greedy way,
                without updating agent's knowledge or epsilon.

        Returns:
            episode reward and total number of steps
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

    def __handle_task_terminated(self, agent: Agent, verbose: int, interrupted=False) -> None:
        """
        Saves most recent agent's state and training statistics (if relevant paths were specified during
        :class:`LearningTask` initialisation.

        Args:
            interrupted: Whether or not the task has been interrupted or has finished normally
        """
        # preserve agent's state
        if self.agent_save_path is not None:
            agent_save_path = self._prep_save_file(self.agent_save_path, interrupted)
            if verbose >= 1:
                _logger.info("Saving agent's state...")
            final_save_path = agent.save(agent_save_path)
            if verbose >= 1:
                _logger.info(f"Agent's state saved to {final_save_path}")

        # save task statistics
        if self.stats_save_path is not None:
            stats_save_path = self._prep_save_file(self.stats_save_path, interrupted)
            if verbose >= 1:
                _logger.info("Saving task's stats...")
            self.stats.save(stats_save_path)
            if verbose >= 1:
                _logger.info(f"Task's stats saved to {stats_save_path}")

    def __is_finished(self) -> bool:
        """
        Checks whether any of the stop conditions is met.

        Returns:
            ``True`` if the task should be terminated or ``False`` otherwise
        """
        # using `if` instead of `elif` we will exit the task it *any* of the condition is true
        if 'max_episodes' in self.stop_conditions:
            return len(self.stats.episode_rewards) >= self.stop_conditions['max_episodes']
        
        if 'max_steps' in self.stop_conditions:
            return np.sum(self.stats.step_counts) >= self.stop_conditions['max_steps']
        
        if 'min_avg_reward' in self.stop_conditions and len(self.stats.episode_rewards_moving_avg) > 5:
            # check if episode_rewards_moving_avg length is greater than because if not it is possibility
            # that agent scored max reward in first episode
            # and then it will stop training because it will think that it has reached min_avg_reward
            return self.stats.episode_rewards_moving_avg[-1] >= self.stop_conditions['min_avg_reward']
        
        if 'min_reward_std_dev' in self.stop_conditions and len(self.stats.episode_rewards) > 10:
            return np.std(self.stats.episode_rewards[-10:]) <= self.stop_conditions['min_reward_std_dev']

        if 'evaluation_score' in self.stop_conditions:
            return np.mean(self.stats.agent_evaluations[-5:]) >= self.stop_conditions['evaluation_score']
        
    def __reset(self) -> None:
        """
        Resets environment and statistics.
        """
        self.env: ScalableEnvironment = self.env_type(**self.env_args)
        self.stats = LearningStats()

    @classmethod
    def load(cls, path: str) -> 'LearningTask':
        """
        Loads a task configuration from the specified file.

        A configuration file should be in YAML format. Properties names should be identical to the arguments
        of the :class:`LearningTask` constructor.

        An example task configuration file::

            # my_config.task.yml
            env_type: academia.environments.LavaCrossing
            env_args:
                difficulty: 2
                render_mode: human
            stop_conditions:
                max_episodes: 1000
            stats_save_path: ./my_task_stats.json

        Args:
            path: Path to a configuration file. If the specified file does not end with '.yml' extension,
                '.task.yml' will be appended to the specified path (for consistency with :func:`save()`
                method).

        Returns:
            A :class:`LearningTask` instance based on the configuration in the specified file.
        """
        # add file extension (consistency with save() method)
        if not path.endswith('.yml'):
            path += '.task.yml'
        with open(path, 'r') as file:
            task_data: dict = yaml.safe_load(file)
        return cls.from_dict(task_data)

    def save(self, path: str) -> str:
        """
        Saves this task's configuration to a file.
        Configuration is stored in a YAML format.

        Args:
            path: Path where a configuration file will be created. If the extension is not provided, it will
                 will be automatically appended ('.task.yml') to the specified path.

        Returns:
            A final (i.e. with an extension), absolute path where the configuration was saved.
        """
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
        """
        Creates a task based on a configuration stored in a dictionary.
        This is a helper method used by the :class:`Curriculum` class and it is not useful for the end user.

        Args:
            task_data: dictionary that contains raw contents from the configuration file

        Returns:
            A :class:`LearningTask` instance based on the provided configuration.
        """
        env_type = cls.get_type(task_data['env_type'])
        # delete env_type because it will be passed to contructor separately
        del task_data['env_type']
        return cls(env_type=env_type, **task_data)

    def to_dict(self) -> dict:
        """
        Puts this :class:`LearningTask`'s configuration to a dictionary.
        This is a helper method used by the :class:`Curriculum` class and it is not useful for the end user.

        Returns:
            A dictionary with the task configuration, ready to be written to a text file.
        """
        task_data = {
            'env_type': self.get_type_name_full(self.env_type),
            'env_args': self.env_args,
            'stop_conditions': self.stop_conditions,
            'evaluation_interval': self.evaluation_interval,
        }
        if self.name is not None:
            task_data['name'] = self.name
        return task_data


class LearningStats(SavableLoadable):
    """
    Container for training statistics from LearningTask

    Attributes:
        episode_rewards (numpy.ndarray): An array of floats which stores total rewards for each
            episode (excluding evaluations).
        agent_evaluations (numpy.ndarray): An array of floats which stores total rewards for each
            evaluation.
        step_counts (numpy.ndarray): An array of integers which stores step counts for each episode
            (excluding evaluations).
        episode_rewards_moving_avg (numpy.ndarray): An array of floats which stores moving averages
            of total rewards for each episode (excluding evaluations). Each average is calculated from 5
            observations.
        step_counts_moving_avg (numpy.ndarray): An array of floats which stores moving averages
            of step counts for each episode (excluding evaluations). Each average is calculated from 5
            observations.
        episode_wall_times (numpy.ndarray): An array of floats which stores elapsed wall times for
            each episode (excluding evaluations).
        episode_cpu_times (numpy.ndarray): An array of floats which stores elapsed CPU times for
            each episode (excluding evaluations).
    """

    def __init__(self):
        self.agent_evaluations = np.array([])
        self.episode_rewards = np.array([])
        self.step_counts = np.array([])
        self.episode_rewards_moving_avg = np.array([])
        self.step_counts_moving_avg = np.array([])
        self.episode_wall_times = np.array([])
        self.episode_cpu_times = np.array([])

    def update(self, episode_no: int, episode_reward: float, steps_count: int, wall_time: float,
               cpu_time: float, verbose: int = 0) -> None:
        """
        Updates and logs training statistics for a given episode

        Args:
            episode_no: Episode number
            episode_reward: Total reward after the episode
            steps_count: Steps count of the episode
            wall_time: Actual time it took for the episode to finish
            cpu_time: CPU time it took for the episode to finish
            verbose: Verbosity level. See :func:`LearningTask.run` for information on different verboisty
                levels
        """
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

    @classmethod
    def load(cls, path: str):
        """
        Loads learning statistics from the specified file.

        Specified file should be in JSON format. Example file::

            {
                "episode_rewards": [1, 0, 0, 1],
                "step_counts": [250, 250, 250, 250],
                "episode_rewards_moving_avg": [1, 0.5, 0.33, 0.5],
                "step_counts_moving_avg": [250, 250, 250, 250],
                "agent_evaluations": [0, 0],
                "episode_wall_times": [
                    0.5392518779990496,
                    0.5948321321360364,
                    0.6083159360059653,
                    0.5948852870060364
                ],
                "episode_cpu_times": [
                    2.1462997890000004,
                    2.3829500180000007,
                    2.4324373569999995,
                    2.3217381230000001
                ]
            }

        Args:
            path: Path to a stats file. If the specified file does not end with '.stats.json' extension,
                this extension will be appended to the specified path (for consistency with :func:`save()`
                method).

        Returns:
            A :class:`LearningStats` instance with statistics from the specified file.
        """
        if not path.endswith('.stats.json'):
            path += '.stats.json'
        with open(path, 'r') as file:
            stats_dict = json.load(file)
        stats_obj = cls()
        stats_obj.episode_rewards = np.array(stats_dict['episode_rewards'])
        stats_obj.step_counts = np.array(stats_dict['step_counts'])
        stats_obj.episode_rewards_moving_avg = np.array(stats_dict['episode_rewards_moving_avg'])
        stats_obj.step_counts_moving_avg = np.array(stats_dict['step_counts_moving_avg'])
        stats_obj.agent_evaluations = np.array(stats_dict['agent_evaluations'])
        stats_obj.episode_wall_times = np.array(stats_dict['episode_wall_times'])
        stats_obj.episode_cpu_times = np.array(stats_dict['episode_cpu_times'])
        return stats_obj

    def save(self, path: str) -> str:
        """
        Saves this :class:`LearningStats`'s contents to a file. Stats are stored in JSON format.

        Args:
            path: Path where a statistics file will be created. If the extension is not provided, it will
                 will be automatically appended ('.stats.json') to the specified path.

        Returns:
            A final (i.e. with an extension), absolute path where the configuration was saved.
        """
        if not path.endswith('.stats.json'):
            path += '.stats.json'
        with open(path, 'w') as file:
            data = {
                'episode_rewards': self.episode_rewards.tolist(),
                'step_counts': self.step_counts.tolist(),
                'episode_rewards_moving_avg': self.episode_rewards_moving_avg.tolist(),
                'step_counts_moving_avg': self.step_counts_moving_avg.tolist(),
                'agent_evaluations': self.agent_evaluations.tolist(),
                'episode_wall_times': self.episode_wall_times.tolist(),
                'episode_cpu_times': self.episode_cpu_times.tolist(),
            }
            json.dump(data, file, indent=4)
        return path
