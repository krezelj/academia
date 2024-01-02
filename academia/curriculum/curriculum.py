import os
import logging
from typing import Optional

import yaml
import numpy as np

from . import LearningTask, LearningStats
from academia.agents.base import Agent
from academia.utils import SavableLoadable


_logger = logging.getLogger('academia.curriculum')


class Curriculum(SavableLoadable):
    """
    Groups and executes instances of :class:`LearningTask` in the specified order.

    Args:
        tasks: Tasks to be run. Tasks are run one by one so their order matters.
        output_dir: A path to a file where agent states and training stats will be saved upon each task's
            completion or interruption. If set to ``None``, an agent's state or training stats will not
            be saved at any point, unless relevant paths are specified for any of the tasks directly.

    Attributes:
        tasks (list[LearningTask]): Tasks to be run. Tasks are run one by one so their order matters.
        output_dir (str, optional): A path to a file where agent states and training stats will be saved upon
            each task's completion or interruption. If set to ``None``, an agent's state or training stats
            will not be saved at any point, unless relevant paths are specified for any of the tasks directly.

    Examples:
        Initialization using class contructor:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human', 'append_step_count': True},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human', 'append_step_count': True},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> curriculum = Curriculum(
        >>>     tasks=[task1, task2],
        >>>     output_dir='./my_curriculum/',
        >>> )

        Initializaton using a config file:

        >>> from academia.curriculum import Curriculum
        >>> curriculum = Curriculum.load('./my_config.curriculum.yml')

        ``./my_config.curriculum.yml``::

            output_dir: './my_curriculum/'
            order:
            - 0
            - 1
            tasks:
              0:
                env_args:
                  difficulty: 0
                  render_mode: human
                  append_step_count: True
                env_type: academia.environments.LavaCrossing
                evaluation_interval: 100
                stop_conditions:
                  max_episodes: 500
              1:
                env_args:
                  difficulty: 1
                  render_mode: human
                  append_step_count: True
                env_type: academia.environments.LavaCrossing
                evaluation_interval: 100
                stop_conditions:
                  max_episodes: 1000

        Running a curriculum:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> agent = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> curriculum.run(agent, verbose=4)
    """

    def __init__(self, tasks: list[LearningTask], output_dir: Optional[str] = None) -> None:
        self.tasks = tasks
        self.output_dir = output_dir

    def run(self, agent: Agent, verbose=0):
        """
        Runs all tasks for the given agent. Agent's states and training statistics will be saved upon each
        task's completion or interruption if save paths are specified either for a specific task, or
        for the whole curriculum through :attr:`agents_save_dir` attribute.

        Args:
            agent: An agent to train
            verbose: Verbosity level. These are common for the entire module - for information on
                different levels see :mod:`academia.curriculum`.
        """
        total_episodes = 0
        total_wall_time = 0
        total_cpu_time = 0
        for i, task in enumerate(self.tasks):
            task_id = self.__get_task_id(i)
            if verbose >= 1:
                _logger.info(f'Running Task {task_id}... ')

            if task.agent_save_path is None and self.output_dir is not None:
                task.agent_save_path = os.path.join(self.output_dir, task_id)
            if task.stats_save_path is None and self.output_dir is not None:
                task.stats_save_path = os.path.join(self.output_dir, task_id)

            task.run(agent, verbose=verbose)
            total_episodes += len(task.stats.episode_rewards)

            task_wall_time = np.sum(task.stats.episode_wall_times)
            task_cpu_time = np.sum(task.stats.episode_cpu_times)
            total_wall_time += task_wall_time
            total_cpu_time += task_cpu_time

            if verbose >= 1:
                _logger.info(f'Task {task_id} finished after '
                             f'{len(task.stats.episode_rewards)} episodes.')
                _logger.info(f'Elapsed task wall time: {task_wall_time:.2f} sec')
                _logger.info(f'Elapsed task CPU time: {task_cpu_time:.2f} sec')
                _logger.info(f'Average steps per episode: {np.mean(task.stats.step_counts):.2f}')
                _logger.info(f'Average reward per episode: {np.mean(task.stats.episode_rewards):.2f}')
        if verbose >= 1:
            _logger.info(f'Curriculum finished after {total_episodes} episodes.')
            _logger.info(f'Elapsed total wall time: {total_wall_time:.2f} sec')
            _logger.info(f'Elapsed total CPU time: {total_cpu_time:.2f} sec')

    @property
    def stats(self) -> dict[str, LearningStats]:
        """
        A dictionary that maps task name/index to task statistics for every task in this curriculum.
        """
        return {self.__get_task_id(i): task.stats for i, task in enumerate(self.tasks)}

    def __get_task_id(self, task_idx: int) -> str:
        """Task name or task's index in :attr:`tasks` if the task has no name"""
        task = self.tasks[task_idx]
        task_id = str(task_idx + 1) if task.name is None else task.name
        return task_id

    @classmethod
    def load(cls, path: str) -> 'Curriculum':
        """
        Loads a task configuration from the specified file.

        A configuration file should be in YAML format. Tasks list should be stored using two properties:
        ``tasks`` and ``order`` - the former mapping task identifiers to their configuration and the latter
        being a list of task identifiers in the order of their execution. Individual task's configurations
        can be either directly specified or a path to task's configuration file can be provided.
        Other properties names should be identical to the arguments of the :class:`Curriculum`'s constructor.

        An example curriculum configuration file::

            # my_config.curriculum.yml
            output_dir: './my_curriculum/'
            order:
            - 0
            - 1
            tasks:
              0:
                # this task's config is specified here directly:
                env_args:
                  difficulty: 0
                  render_mode: human
                env_type: academia.environments.LavaCrossing
                evaluation_interval: 100
                stop_conditions:
                  max_episodes: 500
              1:
                # this task's config lies in a separate file
                # path is relative to the location of my_config.curriculum.yml
                path: ./lava_crossing_hard.task.yml

        Args:
            path: Path to a configuration file. If the specified file does not end with '.yml' extension,
                '.curriculum.yml' will be appended to the specified path (for consistency with :func:`save()`
                method).

        Returns:
            A :class:`Curriculum` instance based on the configuration in the specified file.
        """
        # add file extension (consistency with save() method)
        if not path.endswith('.yml'):
            path += '.curriculum.yml'
        with open(path, 'r') as file:
            curriculum_data: dict = yaml.safe_load(file)
        directory = os.path.dirname(path)
        tasks = []
        for task_id in curriculum_data['order']:
            task_data: dict = curriculum_data['tasks'][task_id]
            # tasks can be stored in two ways:
            # 1. full task data (as stored in Curriculum.save)
            # 2. path to a task config file (relative from curriculum file)
            if 'path' not in task_data.keys():
                task = LearningTask.from_dict(task_data)
            else:
                task_path_abs = os.path.abspath(
                    os.path.join(directory, task_data['path'])
                )
                task = LearningTask.load(task_path_abs)
            tasks.append(task)
        del curriculum_data['order']
        del curriculum_data['tasks']
        return Curriculum(tasks, **curriculum_data)

    def save(self, path: str) -> str:
        """
        Saves this curriculum's configuration to the file.
        Configuration is stored in a YAML format.

        Args:
            path: Path where a configuration file will be created. If the extension is not provided, it will
                 will be automatically appended ('.curriculum.yml') to the specified path.

        Returns:
            A final (i.e. with an extension), absolute path where the configuration was saved.
        """
        curr_data = {
            'order': list(range(len(self.tasks))),
            # dict preserves insertion order
            'tasks': {i: task.to_dict() for i, task in enumerate(self.tasks)},
        }
        if self.output_dir is not None:
            curr_data['output_dir'] = self.output_dir
        # add file extension
        if not path.endswith('.yml'):
            path += '.curriculum.yml'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            yaml.dump(curr_data, file)
        return os.path.abspath(path)
