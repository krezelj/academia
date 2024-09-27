import logging
from typing import Optional, Callable

import numpy as np

from . import LearningTask, LearningStats
from academia.agents.base import Agent


_logger = logging.getLogger('academia.curriculum')


class Curriculum:
    """
    Groups and executes instances of :class:`LearningTask` in the specified order.

    Args:
        tasks: Tasks to be run. Tasks are run one by one so their order matters.
        output_dir: A path to a file where agent states and training stats will be saved upon each task's
            completion or interruption. If set to ``None``, an agent's state or training stats will not
            be saved at any point, unless relevant paths are specified for any of the tasks directly.
        task_callback: A function to be called after each task is finished. It should have the
            following signature::

                >>> def my_callback(agent: Agent,
                >>>                 stats: LearningStats,
                >>>                 task_id: str,
                >>>                 ) -> Optional[Agent]:
                >>>     pass

            The parameter ``task_id`` is either the task name, or, if not specified, the order of the task's
            execution as a string ('1' for the first task, '2' for the second, and so on).
            The callback may or may not return an agent. If it does, the returned agent will be used for
            subsequent episodes.

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

        >>> from academia.curriculum import load_curriculum_config
        >>> curriculum = load_curriculum_config('./my_config.curriculum.yml')

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

    Note:
        For details on how to use configure curricula via YAML
        files please refer to :ref:`config-files`.
    """

    def __init__(self,
                 tasks: list[LearningTask],
                 output_dir: Optional[str] = None,
                 task_callback: Optional[Callable] = None,
                 ) -> None:
        self.tasks = tasks
        self.output_dir = output_dir
        self.__task_callback = task_callback
        if output_dir is not None:
            self.__ensure_tasks_savable()

    def __ensure_tasks_savable(self) -> None:
        """
        Makes sure that each task can be saved (as long as :attr:`output_dir`
        is specified for this curriculum).
        """
        for i, task in enumerate(self.tasks):
            if task.output_dir is None:
                task.output_dir = self.output_dir
            if task.name is None:
                task.name = self.__get_task_id(i)

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

            task.run(agent, verbose=verbose)
            total_episodes += len(task.stats.episode_rewards)

            task_wall_time = np.sum(task.stats.episode_wall_times)
            task_cpu_time = np.sum(task.stats.episode_cpu_times)
            total_wall_time += task_wall_time
            total_cpu_time += task_cpu_time

            if self.__task_callback is not None:
                new_agent = self.__task_callback(agent, self.stats, task_id)
                if new_agent is not None:
                    agent = new_agent

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
