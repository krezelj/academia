import os
import logging
from typing import Optional

import yaml
import numpy as np

from . import LearningTask
from academia.agents.base import Agent
from academia.utils import SavableLoadable


_logger = logging.getLogger('academia.curriculum')


class Curriculum(SavableLoadable):
    """
    Groups and executes instances of :class:`academia.curriculum.LearningTask` in the specified order.

    Args:
        tasks: Tasks to be run. Tasks are run one by one so their order matters.
        agents_save_dir: A path to a file where the agent states and training stats will be saved upon each
            task's completion or interruption. If set to ``None``, agent's state or training stats will not
            be saved at any point, unless relevant paths are specified for any of the tasks directly.

    Attributes:
        tasks: Tasks to be run. Tasks are run one by one so their order matters.
        agents_save_dir: A path to a file where the agent states and training stats will be saved upon each
            task's completion or interruption. If set to ``None``, agent's state or training stats will not
            be saved at any point, unless relevant paths are specified for any of the tasks directly.

    Examples:
        Initialisation using class contructor:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> curriculum = Curriculum(
        >>>     tasks=[task1, task2],
        >>>     agents_save_dir='./my_curriculum/',
        >>> )

        Initialisaton using a config file:

        >>> from academia.curriculum import Curriculum
        >>> curriculum = Curriculum.load('./my_config.curriculum.yml')

        ``./my_config.curriculum.yml``::

            agents_save_dir: './my_curriculum/'
            order:
            - 0
            - 1
            tasks:
              0:
                env_args:
                  difficulty: 0
                  render_mode: human
                env_type: academia.environments.LavaCrossing
                evaluation_interval: 100
                stop_conditions:
                  max_episodes: 500
              1:
                env_args:
                  difficulty: 1
                  render_mode: human
                env_type: academia.environments.LavaCrossing
                evaluation_interval: 100
                stop_conditions:
                  max_episodes: 1000

        Running a curriculum:

        >>> from academia.agents import DQNAgent
        >>> from academia.models import LavaCrossingMLP
        >>> agent = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> curriculum.run(agent, verbose=4, render=True)
    """

    __slots__ = ['tasks', 'agents_save_dir']

    def __init__(self, tasks: list[LearningTask], agents_save_dir: Optional[str] = None) -> None:
        self.tasks = tasks
        self.agents_save_dir = agents_save_dir

    def run(self, agent: Agent, verbose=0, render=False):
        """
        Runs all tasks for the given agent. Agent's states and training statistics will be saved upon each
            task's completion or interruption if save paths are specified either for a specific task, or
            for the whole curriculum through ``agents_save_dir`` attribute.

        Args:
            agent: An agent to train
            verbose: Verbosity level. Possible values: 0 - no logging (except for errors);
                1 - Task finished/Task interrupted + warnings; 2 - Mean evaluation score at each iteration;
                3 - Each evaluation is logged; 4 - Each episode is logged.
            render: Whether or not to render the environment
        """
        total_episodes = 0
        total_wall_time = 0
        total_cpu_time = 0
        for i, task in enumerate(self.tasks):
            task_id = str(i + 1) if task.name is None else task.name
            if verbose >= 1:
                _logger.info(f'Running Task {task_id}... ')

            if task.agent_save_path is None and self.agents_save_dir is not None:
                task.agent_save_path = os.path.join(self.agents_save_dir, task_id)
            if task.stats_save_path is None and self.agents_save_dir is not None:
                task.stats_save_path = os.path.join(self.agents_save_dir, f'{task_id}_stats')

            task.run(agent, verbose=verbose, render=render)
            total_episodes += len(task.episode_rewards)

            task_wall_time = np.sum(task.episode_wall_times)
            task_cpu_time = np.sum(task.episode_cpu_times)
            total_wall_time += task_wall_time
            total_cpu_time += task_cpu_time

            if verbose >= 1:
                _logger.info(f'Task {task_id} finished after '
                             f'{len(task.episode_rewards)} episodes.')
                _logger.info(f'Elapsed task wall time: {task_wall_time:.2f} sec')
                _logger.info(f'Elapsed task CPU time: {task_cpu_time:.2f} sec')
                _logger.info(f'Average steps per episode: {np.mean(task.step_counts):.2f}')
                _logger.info(f'Average reward per episode: {np.mean(task.episode_rewards):.2f}')
        if verbose >= 1:
            _logger.info(f'Curriculum finished after {total_episodes} episodes.')
            _logger.info(f'Elapsed total wall time: {total_wall_time:.2f} sec')
            _logger.info(f'Elapsed total CPU time: {total_cpu_time:.2f} sec')

    @classmethod
    def load(cls, path: str) -> 'Curriculum':
        """
        Loads a task configuration from the specified file.

        A configuration file should be in YAML format. Tasks list should be stored using two properties:
        ``tasks`` and ``order`` - the former mapping task ids to their configuration and the latter being a
        list of task ids in the order of their execution. Individual task's configurations can be either
        directly specified or a path to task's configuration file can be provided.
        Other properties names should be identical to the arguments of the :class:`Curriculum`'s constructor.

        An example curriculum configuration file::

            # my_config.curriculum.yaml
            agents_save_dir: './my_curriculum/'
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
                # path is relative to the location of my_config.curriculum.yaml
                path: ./lava_crossing_hard.task.yaml

        Args:
            path: Path to a configuration file. If the specified file does not end with '.yml' extension,
                '.curriculum.yml' will be appended to the specified path (for consistency with ``save()``
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
        Saves this :class:`Curriculum`'s configuration to the file.
        Configuration is stored in a YAML format.

        Args:
            path: Path where a configuration file will be created. If the extension is not provided, it will
                 will be automatically appended ('.curriculum.yml') to the specified path.

        Returns:
            A final (i.e. with an extension), absolute path where the configuration was saved.
        """
        # dict preserves insertion order
        curr_data = {
            'order': list(range(len(self.tasks))),
            'tasks': {i: task.to_dict() for i, task in enumerate(self.tasks)},
        }
        # add file extension
        if not path.endswith('.yml'):
            path += '.curriculum.yml'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            yaml.dump(curr_data, file)
        return os.path.abspath(path)
