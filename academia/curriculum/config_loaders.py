import os

import yaml

from . import LearningTask, Curriculum
from academia.utils import SavableLoadable


def _load_task_from_dict(task_data: dict) -> 'LearningTask':
    """
    Creates a task based on a configuration stored in a dictionary.
    This is a helper method used by config loaders and it is not useful for the end user.

    Args:
        task_data: dictionary that contains raw contents from the configuration file

    Returns:
        A :class:`LearningTask` instance based on the provided configuration.
    """
    env_type = SavableLoadable.get_type(task_data['env_type'])
    # delete env_type because it will be passed to contructor separately
    del task_data['env_type']
    return LearningTask(env_type=env_type, **task_data)


def load_task_config(path: str) -> LearningTask:
    """
    Loads a task configuration from the specified file.

    A configuration file should be in YAML format, with 'task.yml' being the preferred extension.
    Properties names should be identical to the arguments of the :class:`LearningTask`'s constructor.

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
        path: Path to a configuration file.

    Returns:
        A :class:`LearningTask` instance based on the configuration in the specified file.
    """
    with open(path, 'r') as file:
        task_data: dict = yaml.safe_load(file)
    return _load_task_from_dict(task_data)


def load_curriculum_config(path: str) -> Curriculum:
    """
    Loads a curriculum configuration from the specified file.

    A configuration file should be in YAML format, with 'curriculum.yml' being the preferred extension.
    Tasks list should be stored using two properties: ``tasks`` and ``order`` - the former mapping task
    identifiers to their configuration and the latter being a list of task identifiers in the order of
    their execution. Individual task's configurations can be either directly specified or a path to task's
    configuration file can be provided. Other properties names should be identical to the arguments of
    the :class:`Curriculum`'s constructor.

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
        path: Path to a configuration file.

    Returns:
        A :class:`Curriculum` instance based on the configuration in the specified file.
    """
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
            task = _load_task_from_dict(task_data)
        else:
            task_path_abs = os.path.abspath(
                os.path.join(directory, task_data['path'])
            )
            task = load_task_config(task_path_abs)
        tasks.append(task)
    del curriculum_data['order']
    del curriculum_data['tasks']
    return Curriculum(tasks, **curriculum_data)
