import os
from typing import Optional
import logging

import yaml

from . import LearningTask, Curriculum
from academia.utils import SavableLoadable


VARIABLE_PREFIX = '$'
DEFAULT_ATTR_NAME = '_default'
LOAD_ATTR_NAME = '_load'

_logger = logging.getLogger('academia.curriculum')


def __handle_overrides(default_config: dict, overriding_config: dict) -> dict:
    """
    A helper function which handles overrides to merge two configs

    Args:
        default_config: A configuration with defaults
        overriding_config: A configuration that extends the ``default_config``
            and may override some of its attributes

    Returns:
        A merged config
    """
    merged_data = default_config.copy()
    for key, value in overriding_config.items():
        # merge nested values
        if isinstance(value, dict) and key in default_config.keys():
            merged_data[key] = __handle_overrides(
                default_config=default_config[key],
                overriding_config=value,
            )
        else:
            merged_data[key] = value
    return merged_data


def __inject_variables(config: dict, variables: dict) -> dict:
    """
    Substitute variables with their values in the specified config.

    Args:
        config: Config with variable references
        variables: Variables values

    Returns:
        A config with variable references substituted with their values
    """
    new_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            new_config[key] = __inject_variables(config[key], variables)
        elif isinstance(value, str) and value.startswith(VARIABLE_PREFIX):
            var_name = value.removeprefix(VARIABLE_PREFIX)  # skip the dollar sign
            try:
                new_config[key] = variables[var_name]
            except KeyError as e:
                raise NameError(
                    f'A variable named "{var_name}" was referenced in the configuration'
                    'but no value was provided when loading the configuration'
                ) from e
        else:
            # no variables
            new_config[key] = value
    return new_config


def __handle_single_load(direct_parent: dict, root_path: str) -> dict:
    """
    Apply the ``_load`` directive to a single dictionary.

    Args:
        direct_parent: A direct parent to the ``_load`` directive, i.e. a dictionary
            which contains this ``_load`` key.
        root_path: A path to the configuration file which is being loaded. This is to
            make all paths used by ``_load`` directives relative to this file.

    Returns:
        A copy of ``direct_parent`` dict, with ``_load`` removed and external attributes
        added (if they are missing - to allow overriding)
    """
    argument = direct_parent[LOAD_ATTR_NAME]
    path = os.path.join(os.path.dirname(root_path), argument)
    if os.path.isfile(path):
        with open(path, 'r') as file:
            loaded_data: dict = yaml.safe_load(file)
        # handle _load directives in the loaded file
        loaded_data = __handle_all_loads_from_data(
            data=loaded_data,
            root_path=path,
        )
    else:
        raise FileNotFoundError(
            f'Loading configuration from file {argument} failed because this file does not exist. '
            'Check if the provided path is relative to the configuration file which is currently being read.'
        )
    direct_parent_copy = direct_parent.copy()
    del direct_parent_copy[LOAD_ATTR_NAME]
    merged_data = __handle_overrides(
        default_config=loaded_data,
        overriding_config=direct_parent_copy,
    )
    return merged_data


def __handle_all_loads_from_data(data: dict, root_path: str) -> dict:
    """
    Recursively apply the ``_load`` directive for the whole current data.

    Args:
        data: Data to apply the ``_load`` for.
        root_path: A path to the configuration file which is being loaded. This is to
            make all paths used by ``_load`` directives relative to this file.

    Returns:
        A copy of ``data`` dict, with all ``_load`` directives handled.
    """
    # the reason for 'while' being here and not 'if' is because sometimes _load can load another _load,
    # (see TestLoadTaskConfig.test_chained_load for an example case when this may occur).
    while LOAD_ATTR_NAME in data.keys():
        data = __handle_single_load(data, root_path)
    # handle nested loads
    data_processed = {}
    for key, value in data.items():
        if isinstance(value, dict):
            data_processed[key] = __handle_all_loads_from_data(
                data=value,
                root_path=root_path,
            )
        else:
            data_processed[key] = value
    return data_processed


def _load_task_from_dict(task_data: dict) -> 'LearningTask':
    """
    Creates a task based on a configuration stored in a dictionary.
    This is a helper method used by config loaders and it is not useful for the end user.

    Args:
        task_data: dictionary that contains task configuration.
            It is assumend that all the variables had beed substituted with values,
            and that any directives had been handled

    Returns:
        A :class:`LearningTask` instance based on the provided configuration.
    """
    env_type = task_data.pop('env_type')
    # this check is now necessary because env_type can be passed directly using variables
    if isinstance(env_type, str):
        env_type = SavableLoadable.get_type(env_type)
    return LearningTask(env_type=env_type, **task_data)


def load_task_config(path: str, variables: Optional[dict] = None) -> LearningTask:
    """
    Loads a task configuration from the specified file.

    A configuration file should be in YAML format, with 'task.yml' being the preferred extension.
    Properties names should be identical to the arguments of the :class:`LearningTask`'s constructor.

    An example task configuration file::

        # my_config.task.yml
        name: lava_crossing_hard
        env_type: academia.environments.LavaCrossing
        env_args:
            difficulty: 2
            render_mode: human
        stop_conditions:
            max_episodes: 1000
        output_dir: .

    Args:
        path: Path to a configuration file.
        variables: Variable values for the configuration file.

    Returns:
        A :class:`LearningTask` instance based on the configuration in the specified file.

    Note:
        For details on how to use configure tasks via YAML
        files please refer to :ref:`config-files`.
    """
    if variables is None:
        variables = {}

    with open(path, 'r') as file:
        task_data: dict = yaml.safe_load(file)

    task_data = __handle_all_loads_from_data(task_data, root_path=path)
    task_data = __inject_variables(task_data, variables)

    return _load_task_from_dict(task_data)


def __handle_default_task_config(tasks_data: dict) -> dict:
    """
    Provide default parameters for tasks collection if it contains a task
    with the ID ``_default``.

    Args:
        tasks_data: Raw tasks data obtained from the curriculum configuration
            (``tasks`` attribute)

    Returns:
        A copy of ``tasks_data`` but with defaults injected into other tasks
        (if not overriden)
    """

    if DEFAULT_ATTR_NAME not in tasks_data.keys():
        return tasks_data

    default_task = tasks_data[DEFAULT_ATTR_NAME]
    if not isinstance(default_task, dict):
        # if someone passes a regular LearningTask object or anything else through a variable
        # as a default task, nothing can be done
        _logger.warning(f'Default task is of an unsupported type: {type(default_task)}. '
                        'Its attributes will not be used as defaults for other tasks.')
        return tasks_data

    # maintain default task
    tasks_data_processed = {'_default': default_task}
    for task_id, task_raw in tasks_data.items():
        tasks_data_processed[task_id] = __handle_overrides(
            default_config=default_task,
            overriding_config=task_raw,
        )
    return tasks_data_processed


def load_curriculum_config(path: str, variables: Optional[dict] = None) -> Curriculum:
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
        variables: Variable values for the configuration file.

    Returns:
        A :class:`Curriculum` instance based on the configuration in the specified file.

    Note:
        For details on how to use configure curricula via YAML
        files please refer to :ref:`config-files`.
    """
    if variables is None:
        variables = {}

    with open(path, 'r') as file:
        curriculum_data: dict = yaml.safe_load(file)

    curriculum_data = __handle_all_loads_from_data(curriculum_data, root_path=path)
    curriculum_data = __inject_variables(curriculum_data, variables)

    tasks_data = curriculum_data['tasks']
    tasks_data = __handle_default_task_config(tasks_data)

    tasks = []
    for task_id in curriculum_data['order']:
        try:
            task_raw = tasks_data[task_id]
        except KeyError:
            raise NameError(f'Task "{task_id}" not found in the curriculum configuration.')
        if isinstance(task_raw, dict):
            task = _load_task_from_dict(task_raw)
        else:
            # task is some other object, maybe LearningTask - if not, it will raise errors later anyway
            task = task_raw
        tasks.append(task)

    del curriculum_data['order']
    del curriculum_data['tasks']
    return Curriculum(tasks, **curriculum_data)
