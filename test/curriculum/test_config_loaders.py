import unittest
from unittest import mock
import tempfile
import os

import numpy as np

from academia.environments.base import ScalableEnvironment
from academia.agents.base import Agent
from academia.curriculum import LearningStats, load_curriculum_config, load_task_config
from academia.utils import SavableLoadable


def _mock_load_task_config(path: str):
    with open(path, 'r') as file:
        # first line is something like this: "name: my_name"
        name = file.readline()[6:]
    mock_task = mock.MagicMock()
    mock_task.name = name
    return mock_task


def _mock_load_task_from_dict(task_data: dict):
    mock_task = mock.MagicMock()
    mock_task.name = task_data['name']
    mock_task.output_dir = task_data.get('output_dir')
    return mock_task


@mock.patch(
    'academia.curriculum.config_loaders.load_task_config',
    new=_mock_load_task_config,
)
@mock.patch(
    'academia.curriculum.config_loaders._load_task_from_dict',
    new=_mock_load_task_from_dict,
)
class TestLoadCurriculumConfig(unittest.TestCase):

    def test_loading_simple(self):
        """
        ``Curriculum`` should be able to load all configuration from a YAML file
        in the order specified by the ``order`` attribute.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "output_dir: ./hello_output\n"
                "order:\n"
                "  - 1\n"
                "  - 24\n"
                "  - 28\n"
                "tasks:\n"  # simplified task config
                "  28:\n"
                "    name: Cesar Azpilicueta\n"
                "  1:\n"
                "    name: Petr Cech\n"
                "  24:\n"
                "    name: Gary Cahill"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act
            sut = load_curriculum_config(config_file_path)
        # assert
        self.assertEqual('./hello_output', sut.output_dir)
        self.assertEqual('Petr Cech', sut.tasks[0].name)
        self.assertEqual('Gary Cahill', sut.tasks[1].name)
        self.assertEqual('Cesar Azpilicueta', sut.tasks[2].name)
        self.assertEqual(3, len(sut.tasks))

    def test_name_error_when_task_missing(self):
        """
        NameError should be raised when a task is referenced in the order which has
        not been defined in this curriculum
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "output_dir: ./hello_output\n"
                "order:\n"
                "  - 1\n"
                "tasks:\n"  # simplified task config
                "  28:\n"
                "    name: Cesar Azpilicueta\n"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act & assert
            with self.assertRaises(NameError):
                load_curriculum_config(config_file_path)

    def test_task_unused(self):
        """
        ``Curriculum`` should skip the tasks that are not listed in the ``order`` attribute.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "order:\n"
                "  - 1\n"
                "  - 24\n"
                "tasks:\n"  # simplified task config
                "  28:\n"
                "    name: Cesar Azpilicueta\n"
                "  1:\n"
                "    name: Petr Cech\n"
                "  24:\n"
                "    name: Gary Cahill"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act
            sut = load_curriculum_config(config_file_path)
        # assert
        self.assertEqual('Petr Cech', sut.tasks[0].name)
        self.assertEqual('Gary Cahill', sut.tasks[1].name)
        self.assertEqual(2, len(sut.tasks))

    def test_load(self):
        """
        The ``_load`` directive should load task configuration from a separate file
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.mkdir(os.path.join(temp_dir, 'curricula'))
            os.mkdir(os.path.join(temp_dir, 'tasks'))
            # arrange
            curriculum_config = (
                "order:\n"
                "  - 3\n"
                "tasks:\n"
                "  3:\n"
                "    _load: ../tasks/3.task.yml"
            )
            config_file_path = os.path.join(temp_dir, 'curricula/config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            with open(os.path.join(temp_dir, 'tasks/3.task.yml'), 'w') as f:
                # super simplified task config for mock LearningTask.load()
                f.write("name: Craig Forsyth")
            # act
            sut = load_curriculum_config(config_file_path)
        # assert
        self.assertEqual('Craig Forsyth', sut.tasks[0].name)

    def test_default_task_config(self):
        """
        Attributes defined inside the ``_default`` task should carry over to all other tasks
        unless explicitly overriden
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "order:\n"
                "  - 1\n"
                "tasks:\n"  # simplified task config
                "  _default:\n"
                "    name: base_name\n"
                "    output_dir: ./tasks\n"
                "  1:\n"
                "    name: new_name"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act
            sut = load_curriculum_config(config_file_path)
        # assert
        self.assertEqual('new_name', sut.tasks[0].name)
        self.assertEqual('./tasks', sut.tasks[0].output_dir)

    def test_variables_simple(self):
        """
        Variables should be correctly loaded
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "output_dir: $out\n"
                "order:\n"
                "  - 1\n"
                "tasks:\n"  # simplified task config
                "  1:\n"
                "    name: $name"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act
            sut = load_curriculum_config(config_file_path, variables={
                'out': './my_curriculum',
                'name': 'Alfie May is a baller',
            })
        # assert
        self.assertEqual('./my_curriculum', sut.output_dir)
        self.assertEqual('Alfie May is a baller', sut.tasks[0].name)

    def test_variables_missing(self):
        """
        ``NameError`` should be raised when a variable is missing
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "output_dir: $out\n"
                "order:\n"
                "  - 1\n"
                "tasks:\n"  # simplified task config
                "  1:\n"
                "    name: $name"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act & assert
            with self.assertRaises(NameError):
                load_curriculum_config(config_file_path, variables={
                    'out': './my_curriculum',
                })

    def test_variable_full_tasks_data(self):
        """
        A curriculum task data should be able to be loaded from a variable
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "order:\n"
                "  - 1\n"
                "tasks:\n"  # simplified task config
                "  1: $my_task\n"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act
            sut = load_curriculum_config(config_file_path, variables={
                'my_task': {'name': 'super_task'},
            })
        # assert
        self.assertEqual('super_task', sut.tasks[0].name)

    def test_variable_full_tasks_obj(self):
        """
        A curriculum task should be able to be loaded from a variable directly as an object
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "order:\n"
                "  - 1\n"
                "tasks:\n"  # simplified task config
                "  1: $my_task\n"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act
            sut = load_curriculum_config(config_file_path, variables={
                'my_task': _mock_load_task_from_dict({'name': 'super_task'}),
            })
        # assert
        self.assertEqual('super_task', sut.tasks[0].name)

    def test_variable_order(self):
        """
        Curriculum's tasks order should be able to be loaded from a variable
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # arrange
            curriculum_config = (
                "order: $my_order\n"
                "tasks:\n"  # simplified task config
                "  1:\n"
                "    name: Petr Cech\n"
                "  24:\n"
                "    name: Gary Cahill"
            )
            config_file_path = os.path.join(temp_dir, 'config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            # act
            sut = load_curriculum_config(config_file_path, variables={
                'my_order': [1, 24]
            })
        # assert
        self.assertEqual('Petr Cech', sut.tasks[0].name)
        self.assertEqual('Gary Cahill', sut.tasks[1].name)
        self.assertEqual(2, len(sut.tasks))


@mock.patch(
    'academia.agents.base.Agent',
    **{
        'get_action.return_value': 0,
        'update.return_value': None,
        'save': lambda path: path,
    }
)
@mock.patch(
    'academia.environments.base.ScalableEnvironment',
    **{
        'step.return_value': (0, 100, True),
        'reset.return_value': None,
        'observe.return_value': 0,
        'render.return_value': None,
        'get_legal_mask.return_value': np.array([1]),
    }
)
@mock.patch.object(LearningStats, 'save', lambda self, path: path)
class TestLoadTaskConfig(unittest.TestCase):

    def test_loading_config(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        ``LearningTask`` should be able to load a configuration from a YAML file
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            task_config = ("env_type: placeholder\n"
                           "env_args:\n"
                           "  difficulty: 0\n"
                           "stop_conditions:\n"
                           "  max_episodes: 2\n"
                           "evaluation_interval: 24\n"
                           "name: Reece James\n"
                           f"output_dir: {temp_dir}/out\n")
            config_file_path = os.path.join(temp_dir, 'config.task.yml')
            with open(config_file_path, 'w') as f:
                f.write(task_config)
            # load config - it should be identical to the task defined above
            # patch to avoid error when loading the mock environment
            with mock.patch.object(SavableLoadable, 'get_type',
                                   mock.MagicMock(return_value=lambda *args, **kwargs: mock_env)):
                sut = load_task_config(config_file_path)

        # run the task to be able to check some of the parameters validity later
        sut.run(mock_agent)

        self.assertEqual('Reece James', sut.name)
        self.assertEqual(24, sut.stats.evaluation_interval)
        self.assertEqual(f'{temp_dir}/out', sut.output_dir)
        # stop condition
        self.assertEqual(2, len(sut.stats.episode_rewards))

    def test_load(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        The ``_load`` directive should load and allow to override other task configuration
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_task_config = (
                "name: base_name\n"
                "evaluation_interval: 24\n"
            )
            new_task_config = (
                "_load: ./base.task.yml\n"
                "env_type: placeholder\n"
                "env_args:\n"
                "  difficulty: 0\n"
                "stop_conditions:\n"
                "  max_episodes: 2\n"
                f"output_dir: {temp_dir}/out\n"
            )
            base_config_file_path = os.path.join(temp_dir, 'base.task.yml')
            with open(base_config_file_path, 'w') as f:
                f.write(base_task_config)
            new_config_file_path = os.path.join(temp_dir, 'config.task.yml')
            with open(new_config_file_path, 'w') as f:
                f.write(new_task_config)
            # load config - it should be identical to the task defined above
            # patch to avoid error when loading the mock environment
            with mock.patch.object(SavableLoadable, 'get_type',
                                   mock.MagicMock(return_value=lambda *args, **kwargs: mock_env)):
                sut = load_task_config(new_config_file_path)

        # run the task to be able to check some of the parameters validity later
        sut.run(mock_agent)

        self.assertEqual('base_name', sut.name)
        self.assertEqual(24, sut.stats.evaluation_interval)
        self.assertEqual(f'{temp_dir}/out', sut.output_dir)
        # stop condition
        self.assertEqual(2, len(sut.stats.episode_rewards))

    def test_chained_load(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        The ``_load`` directive should be handled in loaded files as well (chaining)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            first_config = (
                "name: base_name\n"
            )
            second_config = (
                "_load: ./first.task.yml\n"
                "evaluation_interval: 24\n"
            )
            third_config = (
                "_load: ./second.task.yml\n"
                "env_type: placeholder\n"
                "env_args:\n"
                "  difficulty: 0\n"
                "stop_conditions:\n"
                "  max_episodes: 2\n"
                f"output_dir: {temp_dir}/out\n"
            )
            with open(os.path.join(temp_dir, 'first.task.yml'), 'w') as f:
                f.write(first_config)
            with open(os.path.join(temp_dir, 'second.task.yml'), 'w') as f:
                f.write(second_config)
            final_config_path = os.path.join(temp_dir, 'third.task.yml')
            with open(final_config_path, 'w') as f:
                f.write(third_config)
            # load config - it should be identical to the task defined above
            # patch to avoid error when loading the mock environment
            with mock.patch.object(SavableLoadable, 'get_type',
                                   mock.MagicMock(return_value=lambda *args, **kwargs: mock_env)):
                sut = load_task_config(final_config_path)

        # run the task to be able to check some of the parameters validity later
        sut.run(mock_agent)

        self.assertEqual('base_name', sut.name)
        self.assertEqual(24, sut.stats.evaluation_interval)
        self.assertEqual(f'{temp_dir}/out', sut.output_dir)
        # stop condition
        self.assertEqual(2, len(sut.stats.episode_rewards))

    def test_variables_simple(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Variables should be substituted with provided values
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            task_config = ("env_type: placeholder\n"
                           "env_args:\n"
                           "  difficulty: 0\n"
                           "stop_conditions:\n"
                           "  max_episodes: $max_episodes\n"
                           "evaluation_interval: $evaluation_int\n")
            config_file_path = os.path.join(temp_dir, 'config.task.yml')
            with open(config_file_path, 'w') as f:
                f.write(task_config)
            # load config - it should be identical to the task defined above
            # patch to avoid error when loading the mock environment
            with mock.patch.object(SavableLoadable, 'get_type',
                                   mock.MagicMock(return_value=lambda *args, **kwargs: mock_env)):
                sut = load_task_config(config_file_path, variables={
                    'max_episodes': 2,
                    'evaluation_int': 24,
                })

        # run the task to be able to check some of the parameters validity later
        sut.run(mock_agent)

        self.assertEqual(24, sut.stats.evaluation_interval)
        # stop condition
        self.assertEqual(2, len(sut.stats.episode_rewards))

    def test_variable_missing(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        ``NameError`` should be raised when a variable is missing
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            task_config = ("env_type: placeholder\n"
                           "env_args:\n"
                           "  difficulty: 0\n"
                           "stop_conditions:\n"
                           "  max_episodes: $max_episodes\n"
                           "evaluation_interval: $evaluation_int\n")
            config_file_path = os.path.join(temp_dir, 'config.task.yml')
            with open(config_file_path, 'w') as f:
                f.write(task_config)
            # load config - it should be identical to the task defined above
            # patch to avoid error when loading the mock environment
            with mock.patch.object(SavableLoadable, 'get_type',
                                   mock.MagicMock(return_value=lambda *args, **kwargs: mock_env)):
                # act & assert
                with self.assertRaises(NameError):
                    load_task_config(config_file_path, variables={
                        'max_episodes': 2,
                    })

    def test_variables_in_external_files(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Variables should be replaced with values even if they are used in external files
        which are loaded with the ``_load`` directive
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base_task_config = (
                "name: $name\n"
                "evaluation_interval: 24\n"
            )
            new_task_config = (
                "_load: ./base.task.yml\n"
                "env_type: placeholder\n"
                "env_args:\n"
                "  difficulty: 0\n"
                "stop_conditions:\n"
                "  max_episodes: 2\n"
                f"output_dir: {temp_dir}/out\n"
            )
            base_config_file_path = os.path.join(temp_dir, 'base.task.yml')
            with open(base_config_file_path, 'w') as f:
                f.write(base_task_config)
            new_config_file_path = os.path.join(temp_dir, 'config.task.yml')
            with open(new_config_file_path, 'w') as f:
                f.write(new_task_config)
            # load config - it should be identical to the task defined above
            # patch to avoid error when loading the mock environment
            with mock.patch.object(SavableLoadable, 'get_type',
                                   mock.MagicMock(return_value=lambda *args, **kwargs: mock_env)):
                sut = load_task_config(new_config_file_path, variables={
                    'name': 'my_name'
                })
        # assert
        self.assertEqual('my_name', sut.name)


if __name__ == '__main__':
    unittest.main()
