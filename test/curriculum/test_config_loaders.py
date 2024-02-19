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
        name = file.readline()
    mock_task = mock.MagicMock()
    mock_task.name = name
    return mock_task


def _mock_load_task_from_dict(task_data: dict):
    mock_task = mock.MagicMock()
    mock_task.name = task_data['name']
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

    def test_loading_config_basic(self):
        """
        ``Curriculum`` should be able to load a configuration from a YAML file.
        In this scenario every task's configuration is in the same file.
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

    def test_loading_config_task_separate_file(self):
        """
        ``Curriculum`` should be able to load a configuration from a YAML file.
        In this scenario task's configuration is in a separate file.
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
                "    path: ../tasks/3.task.yml"
            )
            config_file_path = os.path.join(temp_dir, 'curricula/config.curriculum.yml')
            with open(config_file_path, 'w') as f:
                f.write(curriculum_config)
            with open(os.path.join(temp_dir, 'tasks/3.task.yml'), 'w') as f:
                # super simplified task config for mock LearningTask.load()
                f.write("Craig Forsyth")
            # act
            sut = load_curriculum_config(config_file_path)
        # assert
        self.assertEqual('Craig Forsyth', sut.tasks[0].name)


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
                           f"agent_save_path: {temp_dir}/secret_agent_123\n"
                           f"stats_save_path: {temp_dir}/super_stats_321")
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
        self.assertEqual(f'{temp_dir}/secret_agent_123', sut.agent_save_path)
        self.assertEqual(f'{temp_dir}/super_stats_321', sut.stats_save_path)
        # stop condition
        self.assertEqual(2, len(sut.stats.episode_rewards))


if __name__ == '__main__':
    unittest.main()
