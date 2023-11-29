from typing import Optional
import unittest
from unittest import mock
import tempfile
import os

from academia.curriculum import LearningTask, Curriculum


def _learning_task_load(path: str):
    with open(path, 'r') as file:
        name = file.readline()
    mock_task = mock.MagicMock()
    mock_task.name = name
    return mock_task


def _learning_task_from_dict(task_data: dict):
    mock_task = mock.MagicMock()
    mock_task.name = task_data['name']
    return mock_task


def _get_mock_learning_tasks(task_names: list[Optional[str]]):
    """Tuple of mock LearningTasks"""
    result = []
    for name in task_names:
        mock_task = mock.MagicMock()
        mock_task.name = name
        mock_task.run = lambda *args, **kwargs: None
        mock_task.to_dict.return_value = {'name': name}
        result.append(mock_task)
    return tuple(result)


@mock.patch.multiple(
    LearningTask,
    load=_learning_task_load,
    from_dict=_learning_task_from_dict,
)
class TestCurriculum(unittest.TestCase):

    @mock.patch('academia.agents.base.Agent')
    def test_task_run_order(self, mock_agent):
        # arrange
        task1, task2, task3 = _get_mock_learning_tasks([None, None, None])
        order_list = []
        # each task will append its order to order_list
        task3.run = lambda *args, **kwargs: order_list.append(3)
        task2.run = lambda *args, **kwargs: order_list.append(2)
        task1.run = lambda *args, **kwargs: order_list.append(1)
        sut = Curriculum([task1, task2, task3])
        # act
        sut.run(mock_agent)
        # assert
        self.assertEqual([1, 2, 3], order_list)

    def test_stats_dict_key_names(self):
        """Keys in ``Curriculum.stats`` dict should be task IDs and in the right order"""
        task1, task2, task3 = _get_mock_learning_tasks([None, 'John Terry', None])
        sut = Curriculum([task1, task2, task3])
        stats_dict_keys = sut.stats.keys()
        self.assertEqual(['1', 'John Terry', '3'], list(stats_dict_keys))

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
            sut = Curriculum.load(config_file_path)
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
            sut = Curriculum.load(config_file_path)
        # assert
        self.assertEqual('Craig Forsyth', sut.tasks[0].name)

    def test_saving_loading_config(self):
        """
        ``Curriculum``'s configuration should be saved in a way that loading it will produce a curriculum of
        identical configuration
        """
        task1, task2 = _get_mock_learning_tasks(['John Terry', 'Frank Lampard'])
        curriculum_to_save = Curriculum([task1, task2], output_dir='./super_output')
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = curriculum_to_save.save(os.path.join(temp_dir, 'config.curriculum.yml'))
            self.assertTrue(os.path.exists(save_path))
            sut = Curriculum.load(save_path)
        self.assertEqual(curriculum_to_save.output_dir, sut.output_dir)
        self.assertEqual(len(curriculum_to_save.tasks), len(sut.tasks),
                         msg='Task count should be equal in both curricula')
        self.assertEqual(curriculum_to_save.tasks[0].name, sut.tasks[0].name, msg='Task order wrong')
        self.assertEqual(curriculum_to_save.tasks[1].name, sut.tasks[1].name, msg='Task order wrong')

    def test_saving_loading_path(self):
        """
        Saving and loading using the same path should always work, regardless whether an expected
        extension is provided or not.
        """
        task, = _get_mock_learning_tasks(['Lando Norris'])
        sut_save = Curriculum([task])
        with tempfile.TemporaryDirectory() as temp_dir:
            path_no_extension = os.path.join(temp_dir, 'config')
            path_extension = os.path.join(temp_dir, 'config.curriculum.yml')
            sut_save.save(path_no_extension)
            sut_save.save(path_extension)
            try:
                Curriculum.load(path_no_extension)
                Curriculum.load(path_extension)
            except FileNotFoundError:
                self.fail('save() and load() path resolving should match')

    @mock.patch('academia.agents.base.Agent')
    def test_save_paths_override(self, mock_agent):
        """
        Curriculum should override task's save paths if and only if they're not specified for that task.
        """
        # arrange
        task1, task2 = _get_mock_learning_tasks(['task_with_agent_path', 'task_with_stats_path'])
        task1.agent_save_path = './my_agent'
        task1.stats_save_path = None
        task2.stats_save_path = './my_stats'
        task2.agent_save_path = None
        # act
        sut = Curriculum([task1, task2], output_dir='./my_curriculum')
        sut.run(mock_agent)
        # assert
        self.assertEqual('./my_agent', task1.agent_save_path)
        self.assertEqual(os.path.join('./my_curriculum', 'task_with_agent_path'), task1.stats_save_path)
        self.assertEqual(os.path.join('./my_curriculum', 'task_with_stats_path'), task2.agent_save_path)
        self.assertEqual('./my_stats', task2.stats_save_path)


if __name__ == '__main__':
    unittest.main()
