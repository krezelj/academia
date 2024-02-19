from typing import Optional
import unittest
from unittest import mock
import os

from academia.curriculum import Curriculum


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
