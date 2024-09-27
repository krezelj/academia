from typing import Optional
import unittest
from unittest import mock

from academia.curriculum import Curriculum


def _get_mock_learning_tasks(task_names: list[Optional[str]]):
    """Tuple of mock LearningTasks"""
    result = []
    for name in task_names:
        mock_task = mock.MagicMock()
        mock_task.name = name
        mock_task.run = lambda *args, **kwargs: None
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
    def test_output_config_override(self, mock_agent):
        """
        Curriculum should override task's output_dir and name if and only if
        they're not specified for that task.
        """
        # arrange
        task1, task2 = _get_mock_learning_tasks(['task_with_output_dir', 'task_with_name'])
        task1.output_dir = './task1'
        task1.name = None
        task2.output_dir = None
        task2.name = 'task2'
        # act
        sut = Curriculum([task1, task2], output_dir='./my_curriculum')
        sut.run(mock_agent)
        # assert
        self.assertEqual('./task1', task1.output_dir)
        self.assertEqual('1', task1.name)
        self.assertEqual('./my_curriculum', task2.output_dir)
        self.assertEqual('task2', task2.name)

    @mock.patch('academia.agents.base.Agent')
    def test_task_callback_triggering(self, mock_agent):
        """
        Task callback should trigger after every task.
        """
        tracking_list = []

        def task_callback(agent, stats, task_id):
            tracking_list.append(task_id)

        # arrange
        task1, task2, task3 = _get_mock_learning_tasks(['1', '2', '3'])
        sut = Curriculum([task1, task2, task3], task_callback=task_callback)
        # act
        sut.run(mock_agent)
        # assert
        self.assertEqual(['1', '2', '3'], tracking_list,
                         msg='Task callback triggered an incorrect number of times')

    @mock.patch('academia.agents.base.Agent')
    def test_task_callback_agent_overwrite(self, mock_agent):
        """
        Task callback should be able to overwrite the currently used agent
        """
        was_new_agent_used = False

        def task_callback(agent, stats, task_id):
            def get_action(*args, **kwargs):
                nonlocal was_new_agent_used
                was_new_agent_used = True

            new_agent = mock.MagicMock()
            new_agent.get_action = get_action
            return new_agent

        # arrange
        task1, task2 = _get_mock_learning_tasks([None, None])

        def task2_run(agent, *args, **kwargs):
            agent.get_action()

        task2.run = task2_run
        sut = Curriculum([task1, task2], task_callback=task_callback)
        # act
        sut.run(mock_agent)
        # assert
        self.assertTrue(was_new_agent_used, msg='New agent was not used')


if __name__ == '__main__':
    unittest.main()
