import unittest
from unittest import mock
import tempfile
import os
import logging

import numpy as np

from academia.environments.base import ScalableEnvironment
from academia.agents.base import Agent
from academia.curriculum import LearningTask, LearningStats, LearningStatsAggregator

# otherwise errors are logged when testing task interruption behaviour
logging.disable(logging.ERROR)


def _get_learning_task(mock_env, stop_conditions=None, other_task_args=None) -> LearningTask:
    """Sample LearningTask"""
    if stop_conditions is None:
        stop_conditions = {'max_episodes': 1}
    if other_task_args is None:
        other_task_args = {}
    return LearningTask(
        # type mismatch for env_type but passing a callable will work the same as passing a class
        env_type=lambda *args, **kwargs: mock_env,
        env_args={},
        stop_conditions=stop_conditions,
        **other_task_args,
    )


def _get_learning_stats() -> LearningStats:
    """Sample LearningStats"""
    stats = LearningStats(evaluation_interval=24)
    stats.episode_rewards = np.array([1, 2, 3])
    stats.step_counts = np.array([1, 2, 3])
    stats.episode_rewards_moving_avg = np.array([1, 2, 3])
    stats.step_counts_moving_avg = np.array([1, 2, 3])
    stats.agent_evaluations = np.array([1, 2, 3])
    stats.episode_wall_times = np.array([1, 2, 3])
    stats.episode_cpu_times = np.array([1, 2, 3])
    return stats


def _mock_save_agent(path: str):
    if not path.endswith('agent.zip'):
        path = f'{path}.agent.zip'
    # create an empty file
    with open(path, 'w'):
        pass
    return path


def _mock_save_stats(path: str):
    if not path.endswith('stats.json'):
        path = f'{path}.stats.json'
    # create an empty file
    with open(path, 'w'):
        pass
    return path


@mock.patch(
    'academia.agents.base.Agent',
    **{
        'get_action.return_value': 0,
        'update.return_value': None,
        'save': _mock_save_agent,
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
@mock.patch.object(LearningStats, 'save', lambda self, path: _mock_save_stats(path))
class TestLearningTask(unittest.TestCase):

    def test_max_episodes_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_learning_task(mock_env, stop_conditions={'max_episodes': 3})
        sut.run(mock_agent)
        self.assertEqual(3, len(sut.stats.episode_rewards))

    def test_max_steps_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        # # arrange
        step_counter = [0]
        # action 1 triggers after 10 steps
        mock_agent.get_action = lambda state, legal_mask, greedy: 1 if step_counter[0] == 9 else 0

        def env_step(action):
            step_counter[0] += 1
            # action 1 ends the episode
            if action == 1:
                step_counter[0] = 0
                return 0, 100, True
            return 0, 100, False

        mock_env.step = env_step
        sut = _get_learning_task(mock_env, stop_conditions={'max_steps': 30})
        # # act
        sut.run(mock_agent)
        # # assert
        self.assertGreaterEqual(np.sum(sut.stats.step_counts), 30, msg='Training should end later')
        self.assertLess(np.sum(sut.stats.step_counts[:-1]), 30, msg='Training should have ended earlier')

    def test_max_wall_time_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        # arrange
        sut = _get_learning_task(mock_env, stop_conditions={'max_wall_time': 0.05})
        # # act
        sut.run(mock_agent)
        # # assert
        self.assertGreaterEqual(np.sum(sut.stats.episode_wall_times), 0.05, msg='Training should end later')
        self.assertLess(np.sum(sut.stats.episode_wall_times[:-1]), 0.05,
                        msg='Training should have ended earlier')

    def test_min_avg_reward_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_learning_task(mock_env, stop_conditions={'min_avg_reward': 1})
        sut.run(mock_agent)
        self.assertGreaterEqual(len(sut.stats.episode_rewards), 5,
                                msg='Condition should be triggered after at least 5 episodes')
        self.assertGreaterEqual(sut.stats.episode_rewards_moving_avg[-1], 1)

    def test_max_reward_std_dev_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_learning_task(mock_env, stop_conditions={'max_reward_std_dev': 1})
        sut.run(mock_agent)
        self.assertGreaterEqual(len(sut.stats.episode_rewards), 10,
                                msg='Std dev should be calculated from at least 10 episodes')
        self.assertLessEqual(np.std(sut.stats.episode_rewards[-10:]), 1)

    def test_min_evaluation_score_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_learning_task(mock_env, stop_conditions={'min_evaluation_score': 100})
        sut.run(mock_agent)
        self.assertGreaterEqual(sut.stats.agent_evaluations[-1], 100)

    def test_unknown_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        with self.assertRaises(ValueError, msg='Unknown stop condition should raise an error'):
            _get_learning_task(mock_env, stop_conditions={'unknown_stop': 123})

    def test_include_init_eval_param(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        # arrange
        sut_init_eval = _get_learning_task(
            mock_env,
            other_task_args={
                'include_init_eval': True,
            }
        )
        sut_no_init_eval = _get_learning_task(
            mock_env,
            other_task_args={
                'include_init_eval': False,
            }
        )
        # act
        sut_init_eval.run(mock_agent)
        sut_no_init_eval.run(mock_agent)

        # assert
        self.assertEqual(0, len(sut_no_init_eval.stats.agent_evaluations))
        self.assertEqual(1, len(sut_init_eval.stats.agent_evaluations))

    def test_evaluation_count_param(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        # arrange
        sut = _get_learning_task(
            mock_env,
            other_task_args={
                'evaluation_count': 28,
                'include_init_eval': True,
            }
        )
        # configure mock agent to count evaluations (to validate evaluation_count)
        eval_counter = [0]

        def agent_get_action(state, legal_mask, greedy):
            if greedy:
                eval_counter[0] += 1
            return 0

        mock_agent.get_action = agent_get_action

        # act
        sut.run(mock_agent)
        # assert
        self.assertEqual(28, eval_counter[0])

    def test_evaluation_interval_param(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        # arrange
        sut_init_eval = _get_learning_task(
            mock_env,
            stop_conditions={
                'max_episodes': 22,
            },
            other_task_args={
                'evaluation_interval': 5,
                'include_init_eval': True,
            }
        )
        sut_no_init_eval = _get_learning_task(
            mock_env,
            stop_conditions={
                'max_episodes': 22,
            },
            other_task_args={
                'evaluation_interval': 5,
                'include_init_eval': False,
            }
        )
        # act
        sut_init_eval.run(mock_agent)
        sut_no_init_eval.run(mock_agent)
        # assert
        self.assertEqual(5, len(sut_init_eval.stats.agent_evaluations))
        self.assertEqual(4, len(sut_no_init_eval.stats.agent_evaluations))

    def test_agent_state_saving_normal(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Save path should not have 'backup' prepended when saving normally
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            sut = _get_learning_task(mock_env, other_task_args={
                'output_dir': os.path.join(temp_dir, 'test'),
                'name': 'test_task',
            })
            # act
            sut.run(mock_agent)
            # assert
            expected_save_path = os.path.join(temp_dir, 'test', 'test_task.agent.zip')
            self.assertTrue(os.path.isfile(expected_save_path))

    def test_agent_state_saving_backup(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Save path should have 'backup' prepended when training interrupted
        """
        mock_agent.get_action.side_effect = Exception()
        with tempfile.TemporaryDirectory() as temp_dir:
            sut = _get_learning_task(mock_env, other_task_args={
                'output_dir': os.path.join(temp_dir, 'test'),
                'name': 'test_task',
            })
            # act
            try:
                sut.run(mock_agent)
            except SystemExit:
                pass
            # assert
            expected_save_path = os.path.join(temp_dir, 'test', 'backup_test_task.agent.zip')
            self.assertTrue(os.path.isfile(expected_save_path))

    def test_stats_saving_normal(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Save path should not have 'backup' prepended when saving normally
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            sut = _get_learning_task(mock_env, other_task_args={
                'output_dir': os.path.join(temp_dir, 'test'),
                'name': 'test_task',
            })
            # act
            sut.run(mock_agent)
            # assert
            expected_save_path = os.path.join(temp_dir, 'test', 'test_task.stats.json')
            self.assertTrue(os.path.isfile(expected_save_path))

    def test_stats_saving_backup(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Save path should have 'backup' prepended when training interrupted
        """
        mock_agent.get_action.side_effect = Exception()
        with tempfile.TemporaryDirectory() as temp_dir:
            sut = _get_learning_task(mock_env, other_task_args={
                'output_dir': os.path.join(temp_dir, 'test'),
                'name': 'test_task',
            })
            # act
            try:
                sut.run(mock_agent)
            except SystemExit:
                pass
            # assert
            expected_save_path = os.path.join(temp_dir, 'test', 'backup_test_task.stats.json')
            self.assertTrue(os.path.isfile(expected_save_path))

    def test_episode_callback_triggering(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Episode callback should trigger after every episode.
        """
        tracking_list = []

        def episode_callback(agent, stats, episode_no):
            tracking_list.append(episode_no)

        sut = _get_learning_task(
            mock_env,
            stop_conditions={'max_episodes': 3},
            other_task_args={
                'episode_callback': episode_callback,
            },
        )
        sut.run(mock_agent)
        self.assertEqual(3, len(tracking_list),
                         msg='Episode callback triggered an incorrect number of times')

    def test_episode_callback_agent_overwrite(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        """
        Episode callback should be able to overwrite the currently used agent
        """
        was_new_agent_used = False

        def episode_callback(agent, stats, episode_no):
            def update(*args, **kwargs):
                nonlocal was_new_agent_used
                was_new_agent_used = True
                return 0

            new_agent = mock.MagicMock()
            new_agent.get_action.return_value = 0
            new_agent.update = update
            return new_agent

        sut = _get_learning_task(
            mock_env,
            stop_conditions={'max_episodes': 2},
            other_task_args={
                'episode_callback': episode_callback,
            },
        )
        sut.run(mock_agent)
        self.assertTrue(was_new_agent_used, msg='New agent was not used')


class TestLearningStats(unittest.TestCase):

    def test_saving_loading(self):
        """
        Stats should be saved in a way that loading them will produce a LearningStats object with
        identical stats
        """
        # arrange
        sut_to_save = _get_learning_stats()
        # act
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        save_path = sut_to_save.save(tmpfile.name)
        sut_loaded = LearningStats.load(save_path)
        # assert
        self.assertEqual(sut_to_save.evaluation_interval, sut_loaded.evaluation_interval)
        self.assertTrue(np.all(sut_to_save.episode_rewards == sut_loaded.episode_rewards))
        self.assertTrue(np.all(sut_to_save.step_counts == sut_loaded.step_counts))
        self.assertTrue(
            np.all(sut_to_save.episode_rewards_moving_avg == sut_loaded.episode_rewards_moving_avg))
        self.assertTrue(np.all(sut_to_save.step_counts_moving_avg == sut_loaded.step_counts_moving_avg))
        self.assertTrue(np.all(sut_to_save.agent_evaluations == sut_loaded.agent_evaluations))
        self.assertTrue(np.all(sut_to_save.episode_wall_times == sut_loaded.episode_wall_times))
        self.assertTrue(np.all(sut_to_save.episode_cpu_times == sut_loaded.episode_cpu_times))
        # cleanup
        tmpfile.close()

    def test_saving_loading_path(self):
        """
        Saving and loading using the same path should always work, regardless whether an expected
        extension is provided or not.
        """
        sut_save = _get_learning_stats()
        with tempfile.TemporaryDirectory() as temp_dir:
            path_no_extension = os.path.join(temp_dir, 'test')
            path_extension = os.path.join(temp_dir, 'test.stats.json')
            sut_save.save(path_no_extension)
            sut_save.save(path_extension)
            try:
                LearningStats.load(path_no_extension)
                LearningStats.load(path_extension)
            except FileNotFoundError:
                self.fail('save() and load() path resolving should match')

    def test_updating(self):
        sut = LearningStats(evaluation_interval=26)
        sut.update(
            episode_no=0,
            episode_reward=111,
            steps_count=222,
            wall_time=333,
            cpu_time=444,
        )
        self.assertEqual(111, sut.episode_rewards[-1])
        self.assertEqual(222, sut.step_counts[-1])
        self.assertEqual(333, sut.episode_wall_times[-1])
        self.assertEqual(444, sut.episode_cpu_times[-1])
        self.assertEqual(1, len(sut.episode_rewards))
        self.assertEqual(1, len(sut.step_counts))
        self.assertEqual(1, len(sut.episode_wall_times))
        self.assertEqual(1, len(sut.episode_cpu_times))
        self.assertEqual(1, len(sut.episode_rewards_moving_avg))
        self.assertEqual(1, len(sut.step_counts_moving_avg))


class TestLearningStatsAggregator(unittest.TestCase):

    def test_timestamp_count(self):
        # arrange
        stats_1 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 3})
        stats_1.step_counts = np.array([0, 5, 3])
        stats_1.episode_rewards = np.array([0, 0, 0])
        stats_2 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 3})
        # cumulative sum is 5 for second episode, the same as for stats_1
        stats_2.step_counts = np.array([1, 4, 2])
        stats_2.episode_rewards = np.array([0, 0, 0])
        stats = [stats_1, stats_2]
        sut = LearningStatsAggregator(stats)

        # act
        aggregate, _ = sut.get_aggregate(value_domain='episode_rewards')

        # assert
        self.assertEqual(5, len(aggregate))

    def test_episodes_time_domain(self):
        # arrange
        stats_1 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 200})
        stats_1.step_counts = np.ones(shape=200)
        stats_1.episode_rewards = np.ones(shape=200)
        stats_2 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 250})
        stats_2.step_counts = np.ones(shape=250)
        stats_2.episode_rewards = np.ones(shape=250) * 2
        stats = [stats_1, stats_2]
        sut = LearningStatsAggregator(stats)

        # act
        aggregate, _ = sut.get_aggregate(time_domain='episodes', value_domain='episode_rewards')

        # assert
        self.assertTrue(np.all(aggregate == 1.5))
        self.assertEqual(250, len(aggregate))

    def test_wall_cpu_time_domains(self):
        for time_domain in ['episode_wall_times', 'episode_cpu_times']:
            # arrange
            stats_1 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 200})
            stats_1.step_counts = np.ones(shape=200)
            stats_1.episode_rewards = np.ones(shape=200)
            setattr(stats_1, time_domain, np.random.random(size=200))
            stats_2 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 250})
            stats_2.step_counts = np.ones(shape=250)
            stats_2.episode_rewards = np.ones(shape=250) * 2
            setattr(stats_2, time_domain, np.random.random(size=250))
            stats = [stats_1, stats_2]
            sut = LearningStatsAggregator(stats)

            # act
            short_name = 'wall_time' if time_domain == 'episode_wall_times' else 'cpu_time'
            _, timestamps = sut.get_aggregate(time_domain=short_name, value_domain='episode_rewards')

            # assert
            expected_timestamps = np.union1d(
                np.cumsum(getattr(stats_1, time_domain)),
                np.cumsum(getattr(stats_2, time_domain)))
            self.assertEqual(len(expected_timestamps), len(timestamps))
            self.assertTrue(np.all(expected_timestamps == timestamps))

    def test_single_task_evaluation_with_init_aggregation(self):
        # arrange
        stats_1 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 100})
        stats_1.step_counts = np.ones(100)
        stats_1.agent_evaluations = np.array([0.0, 0.5])
        stats_1.evaluation_interval = 100
        stats_2 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 200})
        stats_2.step_counts = np.ones(200)
        stats_2.agent_evaluations = np.array([0.0, 1.0, 2.0])
        stats_2.evaluation_interval = 100
        stats = [stats_1, stats_2]
        sut = LearningStatsAggregator(stats)

        # act
        aggregate, timestamps = sut.get_aggregate()

        # assert
        self.assertIsInstance(aggregate, np.ndarray)
        self.assertIsInstance(timestamps, np.ndarray)
        self.assertEqual(3, len(aggregate))
        self.assertTrue(np.all(np.array([0.0, 0.75, 1.25]) == aggregate))
        self.assertEqual(0, timestamps[0])
        self.assertEqual(100, timestamps[1])

    def test_single_task_evaluation_without_init_aggregation(self):
        # arrange
        stats_1 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 100})
        stats_1.step_counts = np.ones(100)
        stats_1.agent_evaluations = np.array([0.5])
        stats_1.evaluation_interval = 100
        stats_2 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 200})
        stats_2.step_counts = np.ones(200)
        stats_2.agent_evaluations = np.array([1.0, 2.0])
        stats_2.evaluation_interval = 100

        stats = [stats_1, stats_2]
        sut = LearningStatsAggregator(stats)

        # act
        aggregate, timestamps = sut.get_aggregate()

        # assert
        self.assertIsInstance(aggregate, np.ndarray)
        self.assertIsInstance(timestamps, np.ndarray)
        self.assertEqual(2, len(aggregate))
        self.assertTrue(np.all(np.array([0.75, 1.25]) == aggregate))
        self.assertEqual(1.0, timestamps[0])
        self.assertEqual(101.0, timestamps[1])

    def test_single_task_reward_aggregation(self):
        # arrange
        stats_1 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 100})
        stats_1.step_counts = np.ones(100)
        stats_1.episode_rewards = np.ones(100)
        stats_2 = mock.MagicMock(spec=LearningStats, **{"__len__.return_value": 200})
        stats_2.step_counts = np.ones(200)
        stats_2.episode_rewards = np.ones(200) * 2
        stats = [stats_1, stats_2]
        sut = LearningStatsAggregator(stats)

        # act
        aggregate, timestamps = sut.get_aggregate(value_domain='episode_rewards')

        # assert
        self.assertIsInstance(aggregate, np.ndarray)
        self.assertIsInstance(timestamps, np.ndarray)
        self.assertEqual(200, len(aggregate))
        self.assertTrue(np.all(1.5 == aggregate))
        self.assertEqual(1, timestamps[0])
        self.assertEqual(2, timestamps[1])

    def test_curriculum_aggregation(self):
        # arrange
        stats = [
            {
                '1': mock.MagicMock(
                    step_counts=np.ones(200),
                    episode_rewards=np.ones(200),
                    **{"__len__.return_value": 200},
                    spec=LearningStats),
                '2': mock.MagicMock(
                    step_counts=np.ones(250),
                    episode_rewards=np.ones(250),
                    **{"__len__.return_value": 250},
                    spec=LearningStats)
            },
            {
                '1': mock.MagicMock(
                    step_counts=np.ones(180),
                    episode_rewards=np.ones(180) * 2,
                    **{"__len__.return_value": 180},
                    spec=LearningStats),
                '2': mock.MagicMock(
                    step_counts=np.ones(270),
                    episode_rewards=np.ones(270) * 3,
                    **{"__len__.return_value": 270},
                    spec=LearningStats)
            }
        ]
        sut = LearningStatsAggregator(stats)

        # act
        aggregate_dict = sut.get_aggregate(value_domain='episode_rewards')

        # assert
        self.assertIsInstance(aggregate_dict, dict)
        self.assertEqual(2, len(aggregate_dict))
        self.assertIn('1', aggregate_dict)
        self.assertIn('2', aggregate_dict)

        self.assertEqual(200, len(aggregate_dict['1'][1]))
        self.assertEqual(270, len(aggregate_dict['2'][1]))

        self.assertTrue(np.all(aggregate_dict['1'][0] == 1.5))
        self.assertTrue(np.all(aggregate_dict['2'][0] == 2))

    def test_inconsistent_curriculum(self):
        # arrange
        stats = [
            {
                '1': mock.MagicMock(
                    step_counts=np.ones(200), episode_rewards=np.ones(200), spec=LearningStats),
                '2': mock.MagicMock(
                    step_counts=np.ones(250), episode_rewards=np.ones(250), spec=LearningStats)
            },
            {
                '1': mock.MagicMock(
                    step_counts=np.ones(180), episode_rewards=np.ones(180) * 2, spec=LearningStats),
                'All Too Well': mock.MagicMock(
                    step_counts=np.ones(270), episode_rewards=np.ones(270) * 3, spec=LearningStats)
            }
        ]

        # act/assert
        with self.assertRaises(ValueError):
            LearningStatsAggregator(stats)

    def test_not_list_like(self):
        with self.assertRaises(ValueError):
            LearningStatsAggregator(0)

    def test_not_composed_of_learning_stats(self):
        with self.assertRaises(ValueError):
            LearningStatsAggregator("Delicate")
        with self.assertRaises(ValueError):
            LearningStatsAggregator([1, 2, 3])
        with self.assertRaises(ValueError):
            LearningStatsAggregator([{'Style': 0, 'Maroon': 1}])

    def test_invalid_parameters(self):
        mock_stats = [LearningStats(1)]
        sut = LearningStatsAggregator(mock_stats)
        with self.assertRaises(ValueError):
            sut.get_aggregate(time_domain='Karma')
        with self.assertRaises(ValueError):
            sut.get_aggregate(value_domain='Ivy')
        with self.assertRaises(ValueError):
            sut.get_aggregate(agg_func_name='Evermore')


if __name__ == '__main__':
    unittest.main()
