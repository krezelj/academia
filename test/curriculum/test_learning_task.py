import unittest
from unittest import mock

import numpy as np

from academia.environments.base import ScalableEnvironment
from academia.agents.base import Agent
from academia.curriculum import LearningTask


def _get_mock_learning_task(mock_env, stop_conditions=None):
    if stop_conditions is None:
        stop_conditions = {}
    return LearningTask(
        # type mismatch for env_type but passing a callable will work the same as passing a class
        env_type=lambda: mock_env,
        env_args={},
        stop_conditions=stop_conditions,
    )


@mock.patch(
    'academia.agents.base.Agent',
    **{
        'get_action.return_value': 0,
        'update.return_value': None,
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
class TestLearningTask(unittest.TestCase):

    def test_max_episodes_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_mock_learning_task(mock_env, stop_conditions={'max_episodes': 3})
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
        sut = _get_mock_learning_task(mock_env, stop_conditions={'max_steps': 30})
        # # act
        sut.run(mock_agent)
        # # assert
        self.assertGreaterEqual(np.sum(sut.stats.step_counts), 30, msg='Training should end later')
        self.assertLess(np.sum(sut.stats.step_counts[:-1]), 30, msg='Training should have ended earlier')

    def test_min_avg_reward_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_mock_learning_task(mock_env, stop_conditions={'min_avg_reward': 1})
        sut.run(mock_agent)
        self.assertGreaterEqual(len(sut.stats.episode_rewards), 5,
                                msg='Condition should be triggered after at least 5 episodes')
        self.assertGreaterEqual(sut.stats.episode_rewards_moving_avg[-1], 1)

    def test_max_reward_std_dev_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_mock_learning_task(mock_env, stop_conditions={'max_reward_std_dev': 1})
        sut.run(mock_agent)
        self.assertGreaterEqual(len(sut.stats.episode_rewards), 10,
                                msg='Std dev should be calculated from at least 10 episodes')
        self.assertLessEqual(np.std(sut.stats.episode_rewards[-10:]), 1)

    def test_min_evaluation_score_stop_condition(self, mock_env: ScalableEnvironment, mock_agent: Agent):
        sut = _get_mock_learning_task(mock_env, stop_conditions={'min_evaluation_score': 100})
        sut.run(mock_agent)
        self.assertGreaterEqual(sut.stats.agent_evaluations[-1], 100)


if __name__ == '__main__':
    unittest.main()
