import unittest
import tempfile
import os

import numpy as np

from academia.curriculum import LearningStats
from academia.tools.visualizations import (
    plot_trajectories,
    plot_evaluation_impact,
    plot_time_impact,
    plot_evaluation_impact_2d
)


class TestVisualizations(unittest.TestCase):

    @staticmethod
    def __fill_stats(stats: LearningStats, n: int = 50, include_init_eval: bool = True):
        if include_init_eval:
            stats.agent_evaluations = np.append(stats.agent_evaluations, np.random.random())
        for i in range(n):
            stats.update(
                i, np.random.random(), np.random.randint(0, 1000),
                np.random.random(), np.random.random()
            )
            if (i+1) % stats.evaluation_interval == 0:
                stats.agent_evaluations = np.append(stats.agent_evaluations, np.random.random())

    def setUp(self):
        self.dummy_task_runs = []
        self.n_runs = 10
        self.evaluation_interval = 10
        self.task_names = ['a', 'b', 'c']

        for _ in range(self.n_runs):
            ls = LearningStats(self.evaluation_interval)
            self.__fill_stats(ls)
            self.dummy_task_runs.append(ls)

        self.dummy_curriculum_runs = []
        for _ in range(self.n_runs):
            self.dummy_curriculum_runs.append({})
            for task_name in self.task_names:
                ls = LearningStats(self.evaluation_interval)
                self.__fill_stats(ls)
                self.dummy_curriculum_runs[-1][task_name] = ls

    def test_plot_trajectories(self):
        try:
            plot_trajectories(self.dummy_task_runs)
            plot_trajectories([self.dummy_task_runs])
            plot_trajectories(self.dummy_curriculum_runs)
            plot_trajectories([self.dummy_curriculum_runs])
            plot_trajectories([self.dummy_curriculum_runs, self.dummy_task_runs],
                              time_domain=['steps', 'episodes'],
                              value_domain=['agent_evaluations', 'episode_rewards_moving_avg'],
                              includes_init_eval=[True, True],
                              show_std=[False, True],
                              task_trace_start=['max', 'most'],
                              show_run_traces=[False, True],
                              common_run_traces_start=[True, False],
                              as_separate_figs=True)
        except Exception as e:
            self.fail(f"plot_trajectories raised an error when it shouldn't: {e}")

    def test_plot_trajectories_saving(self):
        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            self.assertEqual(0, os.stat(tmpf.name).st_size) # file empty
            plot_trajectories(self.dummy_task_runs, save_path=tmpf.name)
            self.assertNotEqual(0, os.stat(tmpf.name).st_size) # file not empty
        finally:
            tmpf.close()

        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            self.assertEqual(0, os.stat(tmpf.name).st_size) # file empty
            plot_trajectories(self.dummy_task_runs, save_path=tmpf.name, save_format='html')
            self.assertNotEqual(0, os.stat(tmpf.name).st_size) # file not empty
        finally:
            tmpf.close()

        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            self.assertFalse(os.path.exists(tmpf.name + '_0.png'))
            self.assertFalse(os.path.exists(tmpf.name + '_1.png'))
            plot_trajectories(
                [self.dummy_task_runs, self.dummy_task_runs],
                as_separate_figs=True,
                save_path=tmpf.name,
                save_format='png')
            self.assertNotEqual(0, os.stat(tmpf.name + '_0.png').st_size) # file not empty
            self.assertNotEqual(0, os.stat(tmpf.name + '_1.png').st_size) # file not empty
        finally:
            tmpf.close()
            os.remove(tmpf.name + '_0.png')
            os.remove(tmpf.name + '_1.png')
            
    def test_plot_evaluation_impact(self):
        n_episodes_x = [10, 20, 30]
        task_runs_y = [self.dummy_task_runs] * len(n_episodes_x)
        try:
            plot_evaluation_impact(n_episodes_x, task_runs_y)
        except Exception as e:
            self.fail(f"plot_evaluation_impact raised an error when it shouldn't: {e}")

        with self.assertRaises(ValueError):
            plot_evaluation_impact(n_episodes_x, task_runs_y[:-1])

    def test_plot_evaluation_impact_saving(self):
        n_episodes_x = [10, 20, 30]
        task_runs_y = [self.dummy_task_runs] * len(n_episodes_x)
        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            self.assertFalse(os.path.exists(tmpf.name + '_evaluation_impact.png'))
            plot_evaluation_impact(n_episodes_x, task_runs_y, save_path=tmpf.name + '.png')
            self.assertNotEqual(0, os.stat(tmpf.name + '_evaluation_impact.png').st_size) # file not empty
        finally:
            tmpf.close()
            os.remove(tmpf.name + '_evaluation_impact.png')

        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            self.assertFalse(os.path.exists(tmpf.name + '_evaluation_impact.html'))
            plot_evaluation_impact(n_episodes_x, task_runs_y, save_path=tmpf.name, save_format='html')
            self.assertNotEqual(0, os.stat(tmpf.name + '_evaluation_impact.html').st_size) # file not empty
        finally:
            tmpf.close()
            os.remove(tmpf.name + '_evaluation_impact.html')

    def test_plot_evaluation_impact_2d(self):
        n_episodes_x = [10, 20, 30]
        n_episodes_y = [10, 20, 30]
        task_runs_z = [self.dummy_task_runs] * len(n_episodes_x)
        try:
            plot_evaluation_impact_2d(n_episodes_x, n_episodes_y, task_runs_z)
        except Exception as e:
            self.fail(f"plot_evaluation_impact_2d raised an error when it shouldn't: {e}")

        with self.assertRaises(ValueError):
            plot_evaluation_impact_2d(n_episodes_x, n_episodes_y, task_runs_z[:-1])
        with self.assertRaises(ValueError):
            plot_evaluation_impact_2d(n_episodes_x, n_episodes_y[:-1], task_runs_z)
        with self.assertRaises(ValueError):
            plot_evaluation_impact_2d(n_episodes_x[:-1], n_episodes_y, task_runs_z)

    def test_plot_evaluation_impact_2d_saving(self):
        n_episodes_x = [10, 20, 30]
        n_episodes_y = [10, 20, 30]
        task_runs_z = [self.dummy_task_runs] * len(n_episodes_x)
        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            self.assertFalse(os.path.exists(tmpf.name + '_evaluation_impact_2d.png'))
            plot_evaluation_impact_2d(
                n_episodes_x, n_episodes_y, task_runs_z, save_path=tmpf.name + '.png')
            self.assertNotEqual(0, os.stat(tmpf.name + '_evaluation_impact_2d.png').st_size) # file not empty
        finally:
            tmpf.close()
            os.remove(tmpf.name + '_evaluation_impact_2d.png')

        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            self.assertFalse(os.path.exists(tmpf.name + '_evaluation_impact_2d.html'))
            plot_evaluation_impact_2d(
                n_episodes_x, n_episodes_y, task_runs_z, save_path=tmpf.name + '.html', save_format='html')
            self.assertNotEqual(0, os.stat(tmpf.name + '_evaluation_impact_2d.html').st_size) # file not empty
        finally:
            tmpf.close()
            os.remove(tmpf.name + '_evaluation_impact_2d.html')

    def test_plot_time_impact(self):
        task_runs_x = [self.dummy_task_runs] * 5
        task_runs_y = [self.dummy_task_runs] * 5
        try:
            plot_time_impact(task_runs_x, task_runs_y)
        except Exception as e:
            self.fail(f"plot_time_impact raised an error when it shouldn't: {e}")

    def test_plot_time_impact_saving(self):
        task_runs_x = [self.dummy_task_runs] * 5
        task_runs_y = [self.dummy_task_runs] * 5
        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            self.assertFalse(os.path.exists(tmpf.name + '_time_impact.png'))
            plot_time_impact(task_runs_x, task_runs_y, save_path=tmpf.name + '.png')
            self.assertNotEqual(0, os.stat(tmpf.name + '_time_impact.png').st_size) # file not empty
        finally:
            tmpf.close()
            os.remove(tmpf.name + '_time_impact.png')

        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False)
            self.assertFalse(os.path.exists(tmpf.name + '_time_impact.html'))
            plot_time_impact(task_runs_x, task_runs_y, save_path=tmpf.name + '.html', save_format='html')
            self.assertNotEqual(0, os.stat(tmpf.name + '_time_impact.html').st_size) # file not empty
        finally:
            tmpf.close()
            os.remove(tmpf.name + '_time_impact.html')


if __name__ == '__main__':
    unittest.main()
