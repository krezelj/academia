import unittest
import tempfile
import os
import copy

import numpy as np

from academia.curriculum import LearningStats
from academia.tools.visualizations import (
    plot_task,
    plot_rewards_curriculum,
    plot_trajectory_curriculum,
    plot_curriculum_vs_nocurriculum,
    plot_evaluation_impact,
    plot_time_impact,
    plot_multiple_evaluation_impact
)


class TestVisualizations(unittest.TestCase):
    def setUp(self):
        # Dummy LearningStats objects for testing
        self.dummy_stats_x = LearningStats(evaluation_interval=5)
        self.dummy_stats_x.agent_evaluations = np.array([-184.66])
        self.dummy_stats_x.episode_rewards = np.array([-262.32, -196.48, -599.60, -439.23, -302.76])
        self.dummy_stats_x.step_counts = np.array([111.0, 98.0, 89.0, 70.0, 87.0])
        self.dummy_stats_x.episode_rewards_moving_avg = np.array([-262.32, -229.40, -352.80, -374.41, -360.08])
        self.dummy_stats_x.step_counts_moving_avg = np.array([111.0, 104.5, 99.33, 92.0, 91.0])
        self.dummy_stats_x.episode_wall_times = np.array([0.122, 0.1381481, 0.13, 0.14, 0.13])
        self.dummy_stats_x.episode_cpu_times = np.array([0.14, 0.28, 0.25, 0.17, 0.26])

        # Dummy Stats Y
        self.dummy_stats_y = LearningStats(evaluation_interval=5)
        self.dummy_stats_y.agent_evaluations = np.array([-150.66])
        self.dummy_stats_y.episode_rewards = np.array([-100.11, -150.22, -200.33, -250.44, -300.55])
        self.dummy_stats_y.step_counts = np.array([50.0, 45.0, 40.0, 35.0, 30.0])
        self.dummy_stats_y.episode_rewards_moving_avg = np.array([-100.11, -125.17, -150.22, -200.33, -250.44])
        self.dummy_stats_y.step_counts_moving_avg = np.array([50.0, 47.5, 45.0, 42.5, 40.0])
        self.dummy_stats_y.episode_wall_times = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
        self.dummy_stats_y.episode_cpu_times = np.array([0.06, 0.08, 0.07, 0.09, 0.1])

        # Dummy Stats Z
        self.dummy_stats_z = LearningStats(evaluation_interval=5)
        self.dummy_stats_z.agent_evaluations = np.array([-160.78])
        self.dummy_stats_z.episode_rewards = np.array([-500.1, -450.2, -400.3, -350.4, -300.5])
        self.dummy_stats_z.step_counts = np.array([70.0, 65.0, 60.0, 55.0, 50.0])
        self.dummy_stats_z.episode_rewards_moving_avg = np.array([-500.1, -475.15, -450.2, -400.3, -350.4])
        self.dummy_stats_z.step_counts_moving_avg = np.array([70.0, 67.5, 65.0, 62.5, 60.0])
        self.dummy_stats_z.episode_wall_times = np.array([0.12, 0.14, 0.13, 0.15, 0.12])
        self.dummy_stats_z.episode_cpu_times = np.array([0.13, 0.15, 0.14, 0.16, 0.13])

    def test_plot_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot_task")
            sut_png = plot_task(self.dummy_stats_x, show=False, save_path=save_path, save_format="png")
            self.assertTrue(os.path.exists(sut_png + "_rewards.png"))
            self.assertTrue(os.path.exists(sut_png + "_steps.png"))
            self.assertTrue(os.path.exists(sut_png + "_evaluations.png"))

            sut_html = plot_task(self.dummy_stats_x, show=False, save_path=save_path, save_format="html")
            self.assertTrue(os.path.exists(sut_html + "_rewards.html"))
            self.assertTrue(os.path.exists(sut_html + "_steps.html"))
            self.assertTrue(os.path.exists(sut_html + "_evaluations.html"))

    def test_plot_rewards_curriculum(self):
        curriculum_stats = {"X": self.dummy_stats_x, "Y": self.dummy_stats_y}
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot_rewards_curriculum")
            sut_png = plot_rewards_curriculum(curriculum_stats, show=False, save_path=save_path, save_format="png")
            self.assertTrue(os.path.exists(sut_png + "_rewards_curriculum.png"))

            sut_html = plot_rewards_curriculum(curriculum_stats, show=False, save_path=save_path, save_format="html")
            self.assertTrue(os.path.exists(sut_html + "_rewards_curriculum.html"))

    def test_plot_trajectory_curriculum(self):
        curriculum_stats = {"X": self.dummy_stats_x, "Y": self.dummy_stats_y}
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot_trajectory_curriculum")
            sut_png = plot_trajectory_curriculum(curriculum_stats, show=False, save_path=save_path)
            self.assertTrue(os.path.exists(sut_png + "_curriculum_eval_trajectory.png"))

            sut_html = plot_trajectory_curriculum(curriculum_stats, show=False, save_path=save_path, save_format="html")
            self.assertTrue(os.path.exists(sut_html + "_curriculum_eval_trajectory.html"))

    def test_plot_curriculum_vs_nocurriculum(self):
        curriculum_stats = {"X": self.dummy_stats_x, "Y": self.dummy_stats_y}
        nocurriculum_stats = self.dummy_stats_z
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot_curriculum_vs_nocurriculum")
            sut_png = plot_curriculum_vs_nocurriculum(curriculum_stats, nocurriculum_stats, show=False,
                                                      save_path=save_path, save_format="png")
            self.assertTrue(os.path.exists(sut_png + "_curriculum_vs_no_curriculum.png"))

            sut_html = plot_curriculum_vs_nocurriculum(curriculum_stats, nocurriculum_stats, show=False,
                                                       save_path=save_path, save_format="html")
            self.assertTrue(os.path.exists(sut_html + "_curriculum_vs_no_curriculum.html"))

        # includes_init_eval = False but initial evaluation
        with self.assertRaises(ValueError):
            # now steps_to_eval has smaller length than evaluations
            longer_evaluations = np.append(curriculum_stats["X"].agent_evaluations, [145.56])
            curr_stats_invalid = copy.copy(curriculum_stats)
            curr_stats_invalid["X"].agent_evaluations = longer_evaluations
            plot_curriculum_vs_nocurriculum(curr_stats_invalid, nocurriculum_stats, show=False, save_path=save_path,
                                            save_format="png")

        # includes_init_eval = True but no initial evaluation
        with self.assertRaises(ValueError):
            plot_curriculum_vs_nocurriculum(curriculum_stats, nocurriculum_stats, show=False, save_path=save_path,
                                            save_format="png",
                                            includes_init_eval=True)

    def test_plot_evaluation_impact(self):
        num_of_episodes_x = [10, 20, 30]
        stats_y = [self.dummy_stats_y] * len(num_of_episodes_x)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot_evaluation_impact")
            sut_png = plot_evaluation_impact(num_of_episodes_x, stats_y, show=False, save_path=save_path,
                                             save_format="png")
            self.assertTrue(os.path.exists(sut_png + "_evaluation_impact.png"))

            sut_html = plot_evaluation_impact(num_of_episodes_x, stats_y, show=False, save_path=save_path,
                                              save_format="html")
            self.assertTrue(os.path.exists(sut_html + "_evaluation_impact.html"))

        # now stats_lvl_y has smaller length than num_of_episodes_x
        with self.assertRaises(ValueError):
            shorter_stats_y = stats_y[:-1]
            plot_evaluation_impact(num_of_episodes_x, shorter_stats_y)

        # now evaluation hasn't been done only at the end of each task
        with self.assertRaises(ValueError):
            stats_y[0].agent_evaluations = np.array([-150.66, -160.78])
            plot_evaluation_impact(num_of_episodes_x, stats_y)

    def test_plot_time_impact(self):
        stats_x = [self.dummy_stats_x] * 3
        stats_y = [self.dummy_stats_y] * 3
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot_time_impact")
            sut_png = plot_time_impact(stats_x, stats_y, show=False, save_path=save_path, save_format="png")
            self.assertTrue(os.path.exists(sut_png + "_time_impact.png"))

            sut_html = plot_time_impact(stats_x, stats_y, show=False, save_path=save_path, save_format="html")
            self.assertTrue(os.path.exists(sut_html + "_time_impact.html"))

        # time domain was not allowed
        with self.assertRaises(ValueError):
            plot_time_impact(stats_x, stats_y, time_domain_x="seconds")

        # now stats_x has smaller length than stats_y
        with self.assertRaises(ValueError):
            shorter_stats_x = stats_x[:-1]
            plot_time_impact(shorter_stats_x, stats_y)

    def test_plot_multiple_evaluation_impact(self):
        num_of_episodes_x = [10, 20, 30]
        num_of_episodes_y = [5, 10, 15]
        stats_z = [self.dummy_stats_z] * len(num_of_episodes_x)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot_multiple_evaluation_impact")
            sut_png = plot_multiple_evaluation_impact(num_of_episodes_x, num_of_episodes_y, stats_z, show=False,
                                                      save_path=save_path, save_format="png")
            self.assertTrue(os.path.exists(sut_png + "_multiple_evaluation_impact.png"))

            sut_html = plot_multiple_evaluation_impact(num_of_episodes_x, num_of_episodes_y, stats_z, show=False,
                                                       save_path=save_path, save_format="html")
            self.assertTrue(os.path.exists(sut_html + "_multiple_evaluation_impact.html"))

        # now stats_z has smaller length than num_of_episodes_x
        with self.assertRaises(ValueError):
            shorter_stats_z = stats_z[:-1]
            plot_multiple_evaluation_impact(num_of_episodes_x, num_of_episodes_y, shorter_stats_z)

        # now num_of_episodes_x has smaller length than num_of_episodes_y
        with self.assertRaises(ValueError):
            shorter_num_of_episodes_y = num_of_episodes_y[:-1]
            plot_multiple_evaluation_impact(num_of_episodes_x, shorter_num_of_episodes_y, stats_z)

        # now evaluation hasn't been done only at the end of each task
        with self.assertRaises(ValueError):
            stats_z[0].agent_evaluations = np.array([-150.66, -160.78])
            plot_multiple_evaluation_impact(num_of_episodes_x, num_of_episodes_y, stats_z)

if __name__ == '__main__':
    unittest.main()
