"""
Functions that can visualise statistics gathered from agents training through
:mod:`academia.curriculum` module.

Exported functions:

- :func:`plot_task`
- :func:`plot_rewards_curriculum`
- :func:`plot_trajectory_curriculum`
- :func:`plot_curriculum_vs_nocurriculum`
- :func:`plot_evaluation_impact`
- :func:`plot_time_impact`
- :func:`plot_multiple_evaluation_impact`

See Also:
    - :class:`academia.curriculum.LearningTask`
    - :class:`academia.curriculum.Curriculum`
"""
import os
from typing import Literal, List, Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from academia.curriculum import LearningStats, LearningStatsAggregator

TimeDomain = Literal['steps', 'episodes', 'wall_time', 'cpu_time']
ValueDomain = Literal['agent_evaluations', 'episode_rewards']
SaveFormat = Literal['png', 'html']
LearningTaskRuns = list[LearningStats]
CurriculumRuns = list[dict[str, LearningStats]]
StartPoint = Literal['mean', 'q3' 'most', 'outliers', 'max']
Runs = Union[LearningTaskRuns, CurriculumRuns]


def plot_task(
        task_stats: LearningStats, 
        show: bool = False, 
        save_path: str = None,
        save_format: SaveFormat = 'png'):
    """
    Plots the learning statistics for a single task.

    The returned plots include:

    - A plot showing rewards and their moving average over episodes.
    - A plot displaying the steps taken by the agent in each episode against the episode numbers.
    - A plot indicating the agent's learning progress, represented by its evaluation score, in 
      relation to the number of steps taken up to the current evaluation.

    Note:
        If save path is provided, the all three plots will be saved to the specified path. To diferentiate between the plots,
        the file names will be appended with ``_rewards``, ``_steps`` and ``_evaluations`` respectively.

        if show is set to ``True``, the plots will be displayed in the browser window.   

    Args:
        task_stats: Learning statistics for the task.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Returns:
        Absolute path to the saved plot file if ``save_path`` was provided.

    Examples:
        Initialisation of a task we want to plot:

        >>> from academia.curriculum import LearningTask
        >>> from academia.environments import LunarLander 
        >>> from academia.tools.visualizations import plot_task
        >>> test_task = LearningTask(
        >>>     env_type= LunarLander,
        >>>     env_args={'difficulty': 2},
        >>>     stop_conditions={'max_episodes': 1000},
        >>>     evaluation_interval=100,
        >>>     stats_save_path='./my_task_stats.json',
        >>> )

        Running a task:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lunar_lander
        >>> agent = DQNAgent(
        >>>     n_actions=LunarLander.N_ACTIONS,
        >>>     nn_architecture=lunar_lander.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> test_task.run(agent, verbose=4)

        Plotting the task:

        >>> plot_task(test_task.stats, save_path='./test_task', save_format='png')
    """
    fig_rewards = px.line(x=np.arange(len(task_stats.episode_rewards)),
                          y=[task_stats.episode_rewards, task_stats.episode_rewards_moving_avg],
                          title='Episode Rewards and Moving Average')

    fig_rewards.update_layout(
        xaxis_title="Episode",
        yaxis_title="Score"
    )
    fig_rewards.data[0].name = 'Episode Rewards'
    fig_rewards.data[1].name = 'Moving Average'
    fig_rewards.update_traces(
        hovertemplate="<br>".join([
            "Episode: %{x}",
            "Reward: %{y}"
        ])
    )
    fig_steps = px.line(x=np.arange(len(task_stats.step_counts)),
                        y=task_stats.step_counts,
                        title='Steps per episode')
    fig_steps.update_layout(
        xaxis_title="Episode",
        yaxis_title="Steps"
    )

    evaluation_interval = task_stats.evaluation_interval
    steps_count = task_stats.step_counts
    steps_cum = np.cumsum(steps_count)
    indices = np.arange(evaluation_interval - 1, len(steps_cum), evaluation_interval)
    steps_to_eval = steps_cum[indices]
    #  Add 0 to the beginning of the array if the first evaluation is at the beginning of the task
    if len(steps_to_eval) < len(task_stats.agent_evaluations):
        steps_to_eval = np.concatenate([[0], steps_to_eval])

    fig_evaluations = px.line(x=steps_to_eval,
                              y=task_stats.agent_evaluations,
                              title='Agent evaluations')
    fig_evaluations.update_layout(
        xaxis_title="Total number of steps to evaluation",
        yaxis_title="Evaluation score"
    )
    fig_evaluations.update_traces(mode="markers+lines")

    if show:
        fig_rewards.show()
        fig_steps.show()
        fig_evaluations.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig_rewards.write_image(f"{save_path}_rewards.png")
            fig_steps.write_image(f"{save_path}_steps.png")
            fig_evaluations.write_image(f"{save_path}_evaluations.png")
        else:
            fig_rewards.write_html(f"{save_path}_rewards.html")
            fig_steps.write_html(f"{save_path}_steps.html")
            fig_evaluations.write_html(f"{save_path}_evaluations.html")
        return os.path.abspath(save_path)


def plot_rewards_curriculum(
        curriculum_stats: dict[str, LearningStats], 
        show: bool = False,
        save_path: str = None, 
        save_format: SaveFormat = 'png'):
    """
    Plots the trajectories of episode rewards for multiple tasks in the curriculum.

    Args:
        curriculum_stats: Learning statistics for multiple tasks in the curriculum.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Returns:
        Absolute path to the saved plot file if the ``save_path`` was provided.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the _rewards_curriculum is added to the end of the ``save_path`` 
    
    Examples:
        Initialisation of a curriculum we want to plot:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> curriculum = Curriculum(
        >>>     tasks=[task1, task2],
        >>>     output_dir='./my_curriculum/',
        >>> )

        Running a curriculum:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> agent = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> curriculum.run(agent, verbose=4)

        Plotting the curriculum:

        >>> from academia.tools.visualizations import plot_rewards_curriculum
        >>> plot_rewards_curriculum(curriculum.stats, save_path='./curriculum', save_format='png')
    """
    num_tasks = len(curriculum_stats)
    num_cols = 2
    num_rows = (num_tasks + 1) // num_cols

    fig = make_subplots(rows=num_rows,
                        cols=num_cols,
                        subplot_titles=[f'Episode rewards for task {task_id}'
                                        for task_id in curriculum_stats.keys()])

    row = 1
    col = 1
    for task_id, task_stats in curriculum_stats.items():
        rewards = task_stats.episode_rewards
        fig.add_trace(go.Scatter(y=rewards, mode='lines', name=f'Task {task_id}'), row=row, col=col)
        fig.update_xaxes(title_text='Step', row=row, col=col)
        fig.update_yaxes(title_text='Episode', row=row,
                         col=1)  # Only the first column has y-axis labels because they are shared across rows
        col += 1
        if col > num_cols:
            col = 1
            row += 1

    fig.update_layout(height=400 * num_rows, title_text='Trajectories of task rewards')
    fig.update_traces(
        hovertemplate="<br>".join([
            "Episode: %{x}",
            "Reward: %{y}"
        ])
    )

    if show:
        fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}_rewards_curriculum.png")
        else:
            fig.write_html(f"{save_path}_rewards_curriculum.html")
        return os.path.abspath(save_path)


def plot_trajectory_curriculum(
        curriculum_stats: dict[str, LearningStats], 
        show: bool = False,
        save_path: str = None, 
        save_format: SaveFormat = 'png'):
    """
    Plots the trajectories of agent evaluations for multiple tasks in the curriculum.

    On the X-axis we have the total number of steps to a given agent's evaluation, while on the Y-axis we have the 
    evaluation score obtained by the agent. The colors of the charts correspond to the appropriate tasks performed
    as part of the curriculum, which is described in the legend.

    Args:
        curriculum_stats: Learning statistics for multiple tasks in the curriculum.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Returns:
        Absolute path to the saved plot file if the ``save_path`` was provided.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the _curriculum_eval_trajectory is added to the end of the ``save_path`` 
    
    Examples:
        Initialisation of a curriculum we want to plot:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> curriculum = Curriculum(
        >>>     tasks=[task1, task2],
        >>>     output_dir='./my_curriculum/',
        >>> )

        Running a curriculum:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> agent = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> curriculum.run(agent, verbose=4)

        Plotting the curriculum:
        
        >>> from academia.tools.visualizations import plot_trajectory_curriculum
        >>> plot_trajectory_curriculum(curriculum.stats, save_path='./curriculum', save_format='png')
    """
    fig = go.Figure()
    total_steps_to_last_eval = 0
    for task_id, task_stats in curriculum_stats.items():
        evaluation_interval = task_stats.evaluation_interval
        evaluations = task_stats.agent_evaluations
        steps_count = task_stats.step_counts
        steps_count[0] += total_steps_to_last_eval
        steps_cum = np.cumsum(steps_count)
        indices = np.arange(evaluation_interval - 1, len(steps_cum), evaluation_interval)
        steps_to_eval = steps_cum[indices]
        if len(steps_to_eval) < len(evaluations):
            steps_to_eval = np.concatenate([[0], steps_to_eval])

        fig.add_trace(go.Scatter(x=steps_to_eval,
                                 y=evaluations,
                                 mode='lines',
                                 name=f'Task {task_id}'))

        total_steps_to_last_eval = steps_to_eval[-1]

    fig.update_layout(title_text='Curriculum evaluation trajectory',
                      xaxis_title='Total number of steps to evaluation',
                      yaxis_title='Evaluation score')
    fig.update_traces(
        hovertemplate="<br>".join([
            "Total num of steps to evaluation: %{x}",
            "Evaluation score: %{y}"
        ])
    )

    if show:
        fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}_curriculum_eval_trajectory.png")
        else:
            fig.write_html(f"{save_path}_curriculum_eval_trajectory.html")
        return os.path.abspath(save_path)


def plot_curriculum_vs_nocurriculum(
        curriculum_stats: dict[str, LearningStats],
        nocurriculum_stats: LearningStats, show: bool = False,
        includes_init_eval: bool = True,
        save_path: str = None, 
        save_format: SaveFormat = 'png',
        ):
    """
    Plots the comparison of curriculum learning with no curriculum learning.

    The chart is used to compare the agent's evaluation when teaching with a curriculum and without it. 
    The X-axis shows the total number of steps to a given agent's evaluation, while the Y-axis shows the 
    evaluation score obtained by the agent. The colors of the charts correspond to the appropriate tasks 
    performed as part of the curriculum and the task that is performed without the curriculum, which is 
    described in the legend.

    Args:
        curriculum_stats: Learning statistics for tasks in the curriculum.
        nocurriculum_stats: Learning statistics for the task without curriculum.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.
        includes_init_eval: Whether to include initial evaluation. Defaults to ``False``.
    
    Raises:
        ValueError: If the number of evaluations is greater than the number of steps to evaluation. 
            This means that the flag of includes_init_eval was set to ``False``, but the number of evaluations
            is greater than the number of steps to evaluation. This may be the problem if the flag was set to
            ``True`` in the LearningTask class.

        ValueError: If the number of evaluations is smaller than the number of steps to evaluation.
            This means that the flag of includes_init_eval was set to ``True``, but the number of evaluations
            is smaller than the number of steps to evaluation. This may be the problem if the flag was set to
            ``False`` in the LearningTask class.

    Returns:
        Absolute path to the saved plot file if the ``save_path`` was provided.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the _curriculum_vs_no_curriculum is added to the end of the ``save_path``

    Warning:
        Change the ``includes_init_eval`` to same value that you initialized in the 
        :class:`academia.curriculum.LearningTask` class. Otherwise, the function will raises an ``ValueError``.
    
    Examples:
        Initialisation of a curriculum we want to plot:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> curriculum = Curriculum(
        >>>     tasks=[task1, task2],
        >>>     output_dir='./my_curriculum/',
        >>> )

        Initialisation of a task without curriculum we want to plot:

        >>> no_curriculum = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 2},
        >>>     stop_conditions={'max_episodes': 1500},
        >>> )

        Defining an agent:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> agent_curriculum = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> agent_no_curriculum = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )

        Running a curriculum:

        >>> curriculum.run(agent_curriculum, verbose=4)

        Running a task without curriculum:

        >>> no_curriculum.run(agent_no_curriculum, verbose=4)

        Plotting the curriculum vs no curriculum:

        >>> from academia.tools.visualizations import plot_curriculum_vs_nocurriculum
        >>> plot_curriculum_vs_nocurriculum(curriculum.stats, 
        >>>                                 no_curriculum.stats, 
        >>>                                 save_path='./curriculum', 
        >>>                                 save_format='png')
    """
    fig = go.Figure()
    total_steps_to_last_eval = 0
    for task_id, task_stats in curriculum_stats.items():
        evaluation_interval = task_stats.evaluation_interval
        evaluations = task_stats.agent_evaluations
        steps_count = task_stats.step_counts
        steps_count[0] += total_steps_to_last_eval
        steps_cum = np.cumsum(steps_count)
        indices = np.arange(evaluation_interval - 1, len(steps_cum), evaluation_interval)
        steps_to_eval = steps_cum[indices]
        if includes_init_eval:
            if len(np.concatenate([[total_steps_to_last_eval], steps_to_eval])) > len(evaluations):
                raise ValueError(
                    f"The flag includes_init_eval is set to True, but the number of evaluations "
                    f"for task {task_id} is smaller than the number of steps to evaluation. "
                    f"Make sure that the flag was set to True in the LearningTask class."
                )
            fig.add_trace(go.Scatter(x=np.concatenate([[total_steps_to_last_eval], steps_to_eval]),
                                     y=evaluations, mode='lines', name=f'Task {task_id}'))
        else:
            if len(steps_to_eval) < len(evaluations):
                raise ValueError(
                    f"The flag includes_init_eval is set to False, but the number of evaluations "
                    f"for task {task_id} is greater than the number of steps to evaluation. "
                    f"Make sure that the flag was set to False in the LearningTask class."
                )
            fig.add_trace(go.Scatter(x=steps_to_eval, y=evaluations, mode='lines', name=f'Task {task_id}'))

        total_steps_to_last_eval = steps_to_eval[-1]
    nocurr_steps_cum = np.cumsum(nocurriculum_stats.step_counts)
    nocurr_indices = np.arange(evaluation_interval - 1, len(nocurr_steps_cum), evaluation_interval)
    no_curr_steps_to_eval = nocurr_steps_cum[nocurr_indices]

    if len(no_curr_steps_to_eval) < len(nocurriculum_stats.agent_evaluations):
        no_curr_steps_to_eval = np.concatenate([[0], no_curr_steps_to_eval])

    fig.add_trace(go.Scatter(x=no_curr_steps_to_eval,
                             y=nocurriculum_stats.agent_evaluations,
                             mode='lines',
                             name='No curriculum'
                             ))
    fig.update_layout(title_text='Curriculum vs No Curriculum',
                      xaxis_title='Total number of steps to evaluation',
                      yaxis_title='Evaluation score')
    fig.update_traces(
        hovertemplate="<br>".join([
            "Total num of steps to evaluation: %{x}",
            "Evaluation score: %{y}"
        ])
    )

    if show:
        fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}_curriculum_vs_no_curriculum.png")
        else:
            fig.write_html(f"{save_path}_curriculum_vs_no_curriculum.html")
        return os.path.abspath(save_path)


def plot_evaluation_impact(
        num_of_episodes_lvl_x: list[int], 
        stats_lvl_y: list[LearningStats],
        show: bool = False, 
        save_path: str = None,
        save_format: SaveFormat = 'png'):
    """
    Plots the impact of learning duration in task with difficulty level = x to evaluation 
    of task with difficulty level = y.

    The purpose of this plot is to show how the learning duration in task with difficulty level = x
    affects the evaluation of task with difficulty level = y. The X-axis shows the number of episodes
    in task with difficulty level = x, while the Y-axis shows the evaluation score obtained by the agent
    in task with difficulty level = y. 

    Args:
        num_of_episodes_lvl_x: Number of episodes in task X.
        stats_lvl_y: Learning statistics for tasks in level Y.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Raises:
        ValueError: If the number of tasks at level x and level y is not equal. It is assumed that 
            the number of tasks at level x and level y is equal because the experiment involves testing 
            the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
            the number of episodes spent in the easier one affects the evaluation of the agent in a more difficult 
            environment.

        ValueError: If the number of evaluation scores is not equal to the number of tasks at level x.
            This means that the evaluation was not performed only at the end of the task, which is necessary to
            correctly measure the impact of learning duration in task with difficulty level = x to evaluation of 
            this task.

    Returns:
        Absolute path to the saved plot file if ``save_path`` was provided.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the _evaluation_impact is added to the end of the ``save_path``

    Warning:
        It is important that evaluations in task with difficulty level = y
        are only performed at the end of the task and that the number of episodes in this task should be fixed
        to correctly measure the impact of learning duration in task with difficulty level = x to evaluation of 
        this task.     
    
    Examples:
        Initialisation of a diffrent pairs we want to analyze:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task0_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task1_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task0_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 700},
        >>> )
        >>> task1_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task0_v1000 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task1_v1000 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )

        Initialisation of agents:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> agent_v500 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> agent_v700 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> agent_v1000 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        
        Initialisation of a curriculums and running them:

        >>> curriculum_v500 = Curriculum(
        >>>     tasks=[task0_v500, task1_v500],
        >>>     output_dir='./curriculum_v500/',
        >>> )
        >>> curriculum_v500.run(agent_v500, verbose=4)
        >>> curriculum_v700 = Curriculum(
        >>>     tasks=[task0_v700, task1_v700],
        >>>     output_dir='./curriculum_v700/',
        >>> )
        >>> curriculum_v700.run(agent_v700, verbose=4)
        >>> curriculum_v1000 = Curriculum(
        >>>     tasks=[task0_v1000, task1_v1000],
        >>>     output_dir='./curriculum_v1000/',
        >>> )
        >>> curriculum_v1000.run(agent_v1000, verbose=4)

        Plotting the evaluation impact:

        >>> from academia.tools.visualizations import plot_evaluation_impact
        >>> plot_evaluation_impact([500, 700, 1000], 
        >>>                        [curriculum_v500.stats['2'], curriculum_v700.stats['2'], curriculum_v1000.stats['2']],
        >>>                         save_path='./evaluation_impact', 
        >>>                         save_format='png')
    """

    agent_evals_lvl_y = [value for task in stats_lvl_y for value in task.agent_evaluations.tolist()]

    if len(num_of_episodes_lvl_x) != len(stats_lvl_y):
        raise ValueError("The number of tasks at level x and level y should be equal.")

    if len(num_of_episodes_lvl_x) != len(agent_evals_lvl_y):
        raise ValueError(
            f"Agent evaluations should only be performed at the end of tasks with a level "
            f"of difficulty y."
        )

    fig = px.line(x=num_of_episodes_lvl_x, y=agent_evals_lvl_y,
                  title='Impact of learning duration in task x to evaluation of task y')
    fig.update_layout(
        xaxis_title="Number of episodes in task X",
        yaxis_title="Evaluation score in task Y"
    )
    fig.update_traces(
        hovertemplate="<br>".join([
            "Number of episodes in task X: %{x}",
            "Evaluation score in task Y: %{y}"
        ])
    )

    if show:
        fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}_evaluation_impact.png")
        else:
            fig.write_html(f"{save_path}_evaluation_impact.html")
        return os.path.abspath(save_path)


def plot_time_impact(
        stats_lvl_x: list[LearningStats], 
        stats_lvl_y: list[LearningStats],
        time_domain_x: Literal["steps", "episodes", "cpu_time", "wall_time"] = "episodes",
        time_domain_y: Literal["steps", "episodes", "cpu_time", "wall_time", "as_x"] = "as_x",
        show: bool = False, 
        save_path: str = None, 
        save_format: SaveFormat = 'png'):
    """
    Plots the impact of the number of episodes in task x on the total time spent in both tasks.

    The purpose of this plot is to show how the number of episodes in task x affects the total 
    time spent in both tasks. It is done by testing the curriculum on pairs of tasks with two
    specific levels of difficulty in order to examine how the number of episodes spent in the easier
    one affects the total time spent in both tasks when the stop condition in harder task is specified to reach 
    the fixed value of agent evaluation eg. equals 200.

    On the X-axis we have the number of episodes in task x, while on the Y-axis we have the total time spent in 
    both tasks.

    Args:
        stats_lvl_x: Learning statistics for tasks in level X.
        stats_lvl_y: Learning statistics for tasks in level Y.
        time_domain_x: Time domain over which time will be displayed on the X-axis.
        time_domain_y: Time domain over which time will be displayed on the Y-axis.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Raises:
        ValueError: If the number of tasks at level x and level y is not equal. It is assumed that 
            the number of tasks at level x and level y is equal because the experiment involves testing 
            the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
            the number of episodes spent in the easier one affects the total time spent in both tasks.

    Returns:
        Absolute path to the saved plot file if ``save_path`` was provided.

    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of
        the name of the saved plot, the _time_impact is added to the end of the ``save_path``

    Examples:
        Initialisation of a diffrent pairs we want to analyze:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task0_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task1_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )
        >>> task0_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 700},
        >>> )
        >>> task1_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )

        Initialisation of agents:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> agent_v500 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> agent_v700 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )

        Initialisation of a curriculums and running them:

        >>> curriculum_v500 = Curriculum(
        >>>     tasks=[task0_v500, task1_v500],
        >>>     output_dir='./curriculum_v500/',
        >>> )
        >>> curriculum_v500.run(agent_v500, verbose=4)
        >>> curriculum_v700 = Curriculum(
        >>>     tasks=[task0_v700, task1_v700],
        >>>     output_dir='./curriculum_v700/',
        >>> )
        >>> curriculum_v700.run(agent_v700, verbose=4)

        Plotting the time impact:

        >>> from academia.tools.visualizations import plot_time_impact
        >>> plot_time_impact([curriculum_v500.stats['1'], curriculum_v700.stats['1']], 
        >>>                  [curriculum_v500.stats['2'], curriculum_v700.stats['2']],
        >>>                   time_domain_x="steps", 
        >>>                   save_path='./time_impact', 
        >>>                   save_format='png')
    """
    if len(stats_lvl_x) != len(stats_lvl_y):
        raise ValueError("The number of tasks at level x and level y should be equal.")

    if time_domain_y == "as_x":
        time_domain_y = time_domain_x

    x_data, x_domain = _extract_time_data(stats_lvl_x, time_domain_x)

    y_data, y_domain = _extract_time_data(stats_lvl_y, time_domain_y)

    # we want to show total time on y-axis, so we need to add x_data to y_data
    y_data = np.sum([x_data, y_data], axis=0)

    fig = px.line(x=x_data, y=y_data, markers=True,
                  title='Impact of learning duration in task x on total time spent in both tasks')
    fig.update_layout(
        xaxis_title=f"Learning duration in task X ({x_domain})",
        yaxis_title=f"Total time spent in both tasks ({y_domain})"
    )
    fig.update_traces(
        hovertemplate="<br>".join([
            "Learning duration in task X: %{x}",
            "Total time spent in both tasks: %{y}"
        ])
    )

    if show:
        fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}_time_impact.png")
        else:
            fig.write_html(f"{save_path}_time_impact.html")
        return os.path.abspath(save_path)


def _extract_time_data(
        stats: list[LearningStats], time_domain: str):
    """
    Extracts the data from the learning statistics for the given time domain.

    Args:
        stats: Learning statistics for tasks in level X.
        time_domain: Time domain to extract data for.

    Returns:
        List of data for the given time domain.
    """
    if time_domain == "steps":
        return [np.sum(task.step_counts) for task in stats], "steps"
    elif time_domain == "episodes":
        return [len(task.step_counts) for task in stats], "episodes"
    elif time_domain == "cpu_time":
        return [np.sum(task.episode_cpu_times) for task in stats], "cpu_time"
    elif time_domain == "wall_time":
        return [np.sum(task.episode_wall_times) for task in stats], "wall_time"
    else:
        raise ValueError(f"Unknown time domain: {time_domain}")


def plot_multiple_evaluation_impact(
        num_of_episodes_lvl_x: list[int], 
        num_of_episodes_lvl_y: list[int],
        stats_lvl_z: list[LearningStats], 
        show: bool = False, 
        save_path: str = None,
        save_format: SaveFormat = 'png'):
    """
    Plots the impact of learning duration in task x and task y to evaluation of task z. The purpose of this plot is 
    to show how the learning duration in task x and task y affects the evaluation of task z. It is done by testing 
    the curriculum on group of three tasks with three specific levels of difficulty in order to examine how the number 
    of episodes spent in the easier ones affects the evaluation of the agent in a more difficult environment.

    Args:
        num_of_episodes_lvl_x: Number of episodes in task X.
        num_of_episodes_lvl_y: Number of episodes in task Y.
        stats_lvl_z: Learning statistics for tasks in level Z.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Raises:
        ValueError: If the number of tasks at level x, level y and level z is not equal. 
            It is assumed that the number of tasks at level x, level y and level z is equal 
            because the experiment involves testing the curriculum on group of three tasks with 
            three specific levels of difficulty in order to examine how the number of episodes spent 
            in the easier ones affects the evaluation of the agent in a more difficult environment.

        ValueError: If the number of evaluation scores is not equal to the number of tasks at level x and level y.
            It is assumed that the evaluation was not performed only at the end of the task, which is necessary to
            correctly measure the impact of learning duration in task x and task y to evaluation of task z.

    Returns:
        Absolute path to the saved plot file if ``save_path`` was provided.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of
        the name of the saved plot, the _multiple_evaluation_impact is added to the end of the ``save_path``

    Warning:
        It is important that evaluations in task with difficulty level = z
        are only performed at the end of the task and that the number of episodes in this task should be fixed
        to correctly measure the impact of learning duration in task with difficulty level = x and task with 
        difficulty level = y to evaluation of this task.
    
    Examples:

        Initialisation of a diffrent groups we want to analyze:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task_curr0_v0 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task_curr0_v1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task_curr0_v2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 2},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )
        >>> curriculum0 = Curriculum(
        >>>     tasks=[task_curr0_v0, task_curr0_v1, task_curr0_v2],
        >>>     output_dir='./curriculum0/',
        >>> )
        >>> task_curr1_v0 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 700},
        >>> )
        >>> task_curr1_v1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 1200},
        >>> )
        >>> task_curr1_v2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 2},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )
        >>> curriculum1 = Curriculum(
        >>>     tasks=[task_curr1_v0, task_curr1_v1, task_curr1_v2],
        >>>     output_dir='./curriculum1/',
        >>> )
        >>> task_curr2_v0 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task_curr2_v1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1},
        >>>     stop_conditions={'max_episodes': 600},
        >>> )
        >>> task_curr2_v2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 2},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )
        >>> curriculum2 = Curriculum(
        >>>     tasks=[task_curr2_v0, task_curr2_v1, task_curr2_v2],
        >>>     output_dir='./curriculum2/',
        >>> )

        Initialisation of agents:

        >>> from academia.agents import DQNAgent
        >>> from academia.utils.models import lava_crossing
        >>> agent0 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> agent1 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )
        >>> agent2 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=lava_crossing.MLPStepDQN,
        >>>     random_state=123,
        >>> )

        Running curriculums:

        >>> curriculum0.run(agent0, verbose=4)
        >>> curriculum1.run(agent1, verbose=4)
        >>> curriculum2.run(agent2, verbose=4)

        Plotting the multiple evaluation impact:

        >>> from academia.tools.visualizations import plot_multiple_evaluation_impact
        >>> plot_multiple_evaluation_impact([500, 700, 1000], [1000, 1200, 600], 
        >>>                                 [curriculum0.stats['3'], curriculum1.stats['3'], curriculum2.stats['3']],
        >>>                                 save_path='./multiple_evaluation_impact', 
        >>>                                 save_format='png')
    """
    agent_evals_lvl_z = [value for task in stats_lvl_z for value in task.agent_evaluations.tolist()]

    if len(num_of_episodes_lvl_x) != len(num_of_episodes_lvl_y) or \
            len(num_of_episodes_lvl_x) != len(stats_lvl_z) or \
            len(num_of_episodes_lvl_y) != len(stats_lvl_z):
        raise ValueError("The number of tasks at level x, level y and level z should be equal.")

    if len(num_of_episodes_lvl_x) != len(agent_evals_lvl_z) or len(num_of_episodes_lvl_y) != len(agent_evals_lvl_z):
        raise ValueError(
            f"Agent evaluations should only be performed at the end of tasks with a level "
            f"of difficulty z."
        )

    fig = px.scatter(x=num_of_episodes_lvl_x,
                     y=num_of_episodes_lvl_y,
                     color=agent_evals_lvl_z,
                     color_continuous_scale='Greens',
                     labels={'color': 'Evaluation score in task Z'},
                     text=np.round(agent_evals_lvl_z, 1),
                     title='Impact of learning duration in task x and task y to evaluation of task z'
                     )

    fig.update_traces(
        marker_size=48,
        hovertemplate="<br>".join(["Number of episodes in task X: %{x}",
                                   "Number of episodes in task Y: %{y}",
                                   "Evaluation score in task Z: %{marker.color:.3f}"
                                   ]),
        textfont_color='black'
    )

    fig.update_xaxes(title_text='Number of Episodes Level X')
    fig.update_yaxes(title_text='Number of Episodes Level Y')

    if show:
        fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}_multiple_evaluation_impact.png")
        else:
            fig.write_html(f"{save_path}_multiple_evaluation_impact.html")
        return os.path.abspath(save_path)


# ===================================== WORK IN PROGRESS =============================================


def _get_task_time_offset(
        task_trace_start: StartPoint, 
        time_offsets: list[Union[float, int]]):
    if task_trace_start == 'mean':
        task_time_offset = np.mean(time_offsets)
    elif task_trace_start == 'max':
        task_time_offset = np.max(time_offsets)
    elif task_trace_start == 'q3':
        task_time_offset = np.quantile(time_offsets, 0.75)
    elif task_trace_start == 'most':
        task_time_offset = np.quantile(time_offsets, 0.95)
    elif task_trace_start == 'outliers':
        q3 = np.quantile(time_offsets, 0.75)
        q1 = np.quantile(time_offsets, 0.25)
        iqr = q3 - q1 
        task_time_offset = np.minimum(q3 + 1.5 * iqr, np.max(time_offsets))
    return task_time_offset


def _get_time_data(
        task_stats: LearningStats,
        time_domain: TimeDomain):
    if time_domain == "steps":
        return np.sum(task_stats.step_counts)
    elif time_domain == "episodes":
        return len(task_stats.step_counts)
    elif time_domain == "cpu_time":
        return np.sum(task_stats.episode_cpu_times)
    elif time_domain == "wall_time":
        return np.sum(task_stats.episode_wall_times)
    else:
        raise ValueError(f"Unknown time domain: {time_domain}")


def _add_trace_trajectory(
        fig: 'go.Figure',
        values: npt.NDArray[np.float32],
        timestamps: npt.NDArray[Union[np.float32, np.int32]],
        color: Optional[str]=None,
        alpha: float=1.0,
        showlegend: bool=True,
        name: Optional[str] = None,):
    """
    Add a single trace (single task run trajectory) to the figure
    """
    color_rgba = color # add alpha

    fig.add_trace(go.Scatter(
        x=timestamps, y=values, mode='lines', name=name,
        opacity=alpha, showlegend=showlegend,
        line=dict(color=color_rgba)
    ))


def _add_std_region(
        fig: 'go.Figure', 
        values: npt.NDArray[np.float32],
        std: npt.NDArray[np.float32], 
        timestamps: npt.NDArray[Union[np.float32, np.int32]], 
        color: Optional[str]=None):
    fig.add_trace(go.Scatter(
        x=timestamps, y=values+std, mode='lines', showlegend=False,
        line_color=color
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=values-std, mode='lines', showlegend=False,
        fill='tonexty', line_color=color
    ))


def _add_task_trajectory(fig: 'go.Figure', 
                        task_runs: list[LearningStats],
                        task_trace_start: StartPoint,
                        includes_init_eval: bool,
                        time_domain: TimeDomain,
                        value_domain: ValueDomain,
                        show_std: bool,
                        show_run_traces: bool,
                        common_run_traces_start: bool,
                        color: Optional[str] = None,
                        name: Optional[str] = None,
                        time_offsets: Optional[list[Union[float, int]]] = None):
    """
    Add a single task trajectory to the figure
    """
    if time_offsets is None:
        time_offsets = np.zeros(len(task_runs))
    task_time_offset = _get_task_time_offset(task_trace_start, time_offsets)

    agg = LearningStatsAggregator(task_runs, includes_init_eval)
    values, timestamps = agg.get_aggregate(time_domain, value_domain)
    timestamps += task_time_offset
    _add_trace_trajectory(fig, values, timestamps, color=color, name=name)
    
    if show_std:
        std, _ = agg.get_aggregate(time_domain, value_domain, 'std')
        _add_std_region(fig, values, std, timestamps, color='#bbbbbb')
    if not show_run_traces:
        return
    
    for i, run in enumerate(task_runs):
        agg = LearningStatsAggregator([run], includes_init_eval)
        values, timestamps = agg.get_aggregate(time_domain, value_domain)
        if common_run_traces_start:
            timestamps += task_time_offset
        else:
            timestamps += time_offsets[i]
        _add_trace_trajectory(
            fig, values, timestamps, color=color, alpha=1/len(task_runs), showlegend=False)


def _add_curriculum_trajectory(fig: 'go.Figure', 
                              curriculum_runs: list[dict[str, LearningStats]],
                              time_domain: TimeDomain,
                              **kwargs):
    time_offsets = np.zeros(shape=len(curriculum_runs))
    for task_name in curriculum_runs[0].keys():
        task_runs = [run[task_name] for run in curriculum_runs]
        _add_task_trajectory(
            fig, task_runs, name=task_name, time_offsets=time_offsets, time_domain=time_domain, **kwargs)

        for i, run in enumerate(curriculum_runs):
            time_offsets[i] += _get_time_data(run[task_name], time_domain)


def plot_trajectories(
        trajectories: Union[Runs, list[Runs]],
        # time_domain: Union[TimeDomain, list[TimeDomain]] = 'steps',
        # value_domain: Union[ValueDomain, list[ValueDomain]] = 'agent_evaluations',
        # includes_init_eval: Union[bool, list[bool]] = True,
        # show_std: Union[bool, list[bool]] = False,
        # show_run_traces: Union[bool, list[bool]] = False,
        # task_trace_start: Union[StartPoint, list[StartPoint]] = 'most',
        # common_run_traces_start: Union[bool, list[bool]] = True,
        as_separate_figs: bool = False,
        show: bool = False,
        save_path: Optional[str] = None, 
        save_format: SaveFormat = 'png',
        **kwargs):
    
    if not isinstance(trajectories):
        trajectories = [trajectories]

    def iterate_kwargs():
        for i in range(len(trajectories)):
            trajectory_kwargs = {kwarg_name: kwargs[kwarg_name][i] for kwarg_name in kwargs}
            yield trajectory_kwargs
    
    # parse kwargs so that all are a list of values
    kwargs_kvp = {
        'time_domain': 'steps',
        'value_domain': 'agent_evaluations',
        'includes_init_eval': True,
        'show_std': False,
        'show_run_traces': False,
        'task_trace_start': 'most',
        'common_run_traces_start': True
    }
    for kwarg_name, value in kwargs_kvp.items():
        if kwarg_name not in kwargs:
            kwargs[kwarg_name] = value
        if not isinstance(kwargs[kwarg_name], list):
            kwargs[kwarg_name] = [kwargs[kwarg_name] for _ in range(len(trajectories))]

    if as_separate_figs:
        # recursively call plot_trajectories for each trajectory
        for i, trajectory_kwargs in enumerate(iterate_kwargs()):
            trajectory = trajectories[i]
            new_save_path = None if save_path is None else save_path + f'_{i}'
            plot_trajectories(
                [trajectory], 
                as_separate_figs=False, 
                show=show, 
                save_path=new_save_path, 
                save_format=save_format, 
                **trajectory_kwargs)

    fig = go.Figure()
    for i, trajectory_kwargs in enumerate(iterate_kwargs()):
        trajectory = trajectories[i]
        if isinstance(trajectory[0], LearningStats):
            _add_task_trajectory(fig, trajectory, **trajectory_kwargs)
        if isinstance(trajectory[0], dict):
            _add_curriculum_trajectory(fig, trajectory, **trajectory_kwargs)

    fig.update_layout(
        xaxis_title=f"Timestamps ({kwargs['time_domain']})",
        yaxis_title=f"Values ({kwargs['value_domain']})"
    )
    if show:
        fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}.png")
        else:
            fig.write_html(f"{save_path}.html")
        return os.path.abspath(save_path)