"""
Functions that can visualise statistics gathered from agents training through
:mod:`academia.curriculum` module.

Exported functions:

- :func:`plot_evaluation_impact`
- :func:`plot_time_impact`
- :func:`plot_multiple_evaluation_impact`
- :func:`plot_trajectories`

See Also:
    - :class:`academia.curriculum.LearningTask`
    - :class:`academia.curriculum.LearningStats`
    - :class:`academia.curriculum.LearningStatsAggregator`
    - :class:`academia.curriculum.Curriculum`
"""
import colorsys
from contextlib import contextmanager
import os
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from academia.curriculum import LearningStats, LearningStatsAggregator

TimeDomain = Literal['steps', 'episodes', 'wall_time', 'cpu_time']
ValueDomain = Literal[
    'agent_evaluations', 
    'episode_rewards', 
    'episode_rewards_moving_avg',
    'step_counts',
    'step_counts_moving_avg']
SaveFormat = Literal['png', 'html']
LearningTaskRuns = list[LearningStats]
CurriculumRuns = list[dict[str, LearningStats]]
Runs = Union[LearningTaskRuns, CurriculumRuns]
StartPoint = Literal['zero', 'mean', 'q3' 'most', 'outliers', 'max']


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


def plot_evaluation_impact(
        n_episodes_list: list[int], 
        task_runs_list: list[LearningTaskRuns],
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

    Returns:
        Absolute path to the saved plot file if ``save_path`` was provided.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the _evaluation_impact is added to the end of the ``save_path``
    
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

    agent_evaluations = []
    for task_runs in task_runs_list:
        mean_eval = np.mean([run.agent_evaluations[-1] for run in task_runs])
        agent_evaluations.append(mean_eval)

    if len(n_episodes_list) != len(task_runs_list):
        raise ValueError("The number of tasks at level x and level y should be equal.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_episodes_list,
        y=agent_evaluations,
    ))
    fig.update_layout(
        title="Impact of learning duration in task x to evaluation of task y",
        xaxis_title="Number of episodes in task x",
        yaxis_title="Evaluation score in task y"
    )
    # fig.update_traces(
    #     hovertemplate="<br>".join([
    #         "Number of episodes in task X: %{x}",
    #         "Evaluation score in task Y: %{y}"
    #     ])
    # )
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


def plot_evaluation_impact_2d(
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

@contextmanager
def create_figure(show: bool = False, 
                  save_path: Optional[str] = None, 
                  suffix: Optional[str] = None,
                  save_format: SaveFormat = 'png'):
    fig = go.Figure()
    yield fig
    
    if show:
        fig.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}_{suffix}.png")
        else:
            fig.write_html(f"{save_path}_{suffix}.html")
        return os.path.abspath(save_path)


def _get_color(
        n_shades: int = 1, 
        seed: Optional[int] = None,
        iter: Optional[int] = None,
        max_iters: Optional[int] = None):
    def hsv_to_hex(h, s, v):
        rgb = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
        return hex_color

    _rng = np.random.default_rng(seed)
    if iter is None or max_iters == 1:
        base_hue = _rng.random()
    else:
        base_hue = 1/max_iters * (iter % max_iters)
    base_color = (base_hue, _rng.uniform(0.8, 1.0), _rng.uniform(0.6, 0.8))
    shades: list = [base_color]

    for i in range(1, n_shades):
        h = np.clip(base_color[0] + _rng.uniform(-0.15, 0.15), 0, 1)
        s = np.clip(base_color[1] - 0.1 * i, 0.4, 1.0)
        b = np.clip(base_color[2] - 0.1 * i, 0.4, 1.0)
        shades.append((h, s, b))

    hex_colors = []
    for shade in shades:
        h, s, b = shade
        hex_colors.append(hsv_to_hex(h, s, b))
    
    return hex_colors


def _get_task_time_offset(
        task_trace_start: StartPoint, 
        time_offsets: list[Union[float, int]]):
    if task_trace_start == 'zero':
        task_time_offset = 0
    elif task_trace_start == 'mean':
        task_time_offset = np.mean(time_offsets)
    elif task_trace_start == 'max':
        task_time_offset = np.max(time_offsets)
    elif task_trace_start == 'q3':
        task_time_offset = np.quantile(time_offsets, 0.75)
    elif task_trace_start == 'most':
        task_time_offset = np.quantile(time_offsets, 0.90)
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


def _add_trace(
        fig: 'go.Figure',
        x: npt.NDArray[Union[np.float32, np.int32]],
        y: npt.NDArray[Union[np.float32, np.int32]],
        color: Optional[str]=None,
        alpha: float=1.0,
        showlegend: bool=True,
        name: Optional[str] = None,
        **kwargs):
    """
    Add a single trace (single task run trajectory) to the figure
    """
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name=name,
        opacity=alpha, showlegend=showlegend,
        line=dict(color=color), **kwargs
    ))


def _add_std_region(
        fig: 'go.Figure', 
        timestamps: npt.NDArray[Union[np.float32, np.int32]], 
        values: npt.NDArray[np.float32],
        std: npt.NDArray[np.float32], 
        color: Optional[str]=None):
    _add_trace(fig, timestamps, values+std, showlegend=False, color=color)
    _add_trace(fig, timestamps, values-std, showlegend=False, color=color, fill='tonexty')
    # fig.add_trace(go.Scatter(
    #     x=timestamps, y=values+std, mode='lines', showlegend=False,
    #     line_color=color
    # ))
    # fig.add_trace(go.Scatter(
    #     x=timestamps, y=values-std, mode='lines', showlegend=False,
    #     fill='tonexty', line_color=color
    # ))


def _add_task_trajectory(
        fig: 'go.Figure', 
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
    _add_trace(fig, timestamps, values, color=color, name=name)
    
    if show_std:
        std, _ = agg.get_aggregate(time_domain, value_domain, 'std')
        _add_std_region(fig, timestamps, values, std, color='#bbbbbb')
    if not show_run_traces:
        return
    
    for i, run in enumerate(task_runs):
        agg = LearningStatsAggregator([run], includes_init_eval)
        values, timestamps = agg.get_aggregate(time_domain, value_domain)
        if common_run_traces_start:
            timestamps += task_time_offset
        else:
            timestamps += time_offsets[i]
        _add_trace(
            fig, timestamps, values, color=color, alpha=1/len(task_runs), showlegend=False)


def _add_curriculum_trajectory(
        fig: 'go.Figure', 
        curriculum_runs: list[dict[str, LearningStats]],
        time_domain: TimeDomain,
        colors: Optional[list[str]] = None,
        **kwargs):
    time_offsets = np.zeros(shape=len(curriculum_runs))
    for i, task_name in enumerate(curriculum_runs[0].keys()):
        task_runs = [run[task_name] for run in curriculum_runs]
        _add_task_trajectory(fig, 
                             task_runs, 
                             name=task_name, 
                             time_offsets=time_offsets, 
                             time_domain=time_domain, 
                             color=colors[i], 
                             **kwargs)

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
    
    if not isinstance(trajectories, list):
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

    with create_figure(show, save_path, save_format=save_format) as fig:
        for i, trajectory_kwargs in enumerate(iterate_kwargs()):
            trajectory = trajectories[i]
            if isinstance(trajectory[0], LearningStats):
                color = _get_color(1, i, i, len(trajectories))[0]
                _add_task_trajectory(fig, trajectory, color=color, **trajectory_kwargs)
            if isinstance(trajectory[0], dict):
                colors = _get_color(len(trajectory[0]), i, i, len(trajectories))
                _add_curriculum_trajectory(fig, trajectory, colors=colors, **trajectory_kwargs)
        fig.update_layout(
            xaxis_title=f"Timestamps ({kwargs['time_domain']})",
            yaxis_title=f"Values ({kwargs['value_domain']})"
        )
    

def plot_evaluation_impact(
        n_episodes_x: list[int], 
        task_runs_y: list[LearningTaskRuns],
        show: bool = False,
        save_path: Optional[str] = None, 
        save_format: SaveFormat = 'png',
        ):
    
    if len(n_episodes_x) != len(task_runs_y):
        raise ValueError("The number of tasks at level x and level y should be equal.")

    agent_evaluations = []
    for task_runs in task_runs_y:
        mean_eval = np.mean([run.agent_evaluations[-1] for run in task_runs])
        agent_evaluations.append(mean_eval)

    with create_figure(show, save_path, save_format=save_format) as fig:
        _add_trace(fig, n_episodes_x, agent_evaluations)
        fig.update_layout(
            title="Impact of learning duration in task x on evaluation of task y",
            xaxis_title="Number of episodes in task x",
            yaxis_title="Evaluation score in task y"
        )


def plot_evaluation_impact_2d(
        n_episodes_x: list[int], 
        n_episodes_y: list[int],
        task_runs_z: list[LearningTaskRuns], 
        show: bool = False, 
        save_path: str = None,
        save_format: SaveFormat = 'png'):
    
    if len(n_episodes_x) != len(n_episodes_y) or len(n_episodes_x) != len(task_runs_z):
        raise ValueError("The number of tasks at level x, level y and level z should be equal.")
    
    agent_evaluations = []
    for task_runs in task_runs_z:
        mean_eval = np.mean([run.agent_evaluations[-1] for run in task_runs])
        agent_evaluations.append(mean_eval)

    with create_figure(show, save_path, save_format=save_format) as fig:
        fig.add_trace(go.Scatter(
            x=n_episodes_x,
            y=n_episodes_y,
            color=agent_evaluations,
            color_continuous_scale='Greens',
            labels={'color': 'Evaluation score in task z'},
        ))
        fig.update_layout(
            title="Impact of learning duration in task x and task y on evaluation of task z",
            xaxis_title="Number of episodes in task x",
            yaxis_title="Number of episodes in task x"
        )


def plot_time_impact(
        task_runs_x: list[LearningTaskRuns], 
        task_runs_y: list[LearningTaskRuns],
        time_domain_x: TimeDomain = "episodes",
        time_domain_xy: Union[TimeDomain, Literal["as_x"]] = "as_x",
        show: bool = False, 
        save_path: str = None, 
        save_format: SaveFormat = 'png'):
    
    if len(task_runs_x) != len(task_runs_y):
        raise ValueError("The number of tasks at level x and level y should be equal.")
    if time_domain_xy == "as_x":
        time_domain_xy = time_domain_x

    x_times = []
    for task_runs in task_runs_x:
        x_times.append(np.mean([_get_time_data(task_stats, time_domain_x) for task_stats in task_runs]))
    xy_times = []
    for task_runs in zip(task_runs_x, task_runs_y):
        x_time = np.mean([_get_time_data(task_stats, time_domain_xy) for task_stats in task_runs[0]])
        y_time = np.mean([_get_time_data(task_stats, time_domain_xy) for task_stats in task_runs[1]])
        xy_times.append(x_time + y_time)

    with create_figure(show, save_path, save_format=save_format) as fig:
        _add_trace(fig, x_times, xy_times)
        fig.update_layout(
            xaxis_title=f"Learning duration in task X ({time_domain_x})",
            yaxis_title=f"Total time spent in both tasks ({time_domain_xy})",
            title="Impact of learning duration in task x on total time spent in both tasks"
        )
