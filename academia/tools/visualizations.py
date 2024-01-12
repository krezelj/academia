"""
Functions that can visualize statistics gathered from agents training through
:mod:`academia.curriculum` module.

Exported functions:

- :func:`create_figure`
- :func:`get_colors`
- :func:`plot_trajectories`
- :func:`plot_evaluation_impact`
- :func:`plot_evaluation_impact_2d`
- :func:`plot_time_impact`

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
import plotly.graph_objects as go

from academia.curriculum import LearningStats, LearningStatsAggregator

TimeDomain = Literal['steps', 'episodes', 'wall_time', 'cpu_time']
ValueDomain = Literal[
    'agent_evaluations', 
    'episode_rewards', 
    'episode_rewards_moving_avg',
    'step_counts',
    'step_counts_moving_avg']
SaveFormat = Literal['png', 'html', 'svg']
LearningTaskRuns = list[LearningStats]
CurriculumRuns = list[dict[str, LearningStats]]
Runs = Union[LearningTaskRuns, CurriculumRuns]
TimeAggFunc = Literal['zero', 'mean', 'q3', 'most', 'max', 'std']


@contextmanager
def create_figure(
    title: Optional[str] = None,
    show: bool = False, 
    save_path: Optional[str] = None, 
    suffix: Optional[str] = None,
    save_format: SaveFormat = 'png'):
    """
    Context manager that simplifies boilerplate code needed in all ``plot`` functions.
    It should also be used to create figures with a consistent style.

    Args:
        title: Figure title. Deafults to ``None``.
        show: Whether to display the plot. Defaults to ``False``.
        save_path: A path where the plot will be saved. The plot will not be
            saved if this is set to ``None``. Defaults to ``None``.
        suffix: A suffix appended at the end of the file. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Yields:
        the created plotly figure object.

    Example:

        >>> with create_figure("Test", True, './test', 'curr_comparison', 'png') as fig:
        >>>     fig.add_trace(...)

        This snippet will create a fig titled "Test", add a trace to it, then show it and save it to
        a specified file with a ``"_curr_comparison"`` suffix.
    """
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        title=dict(
            text=title,
            x=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0
        ),
        font=dict(
            size=12,
        )
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"
    yield fig
    
    if show:
        fig.show()
    if save_path:
        if not save_path.endswith(save_format):
            save_path += f'{suffix}.{save_format}'
        else:
            save_path = f"{'.'.join(save_path.split('.')[:-1])}{suffix}.{save_format}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png' or save_format == 'svg':
            fig.write_image(save_path)
        else:
            fig.write_html(save_path)


def _get_domain_display_name(
        domain: Union[TimeDomain, ValueDomain, list[TimeDomain], list[ValueDomain]]):
    if isinstance(domain, list):
        return "/".join([_get_domain_display_name(domain[i]) for i in range(len(domain))])
    if domain == 'agent_evaluations':
        return "Evaluation Score"
    if domain == 'episode_rewards':
        return "Episode Reward"
    if domain == 'episode_rewards_moving_avg':
        return "Episode Reward"
    if domain == 'step_counts':
        return "Step Count"
    if domain == 'step_counts_moving_avg':
        return "Step Count"
    if domain == 'steps':
        return "Step Count"
    if domain == 'episodes':
        return "Episodes"
    if domain == 'wall_time':
        return "Wall Time (seconds)"
    if domain == 'cpu_time':
        return "CPU Time"
    raise ValueError(f"Invalid domain: {domain}")


def get_colors(n_shades: int = 1, query: int = 1, n_queries: int = 1):
    """
    Generates a set of colors that makes different trajectories as distinguishable
    as possible while maintaining color similarity within a single curriculum trajectory.

    Args:
        n_shades: Number of shades to generate. For a single task it should be ``1``.
            For curriculum is should be set to the number of tasks inside the curriculum.
            Defaults to ``1``.
        query: Number of the query. When generating colors for many trajectories on a single plot it should
            be a number between ``1`` and ``n_queries`` identyfing the trajectory.
        n_queries: The number of expected queries to be performed. This information helps
            the function generate as distinguishable colors as possible.

    Returns:
        a list of hex colors.

    Raises:
        ValueError: If ``query`` value is invalid. Must be a positive integer less than ``n_queries``.

    Example:

        Generating colors for two trajectories. One trajectory is a task 
        and the other is a curriculum consiting of three tasks:

        >>> task_runs: list[LearningStats] = ...
        >>> curriculum_run: list[dict[str, LearningStats]] = ... # 3 tasks in a curriculum
        >>>
        >>> # we have two trajectories so we set n_queries=2
        >>> # this is the first trajectory so query=1
        >>> # it is also a trajectory for a single task so we set n_shades=1
        >>> task_color = get_colors(n_shades=1, query=1, n_queries=2)[0] # take first element
        >>>
        >>> # this is the second trajectory so query=2 (n_queries is still 2)
        >>> # it is also a curriculum with three tasks so we set n_shades=3
        >>> curriculum_colors = get_colors(n_shades=3, query=2, n_queries=2)
    
    """
    def hsv_to_hex(h, s, v):
        rgb = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
        return hex_color
    def wrap_hue(h: float):
        if h < 0: return h + 1
        if h > 1: return h - 1
        return h

    if query > n_queries or query < 1 or not isinstance(query, int):
        raise ValueError("Invalid query number. Must be a positive integer less than ``n_queries``")

    base_hue = query/n_queries
    hue_range = 1/(3*n_queries)
    hue_start = base_hue - hue_range

    shades = []
    for i in range(1, n_shades + 1):
        t = i/(n_shades + 1)
        h = wrap_hue((t * 2 * hue_range) + hue_start)
        s = -t * (t - 1) + 0.75 # sample a parabola
        b = t * (t - 1) + 1 # sample a different parabola
        shades.append(hsv_to_hex(h, s, b))
    return shades


def _get_time_offset_aggregate(
        time_agg_func: TimeAggFunc, 
        time_offsets: list[Union[float, int]],
        ignore_outliers: bool):
    """
    Returns the aggregate of time offsets based on the chosen ``time_agg_func``.
    """
    if ignore_outliers:
        q3 = np.quantile(time_offsets, 0.75)
        q1 = np.quantile(time_offsets, 0.25)
        iqr = q3 - q1 
        cutoff = q3 + 1.5 * iqr
        non_outliers = [time_offset for time_offset in time_offsets if time_offset <= cutoff]
        time_offsets = non_outliers

    if time_agg_func == 'zero':
        return 0
    if time_agg_func == 'mean':
        return np.mean(time_offsets)
    if time_agg_func == 'max':
        return np.max(time_offsets)
    if time_agg_func == 'q3':
        return np.quantile(time_offsets, 0.75)
    if time_agg_func == 'most':
        return np.quantile(time_offsets, 0.90)
    if time_agg_func == 'std':
        return np.std(time_offsets)
    raise ValueError(f"Invalid time_offsets value: {time_offsets}")


def _get_total_time(
        task_stats: LearningStats,
        time_domain: TimeDomain):
    """
    Returns the total time in a given time domain for a specified :class:`LearningStats` object.
    """
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
        color: Optional[str] = None,
        **kwargs):
    """
    Adds a single trace to the figure. 
    This is a wrapper to simplify code since it's used a lot of times
    """
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',line=dict(color=color), **kwargs
    ))


def _add_std_region(
        fig: 'go.Figure', 
        timestamps: npt.NDArray[Union[np.float32, np.int32]], 
        values: npt.NDArray[np.float32],
        std: npt.NDArray[np.float32], 
        color: Optional[str] = None):
    """
    Adds ``std`` values plot to the figure
    """
    r, g, b = tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4))
    rgb_color = f'rgba({r}, {g}, {b}, 0.1)'

    _add_trace(fig, timestamps, values+std, showlegend=False, color=rgb_color)
    _add_trace(fig, timestamps, values-std, showlegend=False, color=rgb_color,
               fill='tonexty', fillcolor=rgb_color)


def _add_stop_time(fig: 'go.Figure',
                   time_offsets: list[Union[float, int]],
                   task_trace_start,
                   show_stop_time_std: bool,
                   ignore_outliers: bool,
                   color: Optional[str] = None,
                   **kwargs):
    """
    Adds a trajectory stop time indicator (and optionally an std region around it).
    """
    stop_time = _get_time_offset_aggregate(task_trace_start, time_offsets, ignore_outliers)
    fig.add_vline(x=stop_time, line_width=2, line_dash="dash", line_color=color)
    if show_stop_time_std:
        std = _get_time_offset_aggregate('std', time_offsets, ignore_outliers)
        fig.add_vrect(x0=stop_time - std, x1=stop_time + std, line_width=0, fillcolor=color, opacity=0.1)


def _add_task_trajectory(
        fig: 'go.Figure', 
        task_runs: list[LearningStats],
        task_trace_start: TimeAggFunc,
        time_domain: TimeDomain,
        value_domain: ValueDomain,
        show_std: bool,
        show_run_traces: bool,
        common_run_traces_start: bool,
        ignore_outliers: bool,
        color: Optional[str] = None,
        name: Optional[str] = None,
        time_offsets: Optional[list[Union[float, int]]] = None,
        **kwargs):
    """
    Adds a single task trajectory to the figure
    """
    if time_offsets is None:
        time_offsets = np.zeros(len(task_runs))
    task_time_offset = _get_time_offset_aggregate(task_trace_start, time_offsets, ignore_outliers)

    agg = LearningStatsAggregator(task_runs)
    values, timestamps = agg.get_aggregate(time_domain, value_domain)
    timestamps += task_time_offset
    _add_trace(fig, timestamps, values, color=color, name=name)
    
    if show_std:
        std, _ = agg.get_aggregate(time_domain, value_domain, 'std')
        _add_std_region(fig, timestamps, values, std, color=color)
    if not show_run_traces:
        return
    
    for i, run in enumerate(task_runs):
        agg = LearningStatsAggregator([run])
        values, timestamps = agg.get_aggregate(time_domain, value_domain)
        if common_run_traces_start:
            timestamps += task_time_offset
        else:
            timestamps += time_offsets[i]
        _add_trace(
            fig, timestamps, values, color=color, opacity=np.sqrt(1/(2*len(task_runs))), showlegend=False)


def _add_curriculum_trajectory(
        fig: 'go.Figure', 
        curriculum_runs: list[dict[str, LearningStats]],
        time_domain: TimeDomain,
        show_stop_time: bool,
        show_stop_time_std: bool,
        colors: Optional[list[str]] = None,
        **kwargs):
    """
    Adds a curriculum trajectory to the figure
    """
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

        for j, run in enumerate(curriculum_runs):
            time_offsets[j] += _get_total_time(run[task_name], time_domain)
    if show_stop_time:
        _add_stop_time(fig, time_offsets, show_stop_time_std=show_stop_time_std, color=colors[-1], **kwargs)


def plot_trajectories(
        runs: Union[Runs, list[Runs]],
        time_domain: Union[TimeDomain, list[TimeDomain]] = 'steps',
        value_domain: Union[ValueDomain, list[ValueDomain]] = 'agent_evaluations',
        show_std: Union[bool, list[bool]] = False,
        task_trace_start: Union[TimeAggFunc, list[TimeAggFunc]] = 'most',
        ignore_outliers: Union[bool, list[bool]] = True,
        show_run_traces: Union[bool, list[bool]] = False,
        common_run_traces_start: Union[bool, list[bool]] = True,
        show_stop_time: Union[bool, list[bool]] = False,
        show_stop_time_std: Union[bool, list[bool]] = False,
        as_separate_figs: bool = False,
        title: Optional[str] = None,
        show: bool = False,
        save_path: Optional[str] = None, 
        save_format: SaveFormat = 'png') -> Union['go.Figure', list['go.Figure']]:
    """
    Plots trajectories of specified task/curriculum runs.

    Args:
        runs: A list of task/curriculum stats or a list which elements are lists of task/curriculum stats
        time_domain: Time domain which will be used on the x-axis.
            Can be either one of ``'steps'``, ``'episodes'``, ``'wall_time'``, ``'cpu_time'``
            or a list of these values, one for each plotted trajectory. Defaults to ``'steps'``.
        value_domain: Value domain which will be used on the y-axis.
            Can be either one of ``'agent_evaluations'``, ``'episode_rewards'``, 
            ``'episode_rewards_moving_avg'``, ``'step_counts'``, ``'step_counts_moving_avg'``
            or a list of these values, one for each plotted trajectory. Defaults to ``'agent_evaluations'``.
        show_std: Whether to show standard deviation region around given trajectory.
            Can be either a single bool or a list of bools, one for each plotted trajectory.
            Defaults to ``False``.
        task_trace_start: Point at which the next task in curriculum should start.
            Only used with curriculum trajectories. Possible values are:

            - ``'zero'`` - Each task trajectory will start at x=0,
            - ``'mean'`` - Each task trajectory will start at the mean termination point of
              previous trajectories,
            - ``'q3'`` - Each task trajectory will start at the third quantile of previous
              trajectories' termination points,
            - ``'most'`` - Each task trajectory will start at 0.9 quantile of previous
              trajectories' termiantion points,
            - ``'max'`` - Each task trajectory will start at the max termination point of
              previous trajectories.

            Can either be a single value or a list of values, one for each plotted curriculum trajectory.
            Defaults to ``'most'``.
        ignore_outliers: Whether to ignore outlying trajectories when calculating time aggregations
            (e.g. task trace starts or trajectory stop times) Can either be a single value or a list
            of values, one for each plotted curriculum trajectory. Default to ``True``.
        show_run_traces: Whether to show individual run traces (with lower opacity)
            Can either be a single value or a list of values, one for each plotted trajectory.
            Defaults to ``False``.
        common_run_traces_start: Whether individual run traces should start at the same point (the same
            as their parent trajectory) or as continuations of their predecessors (when plotting
            curriculum trajectory). Can either be a single value or a list of values, 
            one for each plotted curriculum trajectory. Defaults to ``False``.
        show_stop_time: Whether to show a vertical line indicating when a specific trajectory has stopped.
            It uses ``task_trace_start`` parameter and is placed in the same place a next task would start
            if it existed. Defaults to ``False``.
        show_stop_time_std: Whether to show standard deviation region for trajectory stop time.
            Defaults to ``False``.
        as_separate_figs: Whether each trajectory should be plotted on a separate figure. If set to ``True``
            and ``save_path`` is not ``None`` each figure will be saved in a file with a ``_i`` suffix
            corresponding to the position of the runs stats object in ``runs`` list. Defaults to ``False``.
        title: Figure title. Deafults to ``None``.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: A path where the plot will be saved. The plot will not be
            saved if this is set to ``None``. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Returns:
        a plotly figure object or a list of plotly figures is ``as_separate_figs`` is ``True``.

    Raises:
        ValueError: If ``time_domain`` is invalid
        ValueError: If ``value_domain`` is invalid
        ValueError: If ``task_trace_start`` is invalid
    
    Examples:

        The following imports are needed for all the examples:

        >>> from academia.curriculum import LearningTask, Curriculum, LearningStats
        >>> import academia.tools.visualizations as vis

        For details on how to configure agents see :mod:`academia.agents`.

        Plotting evaluations from a single task run:

        >>> task: LearningTask = ...
        >>> agent = ...
        >>> task.run(agent)
        >>> stats: LearningStats = task.stats
        >>> # note that a single stats object has to be passed inside a list
        >>> vis.plot_trajectories([stats])

        Plotting evaluations from a single curriculum run. See :mod:`academia.curriculum`
        for more details on how to configure and run a curriculum.

        >>> curriculum: Curriculum = ...
        >>> agent = ...
        >>> curriculum.run(agent)
        >>> stats: dict[str, LearningStats] = curriculum.stats
        >>> # note that a single stats object has to be passed inside a list
        >>> vis.plot_trajectories([stats])

        Plotting several runs of a single task with some additonal visuals:

        >>> task: LearningTask = ...
        >>> n_runs =  5
        >>> task_runs_stats: list[LearningStats] = []
        >>> for _ in range(n_runs):
        >>>     agent = ...
        >>>     task.run(agent)
        >>>     task_runs_stats.append(task.stats)
        >>> vis.plot_trajectories(
        >>>     task_runs_stats, 
        >>>     show_run_traces=True,
        >>>     show_std=True,
        >>> )

        Plotting curriculum vs task comparison:

        >>> task: LearningTask = ...
        >>> curriculum: Curriculum = ...
        >>> n_runs = 5
        >>> task_runs_stats: list[LearningStats] = []
        >>> curriculum_runs_stats: list[dict[str, LearningStats]] = []
        >>> for _ in range(n_runs):
        >>>     agent = ...
        >>>     task.run(agent)
        >>>     task_runs_stats.append(task.stats)
        >>>     agent = ...
        >>>     curriculum.run(agent)
        >>>     curriculum_runs_stats.append(curriculum.stats)
        >>> vis.plot_trajectories(
        >>>     [task_runs_stats, curriculum_runs_stats],
        >>>     show_std=True,
        >>>     show_run_traces=[True, False] # show individual traces only for the task
        >>> )

        Plotting curriculum vs curriculum as separate figures with different time domains:

        >>> curriculum_1: Curriculum = ...
        >>> curriculum_2: Curriculum = ...
        >>> n_runs = 5
        >>> curriculum_1_runs_stats: list[dict[str, LearningStats]] = []
        >>> curriculum_2_runs_stats: list[dict[str, LearningStats]] = []
        >>> for _ in range(n_runs):
        >>>     agent = ...
        >>>     curriculum_1.run(agent)
        >>>     curriculum_1_runs_stats.append(curriculum_1.stats)
        >>>     agent = ...
        >>>     curriculum_2.run(agent)
        >>>     curriculum_2_runs_stats.append(curriculum_2.stats)
        >>> vis.plot_trajectories(
        >>>     [curriculum_1_runs_stats, curriculum_2_runs_stats],
        >>>     as_separate_fig=True,
        >>>     time_domain=["steps", "episodes"]
        >>> )

        Plotting a running average of steps in episodes in several runs of a single task:

        >>> task: LearningTask = ...
        >>> n_runs =  5
        >>> task_runs_stats: list[LearningStats] = []
        >>> for _ in range(n_runs):
        >>>     agent = ...
        >>>     task.run(agent)
        >>>     task_runs_stats.append(task.stats)
        >>> vis.plot_trajectories(
        >>>     task_runs_stats,
        >>>     value_domain='step_counts_moving_avg'
        >>>     show_run_traces=True,
        >>> )
    """
    
    if not isinstance(runs[0], list):
        runs = [runs]

    # compile arguments of similar nature into a single dictionary
    # to be passed as kwargs later on
    trajectories_kwargs = {
        'time_domain': time_domain,
        'value_domain': value_domain,
        'show_std': show_std,
        'show_run_traces': show_run_traces,
        'task_trace_start': task_trace_start,
        'ignore_outliers': ignore_outliers,
        'common_run_traces_start': common_run_traces_start,
        'show_stop_time': show_stop_time,
        'show_stop_time_std': show_stop_time_std,
    }
    def _iterate_trajectories_kwargs():
        for i in range(len(runs)):
            trajectory_kwargs = {
                kwarg_name: trajectories_kwargs[kwarg_name][i] for kwarg_name in trajectories_kwargs}
            yield trajectory_kwargs

    # since we allow the user to either pass single values of a list of values (one for each
    # trajectory) we have to convert all non-list values to lists so that it's compatible
    # with the rest of the code
    for kwarg_name, value in trajectories_kwargs.items():
        if not isinstance(value, list):
            trajectories_kwargs[kwarg_name] = [value] * len(runs)

    if as_separate_figs:
        # recursively call plot_trajectories for each trajectory
        figs = []
        for i, trajectory_kwargs in enumerate(_iterate_trajectories_kwargs()):
            trajectory = runs[i]
            new_save_path = None if save_path is None else save_path + f'_{i}'
            fig = plot_trajectories(
                [trajectory], 
                as_separate_figs=False, 
                show=show, 
                save_path=new_save_path, 
                save_format=save_format,
                title=title,
                **trajectory_kwargs)
            figs.append(fig)
        return figs

    with create_figure(title, show, save_path, save_format=save_format) as fig:
        for i, trajectory_kwargs in enumerate(_iterate_trajectories_kwargs()):
            trajectory = runs[i]
            if isinstance(trajectory[0], LearningStats):
                color = get_colors(1, i + 1, len(runs))[0]
                _add_task_trajectory(fig, trajectory, color=color, **trajectory_kwargs)
            if isinstance(trajectory[0], dict):
                colors = get_colors(len(trajectory[0]), i + 1, len(runs))
                _add_curriculum_trajectory(fig, trajectory, colors=colors, **trajectory_kwargs)
        fig.update_layout(
            xaxis_title=f"{_get_domain_display_name(time_domain)}",
            yaxis_title=f"{_get_domain_display_name(value_domain)}"
        )
    return fig
    

def plot_evaluation_impact(
        n_episodes_x: list[int], 
        task_runs_y: list[LearningTaskRuns],
        title: Optional[str] = None,
        show: bool = False,
        save_path: Optional[str] = None, 
        save_format: SaveFormat = 'png',
        )  -> 'go.Figure':
    """
    Plots the impact of learning duration in task with difficulty level = x on evaluation
    in task with difficulty level = Y

    Args:
        n_episodes_x: Number of episodes in task with level X.
        task_runs_y: Learning statistics for tasks with level Y.
        title: Figure title. Deafults to ``None``.
        show: Whether to display the plot. Defaults to ``False``.
        save_path: A path where the plot will be saved. The plot will not be
            saved if this is set to ``None``. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Returns:
        a plotly figure object

    Raises:
        ValueError: If the number of tasks at level X and level Y is not equal. It is assumed that
            the number of tasks at level X and level Y is equal because the experiment involves testing
            the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
            the number of episodes spent in the easier one affects the evaluation of the agent in a more
            difficult environment.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the ``"_evaluation_impact"`` is added to the end of the ``save_path``
    
    Examples:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>>
        >>> n_episodes_x = [100, 500, 1000]
        >>> task_runs_y: list[list[LearningStats]] = []
        >>> n_runs = 5
        >>>
        >>> for nex in n_episodes_x:
        >>>     curriculum = Curriculum([
        >>>         LearningTask(LavaCrossing, {'difficulty': 0}, {'max_episodes': nex})
        >>>         LearningTask(LavaCrossing, {'difficulty': 1}, {'max_episodes': 1000})
        >>>     ])
        >>>     final_task_runs: list[LearningStats] = []
        >>>     for _ in range(n_runs)
        >>>         agent = ...  
        >>>         curriculum.run(agent)
        >>>         final_task_runs.append(curriculum.stats['2'])
        >>>     task_runs_y.append(final_task_runs)
        >>>
        >>> import academia.tools.visualizations as vis
        >>> vis.plot_evaluation_impact(n_episodes_x, task_runs_y)
    """
    
    if len(n_episodes_x) != len(task_runs_y):
        raise ValueError("The number of tasks at level X and level Y should be equal.")

    agent_evaluations = []
    for task_runs in task_runs_y:
        mean_eval = np.mean([run.agent_evaluations[-1] for run in task_runs])
        agent_evaluations.append(mean_eval)

    with create_figure(title, show, save_path, 'evaluation_impact', save_format) as fig:
        _add_trace(fig, n_episodes_x, agent_evaluations)
        fig.update_layout(
            xaxis_title="Number of episodes in task X",
            yaxis_title="Evaluation score in task Y"
        )
    return fig


def plot_evaluation_impact_2d(
        n_episodes_x: list[int], 
        n_episodes_y: list[int],
        task_runs_z: list[LearningTaskRuns],
        title: Optional[str] = None,
        show: bool = False, 
        save_path: str = None,
        save_format: SaveFormat = 'png') -> 'go.Figure':
    """
    Plots the impact of learning duration in task X and task Y to evaluation of task Z.
    See examples for more details

    Args:
        n_episodes_x: Number of episodes in task X.
        n_episodes_y: Number of episodes in task Y.
        task_runs_z: Learning statistics for tasks with level Z.
        title: Figure title. Deafults to ``None``.
        show: Whether to display the plot. Defaults to ``False``.
        save_path: A path where the plot will be saved. The plot will not be
            saved if this is set to ``None``. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.
    
    Returns:
        a plotly figure object

    Raises:
        ValueError: If the number of tasks at level X, level Y and level Z is not equal.
            It is assumed that the number of tasks at level X, level Y and level Z is equal
            because the experiment involves testing the curriculum on group of three tasks with 
            three specific levels of difficulty in order to examine how the number of episodes spent 
            in the easier ones affects the evaluation of the agent in the more difficult environment.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of
        the name of the saved plot, the ``"_eval_impact_2d"`` suffix is added to the end of the ``save_path``

    Examples:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>>
        >>> n_episodes_x = [100, 500]
        >>> n_episodes_y = [200, 400]
        >>> task_runs_z: list[list[LearningStats]] = []
        >>> n_runs = 5
        >>>
        >>> for nex in n_episodes_x:
        >>>     for ney in n_episodes_y:
        >>>         curriculum = Curriculum([
        >>>             LearningTask(LavaCrossing, {'difficulty': 0}, {'max_episodes': nex})
        >>>             LearningTask(LavaCrossing, {'difficulty': 1}, {'max_episodes': ney})
        >>>             LearningTask(LavaCrossing, {'difficulty': 2}, {'max_episodes': 1000})
        >>>         ])
        >>>         final_task_runs: list[LearningStats] = []
        >>>         for _ in range(n_runs)
        >>>             agent = ...  
        >>>             curriculum.run(agent)
        >>>             final_task_runs.append(curriculum.stats['3'])
        >>>         task_runs_z.append(final_task_runs)
        >>>
        >>> import academia.tools.visualizations as vis
        >>> # pairs of elements at the same indexes in the two lists below should
        >>> # cover all tested combinations, i.e. 100-200, 100-400, 500-200, 500-400
        >>> n_episodes_x = [100, 100, 500, 500]
        >>> n_episodes_y = [200, 400, 200, 400]
        >>> vis.plot_evaluation_impact_2d(n_episodes_x, n_episodes_y, task_runs_z)
    """

    if len(n_episodes_x) != len(n_episodes_y) or len(n_episodes_x) != len(task_runs_z):
        raise ValueError("The number of tasks at level X, level Y and level Z should be equal.")
    
    agent_evaluations = []
    for task_runs in task_runs_z:
        mean_eval = np.mean([run.agent_evaluations[-1] for run in task_runs])
        agent_evaluations.append(mean_eval)

    with create_figure(title, show, save_path, 'evaluation_impact_2d', save_format) as fig:
        fig.add_trace(go.Scatter(
            x=n_episodes_x,
            y=n_episodes_y,
            mode='markers',
            marker=dict(
                size=16,
                color=agent_evaluations,
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(
            xaxis_title="Number of episodes in task X",
            yaxis_title="Number of episodes in task Y"
        )
    return fig


def plot_time_impact(
        task_runs_x: list[LearningTaskRuns], 
        task_runs_y: list[LearningTaskRuns],
        time_domain_x: TimeDomain = "episodes",
        time_domain_y: Union[TimeDomain, Literal["as_x"]] = "as_x",
        title: Optional[str] = None,
        show: bool = False, 
        save_path: str = None, 
        save_format: SaveFormat = 'png') -> 'go.Figure':
    """
    Plots the impact of the number of episodes in task X on the total time spent in both tasks.
    See examples for more details

    Args:
        task_runs_x: Learning statistics for tasks with level X.
        task_runs_y: Learning statistics for tasks with level Y.
        time_domain_x: Time domain which will be used on the x-axis.
        time_domain_y: Time domain which will be used on the y-axis.
        title: Figure title. Deafults to ``None``.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: A path where the plot will be saved. The plot will not be
            saved if this is set to ``None``. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Returns:
        a plotly figure object

    Raises:
        ValueError: If the number of tasks at level X and level Y is not equal. It is assumed that
            the number of tasks at level X and level Y is equal because the experiment involves testing
            the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
            the number of episodes spent in the easier one affects the total time spent in both tasks.
        ValueError: If ``time_domain`` is invalid

    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of
        the name of the saved plot, the ``"_time_impact"`` suffix is added to the end of the ``save_path``

    Examples:
        
        >>> from academia.environments import LavaCrossing
        >>> from academia.curriculum import LearningTask, LearningStats, Curriculum
        >>>
        >>> wall_time_stops = [600, 900, 1200] # 10, 15 and 20 minutes
        >>> # the exact time at which the task stops might slightly differ
        >>> # from the set stop condition as it is only checked at the end of each episode
        >>> n_runs = 5
        >>> task_runs_x: list[list[LearningStats]] = []
        >>> task_runs_y: list[list[LearningStats]] = []
        >>>
        >>> for wts in wall_time_stops:
        >>>     curriculum = Curriculum([
        >>>         LearningTask(LavaCrossing, {'difficulty': 0}, {'max_wall_time': wts})
        >>>         LearningTask(LavaCrossing, {'difficulty': 1}, {'min_evaluation_score': 0.9})
        >>>     ])
        >>>     first_task_runs: list[LearningStats] = []
        >>>     second_task_runs: list[LearningStats] = []
        >>>     for _ in range(n_runs):
        >>>         agent = ...
        >>>         curriculum.run(agent)
        >>>         first_task_runs.append(curriculum.stats['1'])
        >>>         second_task_runs.append(curriculum.stats['2'])
        >>>     task_runs_x.append(first_task_runs)
        >>>     task_runs_y.append(second_task_runs)
        >>>
        >>> import academia.tools.visualizations as vis
        >>> vis.plot_time_impact(task_runs_x, task_runs_y, time_domain_x="wall_time")  
    """
    
    if len(task_runs_x) != len(task_runs_y):
        raise ValueError("The number of tasks at level X and level Y should be equal.")
    if time_domain_y == "as_x":
        time_domain_y = time_domain_x

    x_times = []
    for task_runs in task_runs_x:
        x_times.append(np.mean([_get_total_time(task_stats, time_domain_x) for task_stats in task_runs]))
    xy_times = []
    for task_runs in zip(task_runs_x, task_runs_y):
        x_time = np.mean([_get_total_time(task_stats, time_domain_y) for task_stats in task_runs[0]])
        y_time = np.mean([_get_total_time(task_stats, time_domain_y) for task_stats in task_runs[1]])
        xy_times.append(x_time + y_time)

    with create_figure(title, show, save_path, 'time_impact', save_format) as fig:
        _add_trace(fig, x_times, xy_times)
        fig.update_layout(
            xaxis_title=f"Learning duration in task X ({_get_domain_display_name(time_domain_x)})",
            yaxis_title=f"Total time spent in both tasks ({_get_domain_display_name(time_domain_y)})",
        )

    return fig
