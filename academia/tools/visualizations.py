"""
Functions that can visualise statistics gathered from agents training through
:mod:`academia.curriculum` module.

Exported functions:

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
SaveFormat = Literal['png', 'html']
LearningTaskRuns = list[LearningStats]
CurriculumRuns = list[dict[str, LearningStats]]
Runs = Union[LearningTaskRuns, CurriculumRuns]
StartPoint = Literal['zero', 'mean', 'q3' 'most', 'outliers', 'max']


@contextmanager
def _create_figure(show: bool = False, 
                  save_path: Optional[str] = None, 
                  suffix: Optional[str] = None,
                  save_format: SaveFormat = 'png'):
    """
    Context manager that simplifies boilerplate code needed in all ``plot`` functions

    Example:

    >>> with create_figure(False, './test', 'curr_comparison', 'png') as fig:
    >>>     fig.add_trace(...)

    This snippet will create a fig, add a trace to it and then optionally show it and save it to
    a specified file with a specified suffix.
    """
    fig = go.Figure()
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"
    yield fig
    
    if show:
        fig.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_format == 'png':
            fig.write_image(f"{save_path}{suffix}.png")
        else:
            fig.write_html(f"{save_path}{suffix}.html")


def _get_colors(
        n_shades: int = 1, 
        seed: Optional[int] = None,
        query: Optional[int] = None,
        n_queries: Optional[int] = None):
    """
    Returns a list of ``n_shades`` colours that are similar to each other.

    If you know how many sets of shades you will query (i.e. how many times you will call this function
    for a single figure) you can specify it with ``n_queries`` and tell the function which ``query``
    it is in each call. Example

    >>> n_queries: int = 5
    >>> for i in range(n_queries):
    >>>     colors = _get_colors(2, None, i, n_queries)

    This way you ensure that all generated sets of shades are uniformly distributed on the Hue axis
    in the HSV colour encoding.
    """
    def hsv_to_hex(h, s, v):
        rgb = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
        hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
        return hex_color

    _rng = np.random.default_rng(seed)
    if query is None or n_queries == 1:
        base_hue = _rng.random()
    else:
        base_hue = 1/n_queries * (query % n_queries)
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
    """
    Returns the offset of the task trace starting point on the X axis
    based on the chosen ``task_trace_start``.
    """
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
        color: Optional[str]=None,
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
        color: Optional[str]=None):
    """
    Adds ``std`` values plot to the figure
    """
    _add_trace(fig, timestamps, values+std, showlegend=False, color=color)
    _add_trace(fig, timestamps, values-std, showlegend=False, color=color, fill='tonexty')


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
    Adds a single task trajectory to the figure
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
            fig, timestamps, values, color=color, opacity=1/len(task_runs), showlegend=False)


def _add_curriculum_trajectory(
        fig: 'go.Figure', 
        curriculum_runs: list[dict[str, LearningStats]],
        time_domain: TimeDomain,
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

        for i, run in enumerate(curriculum_runs):
            time_offsets[i] += _get_total_time(run[task_name], time_domain)


def plot_trajectories(
        runs: Union[Runs, list[Runs]],
        time_domain: Union[TimeDomain, list[TimeDomain]] = 'steps',
        value_domain: Union[ValueDomain, list[ValueDomain]] = 'agent_evaluations',
        includes_init_eval: Union[bool, list[bool]] = True,
        show_std: Union[bool, list[bool]] = False,
        task_trace_start: Union[StartPoint, list[StartPoint]] = 'most',
        show_run_traces: Union[bool, list[bool]] = False,
        common_run_traces_start: Union[bool, list[bool]] = True,
        as_separate_figs: bool = False,
        show: bool = False,
        save_path: Optional[str] = None, 
        save_format: SaveFormat = 'png'):
    """
    Plots trajectories of specified task/curriculum runs.

    Args:
        runs: A list of task/curriculum stats or a list which elements are lists of task/curriculum stats
        time_domain: Time domain which will be used on the X-axis.
            Can be either one of ``'steps'``, ``'episodes'``, ``'wall_time'``, ``'cpu_time'``
            or a list of these values, one for each plotted trajectory. Defaults to ``'steps'``.
        value_domain: Value domain which will be used on the Y-axis.
            Can be either one of ``'agent_evaluations'``, ``'episode_rewards'``, 
            ``'episode_rewards_moving_avg'``, ``'step_counts'``, ``'step_counts_moving_avg'``
            or a list of these values, one for each plotted trajectory. Defaults to ``'agent_evaluations'``.
        includes_init_eval: Whether provided stats include an initial evaluation.
            Used when ``value_domain`` is set to ``'agent_evaluations``. Defaults to ``True``.
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
            - ``'outliers'`` - Each trajectory task will start at the outlier boundry of previous
              trajectories' termination points (Q3 + 1.5IQR),
            - ``'max'`` - Each task trajectory will start at the max termination point of
              previous trajectories.

            Can either be a single value or a list of values, one for each plotted curriculum trajectory.
            Defaults to ``'most'``.
        show_run_traces: Whether to show individual run traces (with lower opacity)
            Can either be a single value or a list of values, one for each plotted trajectory.
            Defaults to ``False``.
        common_run_traces_start: Whether individual run traces should start at the same point (the same
            as their parent trajectory) or as continuations of their predecessors (when plotting
            curriculum trajectory). Can either be a single value or a list of values, 
            one for each plotted curriculum trajectory. Defaults to ``False``.
        as_separate_figs: Whether each trajectory should be plotted on a separate figure. If set to ``True``
            and ``save_path`` is not ``None`` each figure will be saved in a file with a ``_i`` suffix
            corresponding to the position of the runs stats object in ``runs`` list. Defaults to ``False``.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.
    
    Examples:

        The following imports are needed for the following examples

        >>> from academia.curriculum import LearningTask, Curriculum, LearningStats
        >>> import academia.visualisations as vis

        For details on how to configure agents see :mod:`academia.agents`

        Plotting evaluations from a single task run

        >>> task: LearningTask = ...
        >>> agent = ...
        >>> task.run(agent)
        >>> stats: LearningStats = task.stats
        >>> vis.plot_trajectories([stats]) # note that a single stats object has to be passed inside a list

        Plotting evaluations from a single curriculum run. See :mod:`academia.curriculum`
        for more details on how to configure and run a curriculum.

        >>> curriculum: Currriculum = ...
        >>> agent = ...
        >>> curriculum.run(agent)
        >>> stats: dict[str, LearningStats] = curriculum.stats
        >>> vis.plot_trajectories([stats]) # note that a single stats object has to be passed inside a list

        Plotting several runs of a single task with some additonal visuals

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

        Plotting curriculum vs task comparison

        >>> task: LearningTask = ...
        >>> curriculum: Curriculum = ...
        >>> n_runs = 5
        >>> task_runs_stats: list[LearningStats] = []
        >>> curriculum_runs_stats: list[dict[str, LearningStats]] = []
        >>> for _ in range(n_runs):
        >>>     agent = ....
        >>>     task.run(agent)
        >>>     taks_runs_stats.append(task.stats)
        >>>     agent = ...
        >>>     curriculum.run(agent)
        >>>     curriculum_runs_stats.append(curriculum.stats)
        >>> vis.plot_trajectories(
        >>>     [task_runs_stats, curriculum_runs_stats],
        >>>     show_std=True,
        >>>     show_run_traces=[True, False] # show individual traces only for the task
        >>> )

        Plotting curriculum vs curriculum as separate figures with different time domains

        >>> curriculum_1: Curriculum = ...
        >>> curriculum_2: Curriculum = ...
        >>> n_runs = 5
        >>> curriculum_1_runs_stats: list[dict[str, LearningStats]] = []
        >>> curriculum_2_runs_stats: list[dict[str, LearningStats]] = []
        >>> for _ in range(n_runs):
        >>>     agent = ....
        >>>     curriculum_1.run(agent)
        >>>     curriculum_1_runs_stats.append(curriculum_1.stats)
        >>>     agent = ...
        >>>     curriculum_2.run(agent)
        >>>     curriculum_2_runs_stats.append(curriculum_2.stats)
        >>> vis.plot_trajectories(
        >>>     [curriculum_1_runs_stats, curriculum_2_runs_stats],
        >>>     as_separate_fig: True
        >>>     time_domain=["steps", "episodes"]
        >>> )

        Plotting a running average of steps in episodes in several runs of a single task

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
    
    def _iterate_trajectories_kwargs():
        for i in range(len(runs)):
            trajectory_kwargs = {
                kwarg_name: trajectories_kwargs[kwarg_name][i] for kwarg_name in trajectories_kwargs}
            yield trajectory_kwargs

    if not isinstance(runs[0], list):
        runs = [runs]

    # compile arguments of similar nature into a single dictionary
    # to be passed as kwargs later on
    trajectories_kwargs = {
        'time_domain': time_domain,
        'value_domain': value_domain,
        'includes_init_eval': includes_init_eval,
        'show_std': show_std,
        'show_run_traces': show_run_traces,
        'task_trace_start': task_trace_start,
        'common_run_traces_start': common_run_traces_start
    }

    # since we allow the user to either pass single values of a list of values (one for each
    # trajectory) we have to convert all non-list values to lists so that it's compatible
    # with the rest of the code
    for kwarg_name, value in trajectories_kwargs.items():
        if not isinstance(value, list):
            trajectories_kwargs[kwarg_name] = [value] * len(runs)

    if as_separate_figs:
        # recursively call plot_trajectories for each trajectory
        for i, trajectory_kwargs in enumerate(_iterate_trajectories_kwargs()):
            trajectory = runs[i]
            new_save_path = None if save_path is None else save_path + f'_{i}'
            plot_trajectories(
                [trajectory], 
                as_separate_figs=False, 
                show=show, 
                save_path=new_save_path, 
                save_format=save_format, 
                **trajectory_kwargs)
        return

    with _create_figure(show, save_path, save_format=save_format) as fig:
        for i, trajectory_kwargs in enumerate(_iterate_trajectories_kwargs()):
            trajectory = runs[i]
            if isinstance(trajectory[0], LearningStats):
                color = _get_colors(1, i, i, len(runs))[0]
                _add_task_trajectory(fig, trajectory, color=color, **trajectory_kwargs)
            if isinstance(trajectory[0], dict):
                colors = _get_colors(len(trajectory[0]), i, i, len(runs))
                _add_curriculum_trajectory(fig, trajectory, colors=colors, **trajectory_kwargs)
        fig.update_layout(
            xaxis_title=f"Timestamps ({trajectories_kwargs['time_domain']})",
            yaxis_title=f"Values ({trajectories_kwargs['value_domain']})"
        )
    

def plot_evaluation_impact(
        n_episodes_x: list[int], 
        task_runs_y: list[LearningTaskRuns],
        show: bool = False,
        save_path: Optional[str] = None, 
        save_format: SaveFormat = 'png',
        ):
    """
    Plots the impact of learning duration in task with difficulty level = x to evaluation 

    Args:
        n_episodes_x: Number of episodes in task X.
        task_runs_y: Learning statistics for tasks in level Y.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Raises:
        ValueError: If the number of tasks at level x and level y is not equal. It is assumed that 
            the number of tasks at level x and level y is equal because the experiment involves testing 
            the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
            the number of episodes spent in the easier one affects the evaluation of the agent in a more difficult 
            environment.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the ``"_evaluation_impact"`` is added to the end of the ``save_path``
    
    Examples:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> n_episodes_x = [100, 500, 1000]
        >>> task_runs_y: list[list[LearningStats]] = []
        >>> n_runs = 5
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
        raise ValueError("The number of tasks at level x and level y should be equal.")

    agent_evaluations = []
    for task_runs in task_runs_y:
        mean_eval = np.mean([run.agent_evaluations[-1] for run in task_runs])
        agent_evaluations.append(mean_eval)

    with _create_figure(show, save_path, save_format=save_format) as fig:
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
    """
    Plots the impact of learning duration in task x and task y to evaluation of task z.
    See examples for more details

    Args:
        n_episodes_x: Number of episodes in task X.
        n_episodes_y: Number of episodes in task Y.
        task_runs_z: Learning statistics for tasks in level Z.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Raises:
        ValueError: If the number of tasks at level x, level y and level z is not equal. 
            It is assumed that the number of tasks at level x, level y and level z is equal 
            because the experiment involves testing the curriculum on group of three tasks with 
            three specific levels of difficulty in order to examine how the number of episodes spent 
            in the easier ones affects the evaluation of the agent in the more difficult environment.
    
    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of
        the name of the saved plot, the ``"_eval_impact_2d"`` suffix is added to the end of the ``save_path``

    Examples:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> n_episodes_x = [100, 500, 1000]
        >>> n_episodes_y = [200, 400, 600]
        >>> task_runs_z: list[list[LearningStats]] = []
        >>> n_runs = 5
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
        >>> vis.plot_evaluation_impact_2d(n_episodes_x, n_episodes_y, task_runs_z)
    """

    if len(n_episodes_x) != len(n_episodes_y) or len(n_episodes_x) != len(task_runs_z):
        raise ValueError("The number of tasks at level x, level y and level z should be equal.")
    
    agent_evaluations = []
    for task_runs in task_runs_z:
        mean_eval = np.mean([run.agent_evaluations[-1] for run in task_runs])
        agent_evaluations.append(mean_eval)

    with _create_figure(show, save_path, '_eval_impact_2d', save_format) as fig:
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
        time_domain_y: Union[TimeDomain, Literal["as_x"]] = "as_x",
        show: bool = False, 
        save_path: str = None, 
        save_format: SaveFormat = 'png'):
    """
    Plots the impact of the number of episodes in task x on the total time spent in both tasks.
    See examples for more details

    Args:
        task_runs_x: Learning statistics for tasks in level X.
        task_runs_y: Learning statistics for tasks in level Y.
        time_domain_x: Time domain which will be used on the X-axis.
        time_domain_y: Time domain which will be used on the Y-axis.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Raises:
        ValueError: If the number of tasks at level x and level y is not equal. It is assumed that 
            the number of tasks at level x and level y is equal because the experiment involves testing 
            the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
            the number of episodes spent in the easier one affects the total time spent in both tasks.

    Note:
        If save path is provided, the plot will be saved to the specified path. To increase the clarity of
        the name of the saved plot, the ``"_time_impact"`` suffix is added to the end of the ``save_path``

    Examples:
        
        >>> wall_time_stops = [600, 900, 1200] # 10, 15 and 20 minutes
        >>> # the exact time at which the task stops might slightly differ
        >>> # from the set stop condition as it is only checked at the end of each episode
        >>> n_runs = 5
        >>> task_runs_x: list[list[LearningStats]] = 0
        >>> task_runs_y: list[list[LearningStats]] = 0
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
        raise ValueError("The number of tasks at level x and level y should be equal.")
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

    with _create_figure(show, save_path, save_format=save_format) as fig:
        _add_trace(fig, x_times, xy_times)
        fig.update_layout(
            xaxis_title=f"Learning duration in task X ({time_domain_x})",
            yaxis_title=f"Total time spent in both tasks ({time_domain_y})",
            title="Impact of learning duration in task x on total time spent in both tasks"
        )
