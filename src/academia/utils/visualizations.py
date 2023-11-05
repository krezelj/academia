import os
from typing import Dict, Literal, List

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from academia.curriculum import LearningStats


def plot_task(task_stats: LearningStats, show: bool = True, save_path: str = None, 
              save_format: Literal['png', 'html'] = 'png'):
    """
    Plots the learning statistics for a single task.

    The returned plots include:
    - A plot showing rewards and their moving average over episodes.
    - A plot displaying the steps taken by the agent in each episode against the episode numbers.
    - A plot indicating the agent's learning progress, represented by its evaluation score, in 
        relation to the number of steps taken up to the current evaluation.

    Note:
        - If save path is provided, the all three plots will be saved to the specified path. To diferentiate between the plots,
            the file names will be appended with ``_rewards``, ``_steps`` and ``_evaluations`` respectively.
        - if show is set to ``True``, the plots will be displayed in the browser window.   
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
        >>> from academia.utils.visualizations import plot_task
        >>> test_task = LearningTask(
        >>>     env_type= LunarLander,
        >>>     env_args={'difficulty': 2, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>>     eval_interval=100,
        >>>     stats_save_path='./my_task_stats.json',
        >>> )

        Running a task:

        >>> from academia.agents import DQNAgent
        >>> from academia.models import LunarLanderMLP
        >>> agent = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LunarLanderMLP,
        >>>     random_state=123,
        >>> )
        >>> test_task.run(agent, verbose=4, render=True)

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

    eval_interval = task_stats.eval_interval
    steps_count = task_stats.step_counts
    steps_cum = np.cumsum(steps_count)
    indices = np.arange(eval_interval - 1, len(steps_cum), eval_interval)
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
    
    
def plot_trajectory_curriculum(curriculum_stats: Dict[str, LearningStats], show: bool = True,
                               save_path: str = None, save_format: Literal['png', 'html'] = 'png'):
    """
    Plots the trajectories of episode rewards for multiple tasks in the curriculum.

    Args:
        curriculum_stats: Learning statistics for multiple tasks in the curriculum.
        show: Whether to display the plot. Defaults to True.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Returns:
        Absolute path to the saved plot file if the ``save_path`` was provided.
    
    Note:
        - If save path is provided, the plot will be saved to the specified path. To increase the clarity of the name of the saved plot, 
            the _rewards_curriculum is added to the end of the ``save_path`` 
    
    Examples:
        Initialisation of a curriculum we want to plot:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> curriculum = Curriculum(
        >>>     tasks=[task1, task2],
        >>>     output_dir='./my_curriculum/',
        >>> )

        Running a curriculum:

        >>> from academia.agents import DQNAgent
        >>> from academia.models import LavaCrossingMLP
        >>> agent = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> curriculum.run(agent, verbose=4, render=True)

        Plotting the curriculum:
        >>> from academia.utils.visualizations import plot_trajectory_curriculum
        >>> plot_trajectory_curriculum(curriculum.stats, save_path='./curriculum', save_format='png')
    """
    num_tasks = len(curriculum_stats)
    num_cols = 2
    num_rows = (num_tasks + 1) // num_cols

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'Episode rewards for task {task_id}' for task_id in curriculum_stats.keys()])

    row = 1
    col = 1
    for task_id, task_stats in curriculum_stats.items():
        rewards = task_stats.episode_rewards
        fig.add_trace(go.Scatter(y=rewards, mode='lines', name=f'Task {task_id}'), row=row, col=col)
        fig.update_xaxes(title_text='Step', row=row, col=col)
        fig.update_yaxes(title_text='Episode', row=row, col=1)
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


def plot_curriculum_vs_nocurriculum(curriculum_stats: Dict[str, LearningStats], 
                                    nocurriculum_stats: LearningStats, show: bool = True,
                                    save_path: str = None, save_format: Literal['png', 'html'] = 'png', 
                                    includes_init_eval: bool = False):
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
    
    Returns:
        Absolute path to the saved plot file if the ``save_path`` was provided.
    
    Note:
        - If save path is provided, the plot will be saved to the specified path. To increase the clarity of the name of the saved plot, 
            the _curriculum_vs_no_curriculum is added to the end of the ``save_path``
        - Change the ``includes_init_eval`` to ``True`` if you specified this flag in :class:`LearningTask`.
    
    Examples:
        Initialisation of a curriculum we want to plot:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task1 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task2 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> curriculum = Curriculum(
        >>>     tasks=[task1, task2],
        >>>     output_dir='./my_curriculum/',
        >>> )

        Initialisation of a task without curriculum we want to plot:

        >>> no_curriculum = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 2, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1500},
        >>> )

        Defining an agent:

        >>> from academia.agents import DQNAgent
        >>> from academia.models import LavaCrossingMLP
        >>> agent_curriculum = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> agent_no_curriculum = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )

        Running a curriculum:

        >>> curriculum.run(agent_curriculum, verbose=4, render=True)

        Running a task without curriculum:

        >>> no_curriculum.run(agent_no_curriculum, verbose=4, render=True)

        Plotting the curriculum vs no curriculum:

        >>> from academia.utils.visualizations import plot_curriculum_vs_nocurriculum
        >>> plot_curriculum_vs_nocurriculum(curriculum.stats, no_curriculum.stats, save_path='./curriculum', save_format='png')
    """
    fig = go.Figure()
    total_steps_to_last_eval = 0
    for task_id, task_stats in curriculum_stats.items():
        eval_interval = task_stats.eval_interval
        evaluations = task_stats.agent_evaluations
        steps_count = task_stats.step_counts
        steps_count[0] += total_steps_to_last_eval
        steps_cum = np.cumsum(steps_count)
        indices = np.arange(eval_interval - 1, len(steps_cum), eval_interval)
        steps_to_eval = steps_cum[indices]
        if includes_init_eval:
            fig.add_trace(go.Scatter(x=np.concatenate([[total_steps_to_last_eval],steps_to_eval]), 
                                    y=evaluations, mode='lines', name=f'Task {task_id}'))
        else:
            fig.add_trace(go.Scatter(x=steps_to_eval, y=evaluations, mode='lines', name=f'Task {task_id}'))
            
        total_steps_to_last_eval = steps_to_eval[-1]
    nocurr_steps_cum =  np.cumsum(nocurriculum_stats.step_counts)
    nocurr_indices = np.arange(eval_interval - 1, len(nocurr_steps_cum), eval_interval)
    no_curr_steps_to_eval = nocurr_steps_cum[nocurr_indices]
    fig.add_trace(go.Scatter(x=np.concatenate([[0], no_curr_steps_to_eval]), y=nocurriculum_stats.agent_evaluations, mode='lines', name='No curriculum'))
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


def plot_evaluation_impact(num_of_episodes_lvl_x: List[int], stats_lvl_y: List[LearningStats],
                           show: bool = True, save_path: str = None, 
                           save_format: Literal['png', 'html'] = 'png'):
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
        ``ValueError``: If the number of tasks at level x and level y is not equal. It is assumed that 
        the number of tasks at level x and level y is equal because the experiment involves testing 
        the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
        the number of episodes spent in the easier one affects the evaluation of the agent in a more difficult 
        environment.

        ``ValueError``: If the number of evaluation scores is not equal to the number of tasks at level x.
        This means that the evaluation was not performed only at the end of the task, which is necessary to
        correctly measure the impact of learning duration in task with difficulty level = x to evaluation of this task.

    Returns:
        Absolute path to the saved plot file if ``save_path`` was provided.
    
    Note:
        - If save path is provided, the plot will be saved to the specified path. To increase the clarity of 
        the name of the saved plot, the _evaluation_impact is added to the end of the ``save_path``
        - It is important that evaluations in task with difficulty level = y
        are only performed at the end of the task and that the number of episodes in this task should be fixed
        to correctly measure the impact of learning duration in task with difficulty level = x to evaluation of this task.     
    
    Examples:
        Initialisation of a diffrent pairs we want to analyze:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task0_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task1_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task0_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 700},
        >>> )
        >>> task1_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task0_v1000 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task1_v1000 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )

        Initialisation of agents:

        >>> from academia.agents import DQNAgent
        >>> from academia.models import LavaCrossingMLP
        >>> agent_v500 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> agent_v700 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> agent_v1000 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        
        Initialisation of a curriculums and running them:

        >>> curriculum_v500 = Curriculum(
        >>>     tasks=[task0_v500, task1_v500],
        >>>     output_dir='./curriculum_v500/',
        >>> )
        >>> curriculum_v500.run(agent, verbose=4, render=True)
        >>> curriculum_v700 = Curriculum(
        >>>     tasks=[task0_v700, task1_v700],
        >>>     output_dir='./curriculum_v700/',
        >>> )
        >>> curriculum_v700.run(agent, verbose=4, render=True)
        >>> curriculum_v1000 = Curriculum(
        >>>     tasks=[task0_v1000, task1_v1000],
        >>>     output_dir='./curriculum_v1000/',
        >>> )
        >>> curriculum_v1000.run(agent, verbose=4, render=True)

        Plotting the evaluation impact:

        >>> from academia.utils.visualizations import plot_evaluation_impact
        >>> plot_evaluation_impact([500, 700, 1000], 
                                   [curriculum_v500.stats[1], curriculum_v700.stats[1], curriculum_v1000.stats[1]],
                                    save_path='./evaluation_impact', 
                                    save_format='png')
    """
    agent_evals_lvl_y = [task.agent_evaluations for task in stats_lvl_y]

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


def plot_time_impact(stats_lvl_x: List[LearningStats], stats_lvl_y: List[LearningStats], show: bool = True, 
                     save_path: str = None, save_format: Literal['png', 'html'] = 'png'):
    """
    Plots the impact of the number of episodes in task x on the total time spent in both tasks.

    The purpose of this plot is to show how the number of episodes in task x affects the total 
    time spent in both tasks. It is done by testing the curriculum on pairs of tasks with two
    specific levels of difficulty in order to examine how the number of episodes spent in the easier
    one affects the total time spent in both tasks when the stop condition in harder task is specified to reach the 
    fixed value of agent evaluation eg. equals 200.

    On the X-axis we have the number of episodes in task x, while on the Y-axis we have the total time spent in 
    both tasks.

    Args:
        stats_lvl_x: Learning statistics for tasks in level X.
        stats_lvl_y: Learning statistics for tasks in level Y.
        show: Whether to display the plot. Defaults to ``True``.
        save_path: Path to save the plot. Defaults to ``None``.
        save_format: File format for saving the plot. Defaults to 'png'.

    Raises:
        ``ValueError``: If the number of tasks at level x and level y is not equal. It is assumed that 
        the number of tasks at level x and level y is equal because the experiment involves testing 
        the curriculum on pairs of tasks with two specific levels of difficulty in order to examine how 
        the number of episodes spent in the easier one affects the total time spent in both tasks.

    Returns:
        Absolute path to the saved plot file if ``save_path`` was provided.

    Note:
        - If save path is provided, the plot will be saved to the specified path. To increase the clarity of
        the name of the saved plot, the _time_impact is added to the end of the ``save_path``

    Examples:
        Initialisation of a diffrent pairs we want to analyze:

        >>> from academia.curriculum import LearningTask, Curriculum
        >>> from academia.environments import LavaCrossing
        >>> task0_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 500},
        >>> )
        >>> task1_v500 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )
        >>> task0_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 700},
        >>> )
        >>> task1_v700 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )
        >>> task0_v1000 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 0, 'render_mode': 'human'},
        >>>     stop_conditions={'max_episodes': 1000},
        >>> )
        >>> task1_v1000 = LearningTask(
        >>>     env_type=LavaCrossing,
        >>>     env_args={'difficulty': 1, 'render_mode': 'human'},
        >>>     stop_conditions={'min_evaluation_score': 200},
        >>> )

        Initialisation of agents:

        >>> from academia.agents import DQNAgent
        >>> from academia.models import LavaCrossingMLP
        >>> agent_v500 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> agent_v700 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )
        >>> agent_v1000 = DQNAgent(
        >>>     n_actions=LavaCrossing.N_ACTIONS,
        >>>     nn_architecture=LavaCrossingMLP,
        >>>     random_state=123,
        >>> )

        Initialisation of a curriculums and running them:

        >>> curriculum_v500 = Curriculum(
        >>>     tasks=[task0_v500, task1_v500],
        >>>     output_dir='./curriculum_v500/',
        >>> )
        >>> curriculum_v500.run(agent, verbose=4, render=True)
        >>> curriculum_v700 = Curriculum(
        >>>     tasks=[task0_v700, task1_v700],
        >>>     output_dir='./curriculum_v700/',
        >>> )
        >>> curriculum_v700.run(agent, verbose=4, render=True)
        >>> curriculum_v1000 = Curriculum(
        >>>     tasks=[task0_v1000, task1_v1000],
        >>>     output_dir='./curriculum_v1000/',
        >>> )
        >>> curriculum_v1000.run(agent, verbose=4, render=True)

        Plotting the time impact:

        >>> from academia.utils.visualizations import plot_time_impact
        >>> plot_time_impact([curriculum_v500.stats[0], curriculum_v700.stats[0], curriculum_v1000.stats[0]], 
                             [curriculum_v500.stats[1], curriculum_v700.stats[1], curriculum_v1000.stats[1]],
                              save_path='./time_impact', 
                              save_format='png')
    """
    if len(stats_lvl_x) != len(stats_lvl_y):
        raise ValueError("The number of tasks at level x and level y should be equal.")
    
    episoded_lvl_x = [len(task.step_counts) for task in stats_lvl_x]
    agent_time_lvl_x = [np.sum(task.episode_cpu_times) for task in stats_lvl_x]
    agent_time_lvl_y = [np.sum(task.episode_cpu_times) for task in stats_lvl_y]
    total_times_for_both = agent_time_lvl_x + agent_time_lvl_y
    fig = px.line(x=episoded_lvl_x, y=total_times_for_both,
                          title='Impact of number of episodes in task x on total time spent in both tasks')
    fig.update_layout(
        xaxis_title="Number of episodes in task X",
        yaxis_title="Total time spent in both tasks"
    )
    fig.update_traces(
        hovertemplate="<br>".join([
            "Number of episodes in task X: %{x}",
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


def plot_multiple_evaluation_impact(num_of_episodes_lvl_x: List[int], num_of_episodes_lvl_y: List[int], 
                                    stats_lvl_z: List[LearningStats], show: bool = True, save_path: str = None, 
                                    save_format: Literal['png', 'html'] = 'png'):
    agent_evals_lvl_z = [task.agent_evaluations for task in stats_lvl_z]

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
                     labels = {'color': 'Evaluation score in task Z'},
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