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

    fig_steps = px.line(x=np.arange(len(task_stats.step_counts)), y=task_stats.step_counts)
    fig_steps.update_layout(
        xaxis_title="Episode",
        yaxis_title="Steps",
        title="Steps per episode"
    )


    fig_evaluations = px.line(x=np.arange(len(task_stats.agent_evaluations)), y=task_stats.agent_evaluations)
    fig_evaluations.update_layout(
        xaxis_title="Episode",
        yaxis_title="Score",
        title="Agent evaluations"
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
        fig.update_yaxes(title_text='Reward', row=row, col=1)
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
                                    eval_interval: int = 20, includes_init_eval: bool = False):
    fig = go.Figure()
    total_steps_to_last_eval = 0
    for task_id, task_stats in curriculum_stats.items():
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