import os
from typing import Dict, Literal

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from academia.curriculum import LearningStats

def plot_task(task_stats: LearningStats, show: bool = True, save_path: str = None, save_format: Literal['png', 'html'] = 'png'):
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

def plot_curriculum_vs_nocurriculum(curriculum_stats: Dict[str, LearningStats], nocurriculum_stats: LearningStats, show: bool = True,
                               save_path: str = None, save_format: Literal['png', 'html'] = 'png', eval_interval: int = 20):
    fig = go.Figure()
    total_steps_to_last_eval = 0
    for task_id, task_stats in curriculum_stats.items():
        evaluations = task_stats.agent_evaluations
        steps_count = task_stats.step_counts
        steps_count[0] += total_steps_to_last_eval
        steps_cum = np.cumsum(steps_count)
        indices = np.arange(eval_interval - 1, len(steps_cum), eval_interval)
        steps_to_eval = steps_cum[indices]
        fig.add_trace(go.Scatter(x=np.concatenate([[total_steps_to_last_eval],steps_to_eval]), 
                                 y=evaluations, mode='lines', name=f'Task {task_id}'))
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



