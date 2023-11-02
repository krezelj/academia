import json
import os
from typing import Dict

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from academia.curriculum import LearningStats

def plot_task(task_stats: LearningStats, show: bool = True, save_path: str = None):
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
        fig_rewards.write_image(f"{save_path}_rewards.png")
        fig_steps.write_image(f"{save_path}_steps.png")
        fig_evaluations.write_image(f"{save_path}_evaluations.png")
        return os.path.abspath(save_path)
