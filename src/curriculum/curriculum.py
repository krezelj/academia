import numpy as np
from typing import List

from . import Task

class Curriculum():

    __slots__ = ['tasks']

    def __init__(self, tasks : List[Task]) -> None:
        self.tasks = tasks

    def run_curriculum(self, agent, verbose=0):
        total_episodes = 0
        for i, task in enumerate(self.tasks):
            if verbose > 0:
                print(f"Running Task: {i + 1 if task.task_name is None else task.task_name}... ", end="")
            task.run_task()
            total_episodes += len(task.episode_rewards)
            if verbose > 0:
                print(f"finished after {len(task.episode_rewards)} episodes.")
        if verbose > 0:
            print(f"Curriculum finished after {total_episodes} episodes.")