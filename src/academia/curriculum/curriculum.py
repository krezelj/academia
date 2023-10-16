import numpy as np
from . import Task

# TODO Add more statistics about agent training process (e.g. time to train)
# TODO Possibly add ability to form the curriculum as a dynamic directed graph
# (many parallel dependencies, adding new tasks automatically etc. probably very advanced)

class Curriculum:

    __slots__ = ['tasks']

    def __init__(self, tasks : list[Task]) -> None:
        self.tasks = tasks

    def run(self, agent, verbose=0):
        total_episodes = 0
        for i, task in enumerate(self.tasks):
            if verbose > 0:
                print(f"Running Task: {i + 1 if task.name is None else task.name}... ", end="")
            task.run()
            total_episodes += len(task.episode_rewards)
            if verbose > 0:
                print(f"finished after {len(task.episode_rewards)} episodes.")
        if verbose > 0:
            print(f"Curriculum finished after {total_episodes} episodes.")