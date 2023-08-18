import numpy as np

class Task():

    __slots__ = ['environment', 'max_episodes', 'greedy_episode_frequency',
                 'min_target_reward', 'episodes_to_early_stop', 'environment_params']

    def __init__(self,
                 environment,
                 max_episodes, 
                 greedy_episode_frequency,
                 min_target_reward, 
                 episodes_to_early_stop = np.inf,
                 environment_params = {}) -> None:
        self.environment = environment
        self.max_episodes = max_episodes
        self.greedy_episode_frequency = greedy_episode_frequency
        self.min_target_reward = min_target_reward
        self.episodes_to_early_stop = episodes_to_early_stop
        self.environment_params = environment_params

    def run_task(self, agent):
        pass


class Curriculum():

    __slots__ = ['tasks']

    def __init__(self, tasks) -> None:
        self.task = tasks

    def run_curriculum(self, agent):
        pass