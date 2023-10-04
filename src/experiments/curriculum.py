import numpy as np
import time
from typing import List
import sys
import os

sys.path.append(os.path.abspath("./src"))

from environments import *

def environment_type_from_name(name):
    if name == "BridgeBuilding":
        return BridgeBuilding
    elif name == "LunarLander":
        pass

class TaskParameters():
    
    __slots__ = ['environment_name', 'environment_params', 'max_episodes', 'greedy_evaluation_frequency',
                 'min_target_reward', 'episodes_to_early_stop']
    
    def __init__(self,
                 environment_name,
                 environment_params,
                 max_episodes,
                 greedy_evaluation_frequency,
                 min_target_reward,
                 episodes_to_early_stop=np.inf):
        self.environment_name = environment_name
        self.environment_params = environment_params
        self.max_episodes = max_episodes
        self.greedy_evaluation_frequency = greedy_evaluation_frequency
        self.min_target_reward = min_target_reward
        self.episodes_to_early_stop = episodes_to_early_stop

    @classmethod
    def from_dict(cls, params_dict):
        pass

    def to_dict(self):
        pass

    @classmethod
    def from_json(cls, path):
        pass
    
    def to_json(self, path):
        pass


class Task():

    __slots__ = ['environment_type', 'task_params',
                 'episode_rewards', 'greedy_evaluations', 
                 'environment']

    def __init__(self, task_params : TaskParameters) -> None:
        if type(task_params) is dict:
            self.task_params = TaskParameters.from_dict(task_params)
        else:
            self.task_params = task_params
        self.environment_type = environment_type_from_name(task_params.environment_name)
        

    def run_task(self, agent):
        self.environment = self.environment_type(**self.task_params.environment_params)
        self.episode_rewards = []
        self.greedy_evaluations = []
        agent.epsilon_decay = np.power(0.01, 1/self.task_params.max_episodes)

        for episode in range(self.task_params.max_episodes):
            reward = self.__run_episode(agent, self.environment)
            self.episode_rewards.append(reward)
            if (episode + 1) % self.task_params.greedy_evaluation_frequency != 0:
                continue # do not evaluate

            reward = self.__run_greedy_evaluation(agent, self.environment)
            self.greedy_evaluations.append(reward)

            if len(self.greedy_evaluations) < self.task_params.episodes_to_early_stop:
                continue
            for reward in self.greedy_evaluations[-self.task_params.episodes_to_early_stop:]:
                if reward < self.task_params.min_target_reward:
                    break
            else:
                return episode + 1, self.episode_rewards, self.greedy_evaluations
            
        return self.task_params.max_episodes, self.episode_rewards, self.greedy_evaluations

    def __run_episode(self, agent, env):
        total_reward = 0
        current_state = self.environment.reset()
        done = False
        while not done:
            action = agent.get_action(current_state, legal_mask=env.get_legal_mask())
            new_state, reward, done = env.step(action)

            agent.update(current_state, action, reward, new_state, done)
            agent.decay_epsilon()

            current_state = new_state
            total_reward += reward
        return total_reward

    def __run_greedy_evaluation(self, agent, env):
        total_reward = 0
        current_state = env.reset()
        done = False
        while not done:
            action = agent.get_action(current_state, legal_mask=env.get_legal_mask(), greedy=True)
            new_state, reward, done = env.step(action)
            current_state = new_state
            total_reward += reward
        return total_reward


class Curriculum():

    __slots__ = ['tasks']

    def __init__(self, tasks : List[Task]) -> None:
        self.tasks = tasks

    def run_curriculum(self, agent, verbose=0):
        total_episodes = 0
        total_duration = 0
        for i, task in enumerate(self.tasks):
            start_time = time.time()
            episodes, rewards, evaluations = task.run_task(agent)
            duration = time.time() - start_time
            if verbose>1:
                print(f"Task {i+1} finished:\n\
\tEpisodes: {episodes}({(100*episodes/task.task_params.max_episodes):.2f}%)\n\
\tMean reward in last {task.task_params.episodes_to_early_stop} episodes: {np.mean(evaluations[-task.task_params.episodes_to_early_stop:])}\n\
\tDuration: {duration:.2f}s")
            total_episodes += episodes
            total_duration += duration
        if verbose>0:
            print(f"Curriculum finished:\n\tTotal episodes: {total_episodes}\n\tTotal duration: {total_duration:.2f}s")
        return total_episodes