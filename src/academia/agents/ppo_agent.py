from .base import Agent

import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Type


# TODO Things that will probably have to be implemented outside
# `Task`
#   early episode stop -- stop episode after a set number of steps
# `Agent`
#   get_action returns logits as well, unless I implement a private method for PPO only


class PPOAgent(Agent):

    __slots__ = ['n_actions', 'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'min_epsilon',
                 'actor', 'critic', 'memory', 'steps_per_train', 
                 '__steps_counter', '__episode_length']

    def __init__(self, 
                 actor_architecture: Type[nn.Module],
                 critic_architecture: Type[nn.Module],
                 steps_per_train : int,
                 n_actions: int,
                 gamma: float = 0.99, 
                 epsilon: float = 1.,
                 epsilon_decay: float = 0.99,
                 min_epsilon: float = 0.01) -> None:
        super(PPOAgent, self).__init__(epsilon, min_epsilon, epsilon_decay)
        self.n_actions = n_actions
        self.gamma = gamma
        self.steps_per_train = steps_per_train

        self.actor = actor_architecture()
        self.critic = critic_architecture()

        self.__reset_memory()

    def __init_networks(self, actor_architecture : Type[nn.Module], critic_architecture : Type[nn.Module]):
        # TODO add parameter to control lr in optimisers
        pass

    def __get_action_logits(self, state) -> npt.NDArray[np.float_]:
        pass

    def __reset_memory(self) -> None:
        # states, actions, log probs of actions, rewards, rewards-to-go, episode length
        # using dictionary for better readability
        # TODO consider using internal `class PPOMemory` to deal with all that
        self.memory = {
            'states': [],
            'actions': [],
            'actions_logits': [],
            'rewards': [],
            'rewards_to_go': [],
            'episode_lengths': [0]
        }
        self.__steps_counter = 0
        self.__episode_length = 0

    def get_action(self, state, legal_mask=None, greedy=False):
        pass

    def update(self, state, action, reward: float, new_state, is_terminal: bool):
        # TODO move this to __update_memory function
        # TODO create issue to rename `remember` in DQN to `__update_memory`
        
        self.__steps_counter += 1
        self.__episode_length += 1
        self.memory['states'].append(state)
        self.memory['actions'].append(action)

        # TODO This piece of code puts the actor network to work twice
        # we should probably cache the logits from the last time we called get_action
        # and check if the states match, if they do, use cached value otherwise recalculate
        self.memory['actions_logits'].append(self.__get_action_logits(state))
        self.memory['rewards'].append(reward)
        if is_terminal or self.__steps_counter >= self.steps_per_train:
            self.memory['episode_lengths'].append(self.__episode_length)
            self.__episode_length = 0

        if self.__steps_counter > self.steps_per_train:
            # calculate rewards-to-go
            for episode_length in self.memory['episode_lengths']:
                # TODO currently I use `t_offset` to keep track of the index of the reward
                # in memory['rewards'] however if I decide to use a separate class for PPOMemory
                # it should probably keep track of stats for each episode in a separate list
                t_offset = 0 
                discounted_reward = 0
                for t in range(episode_length, -1, -1):
                    discounted_reward += self.memory['rewards'][t + t_offset] + self.gamma * discounted_reward
                t_offset += episode_length

            # TODO before training convert memory to tensors
            pass # DO TRAINING

    


    @classmethod
    def load(cls, path: str):
        pass

    def save(self, path: str) -> None:
        pass