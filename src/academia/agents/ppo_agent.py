from typing import Type

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from .base import Agent




# TODO Things that will probably have to be implemented outside
# `Task`
#   early episode stop -- stop episode after a set number of steps


class PPOAgent(Agent):

    class PPOBuffer:
        # TODO Check if actions are neccessary to remember

        __slots__ = ['buffer_size',
                     '__update_count',
                     '__episode_length_counter'
                     'states', 
                     'actions', 
                     'actions_logits', 
                     'rewards', 
                     'rewards_to_go', 
                     'episode_lengths',
                     ]

        def __init__(self, buffer_size):
            self._reset()
            self.buffer_size = buffer_size
            pass

        def _update(self, state, action, action_logits, reward, is_terminal) -> bool:
            self.__update_counter += 1
            self.__episode_length_counter += 1

            self.states.append(state)
            self.actions.append(action)
            self.actions_logits.append(action_logits)
            self.rewards.append(reward)
            
            if is_terminal or self.__update_counter >= self.buffer_size:
                self.episode_lengths.append(self.__episode_length_counter)
                self.__episode_length_counter = 0

            return self.__update_counter >= self.buffer_size
            

        def _calculate_rewards_to_go(self):
            for episode_length in self.episode_lengths:
                # TODO currently I use `t_offset` to keep track of the index of the reward
                # in memory['rewards'] however if I decide to use a separate class for PPOMemory
                # it should probably keep track of stats for each episode in a separate list
                t_offset = 0 
                discounted_reward = 0
                for t in range(episode_length, -1, -1):
                    discounted_reward += self.rewards[t + t_offset] + self.gamma * discounted_reward
                t_offset += episode_length

        def _reset(self):
            self.__episode_length_counter = 0
            self.__update_counter = 0

            self.states = []
            self.actions = []
            self.actions_logits = []
            self.rewards = []
            self.rewards_to_go = []
            self.episode_lengths = []

        def _get_tensors(self):
            return torch.tensor(self.states), torch.tensor(self.actions_logits), torch.tensor(self.rewards_to_go)

    __slots__ = ['n_actions', 'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'min_epsilon',
                 'clip', 'actor', 'critic', 'memory', 'batch_size', 'n_epochs']

    def __init__(self, 
                 actor_architecture: Type[nn.Module],
                 critic_architecture: Type[nn.Module],
                 buffer_size : int,
                 batch_size : int,
                 n_epochs: int,
                 n_actions: int,
                 clip: float,
                 gamma: float = 0.99, 
                 epsilon: float = 1.,
                 epsilon_decay: float = 0.99,
                 min_epsilon: float = 0.01) -> None:
        super(PPOAgent, self).__init__(epsilon, min_epsilon, epsilon_decay)
        self.n_actions = n_actions
        self.gamma = gamma
        self.clip = clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.__init_networks(actor_architecture, critic_architecture)

        self.memory = PPOAgent.PPOBuffer(buffer_size)

    def __init_networks(self, actor_architecture : Type[nn.Module], critic_architecture : Type[nn.Module]):
        # TODO add parameter to control lr in optimisers

        self.actor = actor_architecture()
        self.critic = critic_architecture()

        self.actor_optimiser = Adam(self.actor.parameters())
        self.critic_optimiser = Adam(self.critic.parameters())

    def __get_action_logits(self, states) -> npt.NDArray[np.float_]:
        # TODO move this somewhere else later
        cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
        cov_mat = torch.diag(cov_var)

        mean = self.actor(states)
        distribution = MultivariateNormal(mean, cov_mat)
        action = distribution.sample()
        return action.detach().numpy, distribution.log_prob(action).detach()
    
    def __get_stochastic_action(self, states):
        pass

    def __evaluate(self, states) -> torch.Tensor:
        return self.critic(states).squeeze()

    def get_action(self, state, legal_mask=None, greedy=False): 
        action, logits = self.__get_action_logits(state)
        # TODO Add caching logic
        return action

    def update(self, state, action, reward, new_state, is_terminal):
        # TODO create issue to rename `remember` in DQN to `__update_memory`

        # TODO This piece of code puts the actor network to work twice
        # we should probably cache the logits from the last time we called get_action
        # and check if the states match, if they do, use cached value otherwise recalculate
        _, action_logits = self.__get_action_logits(state)
        buffer_full = self.memory._update(state, action, action_logits, reward, is_terminal)

        if buffer_full:
            self.memory._calculate_rewards_to_go()
            self.__train()
            self.memory._reset()

    def __train(self):
        states, actions_logits, rewards_to_go = self.memory._get_tensors()
        V = self.__evaluate(states)
        A = rewards_to_go - V.detach()

        # TODO is this neccessary?
        # Normalize advantages
        A = (A - A.mean()) / (A.std() + 1e-10)

        for _ in range(self.n_epochs):
            current_V = self.__evaluate(states)
            _, current_actions_logits = self.__get_action_logits(states)

            ratios = torch.exp(current_actions_logits - actions_logits)

            # TODO !!!! Rename this!
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A

            # train actor
            # negative sign as we are performing gradient **ascent**
            actor_loss = (-torch.min(surr1, surr2)).mean()
            self.actor_optimiser.zero_grad()

            # TODO read more about retain_graph, is it neccessary?
            # perhaps it's only useful is actor and critic share parameters
            actor_loss.backward(retain_graph=True)
            self.actor_optimiser.step()

            # train critic
            # TODO should this typing be left here? intellisense doesn't seem
            # to recognise it's a tensor (says it's of type 'any')
            critic_loss : torch.Tensor = nn.MSELoss()(current_V, rewards_to_go)
            self.critic_optimiser.zero_grad()    
            critic_loss.backward()    
            self.critic_optimiser.step()

    @classmethod
    def load(cls, path: str):
        pass

    def save(self, path: str) -> None:
        pass