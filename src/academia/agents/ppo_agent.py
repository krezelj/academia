from typing import Optional, Type

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from .base import Agent

# TODO annotate all paramter types
# TODO annotate all return types
# TODO add documentation

class PPOAgent(Agent):

    class PPOBuffer:

        __slots__ = ['buffer_size', '__update_counter', '__episode_length_counter', 'states', 'actions', 'actions_logits', 'rewards', 'rewards_to_go', 'episode_lengths']

        def __init__(self, buffer_size):
            self._reset()
            self.buffer_size = buffer_size


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
            

        def _calculate_rewards_to_go(self, gamma : float):
            t_offset = 0 
            for episode_length in self.episode_lengths:
                discounted_reward = 0
                discounted_rewards = []
                for t in range(episode_length - 1, -1, -1):
                    discounted_reward = self.rewards[t + t_offset] + gamma * discounted_reward
                    discounted_rewards.insert(0, discounted_reward)
                self.rewards_to_go.extend(discounted_rewards)
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
            # converting a list to a tensor is slow; pytorch suggests
            # converting to numpy array first
            states_t = torch.tensor(np.array(self.states), dtype=torch.float)
            actions_t = torch.tensor(np.array(self.actions), dtype=torch.float)
            actions_logits_t = torch.tensor(np.array(self.actions_logits), dtype=torch.float)
            rewards_to_go_t = torch.tensor(np.array(self.rewards_to_go), dtype=torch.float)
            return states_t, actions_t, actions_logits_t, rewards_to_go_t

    __slots__ = ['n_actions', 'alpha', 'gamma', 'epsilon', 'epsilon_decay', 'min_epsilon',
                 'clip', 'actor', 'critic', 'memory', 'batch_size', 'n_epochs',
                 '__covariance_matrix', '__action_logit_cache']

    def __init__(self, 
                 actor_architecture: Type[nn.Module],
                 critic_architecture: Type[nn.Module],
                 buffer_size : int,
                 batch_size : int,
                 n_epochs: int,
                 n_actions: int,
                 clip: float = 0.2,
                 gamma: float = 0.99, 
                 epsilon: float = 1.,
                 epsilon_decay: float = 0.99,
                 min_epsilon: float = 0.01,
                 random_state = Optional[int]) -> None:
        # TODO make network weights based on random state
        super(PPOAgent, self).__init__(n_actions, epsilon, min_epsilon, epsilon_decay, gamma, random_state)
        self.clip = clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.memory = PPOAgent.PPOBuffer(buffer_size)
        self.__init_networks(actor_architecture, critic_architecture)
        # TODO parametrise `fill_value`
        self.__covariance_matrix = torch.diag(torch.full(size=(self.n_actions,), fill_value=0.5))


    def __init_networks(self, actor_architecture : Type[nn.Module], critic_architecture : Type[nn.Module]):
        # TODO add parameter to control lr in optimisers
        self.actor = actor_architecture()
        self.critic = critic_architecture()

        self.actor_optimiser = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=3e-4)


    def __evaluate(self, states, actions) -> torch.Tensor:
        V = self.critic(states).squeeze()
        mean = self.actor(states)
        distribution = MultivariateNormal(mean, self.__covariance_matrix)
        actions_logits = distribution.log_prob(actions)
        return V, actions_logits


    def __get_action_with_logits(self, states, greedy=False) -> npt.NDArray[np.float_]:
        mean = self.actor(states)
        distribution = MultivariateNormal(mean, self.__covariance_matrix)
        if greedy:
            return mean.detach().numpy(), distribution.log_prob(mean).detach()
        
        action = distribution.sample()
        return action.detach().numpy(), distribution.log_prob(action).detach()


    def get_action(self, state, legal_mask=None, greedy=False): 
        # PPOAgent currently doesn't support legal_masks 
        action, action_logit = self.__get_action_with_logits(state, greedy)
        # TODO add better caching logic
        # so that we can check if the logit corresponds to the the same state
        self.__action_logit_cache = action_logit
        return action


    def update(self, state, action, reward, new_state, is_terminal):
        buffer_full = self.memory._update(state, action, self.__action_logit_cache, reward, is_terminal)
        if buffer_full:
            self.memory._calculate_rewards_to_go(self.gamma)
            self.__train()
            self.memory._reset()


    def __train(self):
        states, actions, actions_logits, rewards_to_go = self.memory._get_tensors()
        V, _ = self.__evaluate(states, actions)
        A = rewards_to_go - V.detach()
        A = (A - A.mean()) / (A.std() + 1e-10)

        for _ in range(self.n_epochs):
            current_V, current_actions_logits = self.__evaluate(states, actions)
            ratios = torch.exp(current_actions_logits - actions_logits)

            surrogate_1 = ratios * A
            surrogate_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A

            # negative sign since we are performing gradient **ascent**
            actor_loss = (-torch.min(surrogate_1, surrogate_2)).mean()
            self.actor_optimiser.zero_grad()

            # why retain_graph=True?
            # good explanation: https://t.ly/NjM38 (stackoverflow)
            # short explanation: if actor and critic models share paramets (i.e. layers) (not uncommon)
            # and we perform backward pass on actor_loss we will lose the values from the forward pass 
            # in the shared layers (default pytorch behaviour that saves memory)
            # however this would result in an error being thrown when we perform 
            # a backward pass on the critic_loss. 
            actor_loss.backward(retain_graph=True)
            self.actor_optimiser.step()

            # added typing since return type of MSELoss.__call__ is `Any`
            critic_loss : torch.Tensor = nn.MSELoss()(current_V, rewards_to_go)
            self.critic_optimiser.zero_grad()    
            critic_loss.backward()    
            self.critic_optimiser.step()


    @classmethod
    def load(cls, path: str):
        pass


    def save(self, path: str) -> None:
        pass