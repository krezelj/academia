import os
import zipfile
import tempfile
from typing import Any, Literal, Optional, Tuple, Type, Union
import json

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Categorical

from .base import Agent

# TODO Add KL estimation
# TODO Add GAES
# TODO Add lr scheduler
# TODO Add legal mask

# PPO improvements branch todo
# TODO [x] remove slots
# TODO [x] address other todos scatter around the code
# TODO [x] add CUDA
# TODO [ ] add documentation

class PPOAgent(Agent):

    class PPOBuffer:

        def __init__(self, 
                     n_steps: Optional[int] = None, 
                     n_episodes: Optional[int] = None) -> None:
            if (n_steps is None and n_episodes is None)\
                    or (n_steps is not None and n_episodes is not None):
                raise ValueError("Exactly one of n_steps and n_episodes must be not None")
            self.reset()
            self.n_steps = n_steps
            self.n_episodes = n_episodes

        def __is_full(self):
            if self.n_episodes is not None and self.episode_counter >= self.n_episodes:
                return True
            if self.n_steps is not None and self.steps_counter >= self.n_steps:
                return True
            return False

        def update(self, 
                    state: Any, 
                    action: Any, 
                    action_logits: float, 
                    reward: float, 
                    is_terminal: bool) -> bool:
            self.steps_counter += 1
            self.episode_length_counter += 1

            self.states.append(state)
            self.actions.append(action)
            self.actions_logits.append(action_logits)
            self.rewards.append(reward)

            if is_terminal:
                self.episode_counter += 1
                self.episode_lengths.append(self.episode_length_counter)
                self.episode_length_counter = 0

            # we are also checking if the state is terminal to avoid updating the agent
            # before the end of the episode when using n_steps instead of n_episodes
            return is_terminal and self.__is_full()

        def calculate_rewards_to_go(self, gamma: float) -> None:
            t_offset = 0 
            for episode_length in self.episode_lengths:
                discounted_reward = 0
                discounted_rewards = []
                for t in range(episode_length - 1, -1, -1):
                    discounted_reward = self.rewards[t + t_offset] + gamma * discounted_reward
                    discounted_rewards.insert(0, discounted_reward)
                self.rewards_to_go.extend(discounted_rewards)
                t_offset += episode_length

        def reset(self) -> None:
            self.episode_length_counter = 0
            self.steps_counter = 0
            self.episode_counter = 0

            self.states = []
            self.actions = []
            self.actions_logits = []
            self.rewards = []
            self.rewards_to_go = []
            self.episode_lengths = []

        def get_tensors(self, device) \
                -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
            # converting a list to a tensor is slow; pytorch suggests converting to numpy array first
            states_t = torch.tensor(np.array(self.states), dtype=torch.float).to(device)
            actions_t = torch.tensor(np.array(self.actions), dtype=torch.float).to(device)
            actions_logits_t = torch.tensor(np.array(self.actions_logits), dtype=torch.float).to(device)
            rewards_to_go_t = torch.tensor(np.array(self.rewards_to_go), dtype=torch.float).to(device)
            return states_t, actions_t, actions_logits_t, rewards_to_go_t

    def __init__(self, 
                 discrete: bool,
                 actor_architecture: Type[nn.Module],
                 critic_architecture: Type[nn.Module],
                 batch_size: int,
                 n_epochs: int,
                 n_actions: int,
                 n_steps: Optional[int] = None,
                 n_episodes: Optional[int] = None,
                 clip: float = 0.2,
                 lr: float = 3e-4,
                 covariance_fill: float = 0.5,
                 entropy_coefficient: float = 0.01,
                 gamma: float = 0.99, 
                 epsilon: float = 1.,
                 epsilon_decay: float = 0.99,
                 min_epsilon: float = 0.01,
                 random_state: Optional[int] = None,
                 device: Literal['cpu', 'cuda'] = 'cpu'
                 ) -> None:
        super(PPOAgent, self).__init__(n_actions, epsilon, min_epsilon, epsilon_decay, gamma, random_state)
        self.discrete = discrete
        self.clip = clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.actor_architecture = actor_architecture
        self.critic_architecture = critic_architecture
        self.lr = lr
        self.entropy_coefficient = entropy_coefficient
        self.buffer = PPOAgent.PPOBuffer(n_steps=n_steps, n_episodes=n_episodes)

        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if random_state is not None:
            torch.manual_seed(random_state)
        self.__init_networks()
        self.__covariance_matrix = torch.diag(torch.full(size=(self.n_actions,), fill_value=covariance_fill))
        self.__action_logit_cache = 0

    def __init_networks(self) -> None:
        self.actor = self.actor_architecture()# .to(self.device)
        self.critic = self.critic_architecture()# .to(self.device)

        self.actor_optimiser = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=self.lr)

    def __evaluate(self, states: torch.FloatTensor, actions: torch.FloatTensor) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        V = self.critic(states).squeeze(dim=1)
        if self.discrete:
            pi = self.actor(states)
            distribution = Categorical(pi)
        else:
            mean = self.actor(states)
            distribution = MultivariateNormal(mean, self.__covariance_matrix)

        actions_logits = distribution.log_prob(actions)
        return V, actions_logits, distribution.entropy()

    def __get_discrete_action_with_logits(self, states: torch.FloatTensor, legal_mask, greedy) \
            -> Tuple[npt.NDArray, torch.FloatTensor]:
        pi = self.actor(states)
        distribution = Categorical(pi)
        if greedy:
            action = torch.argmax(pi).detach().numpy().reshape(1,)
            action_logit = torch.full((len(states),), fill_value=1.0)
            return action, action_logit
        action = distribution.sample()
        return action.detach().numpy(), distribution.log_prob(action).detach()

    def __get_continuous_action_with_logits(self, states: torch.FloatTensor, greedy) \
            -> Tuple[npt.NDArray, torch.FloatTensor]:
        mean = self.actor(states)
        distribution = MultivariateNormal(mean, self.__covariance_matrix)
        if greedy:
            return mean.detach().numpy(), distribution.log_prob(mean).detach()
        action = distribution.sample()
        return action.detach().numpy(), distribution.log_prob(action).detach()

    def __get_action_with_logits(self, states: torch.FloatTensor, legal_mask=None, greedy=False):
        with torch.no_grad():
            if self.discrete:
                return self.__get_discrete_action_with_logits(states, legal_mask, greedy)
            else:
                return self.__get_continuous_action_with_logits(states, greedy)

    def get_action(self, state: Any, legal_mask=None, greedy=False) -> Union[float, int]:
        # PPOAgent currently doesn't support legal_masks TODO add note in documentation
        
        # in `get_action` we will always receive a single state
        # but we prefer to operate on batches of states so we add one dimension
        # to `state`` so that it behaves like a batch with single sample
        state = torch.unsqueeze(torch.tensor(state), dim=0)
        action, action_logit = self.__get_action_with_logits(state, legal_mask, greedy)

        # however converting the state to a batch means we have to 'unbatch' action (and logits). 
        # Otherwise gym environments return new states as batches which we try to unsqueeze again
        # and this leads to shape errors during inference.
        action = action[0]
        action_logit = action_logit.item()
        self.__action_logit_cache = action_logit
        return action

    def update(self, 
               state: Any, 
               action: Any, 
               reward: float, 
               new_state: Any, 
               is_terminal: bool) -> None:
        buffer_full = self.buffer.update(state, action, self.__action_logit_cache, reward, is_terminal)
        if buffer_full:
            self.buffer.calculate_rewards_to_go(self.gamma)
            self.__train()
            self.buffer.reset()

    def __train(self) -> None:
        self.actor.to(self.device)
        self.critic.to(self.device)

        states, actions, actions_logits, rewards_to_go = self.buffer.get_tensors(self.device)
        V, _, _ = self.__evaluate(states, actions)
        A = rewards_to_go - V.detach()
        A = (A - A.mean()) / (A.std() + 1e-10)

        for _ in range(self.n_epochs):
            idx_permutation = np.arange(self.buffer.steps_counter)
            self._rng.shuffle(idx_permutation)
            n_batches = np.ceil(self.buffer.steps_counter / self.batch_size).astype(np.int32)
            for batch_idx in range(n_batches):
                idx_in_batch = idx_permutation[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                batch_states = states[idx_in_batch]
                batch_actions = actions[idx_in_batch]
                batch_actions_logits = actions_logits[idx_in_batch]
                batch_A = A[idx_in_batch]
                batch_rewards_to_go = rewards_to_go[idx_in_batch]

                current_V, current_actions_logits, entropy = self.__evaluate(batch_states, batch_actions)
                ratios = torch.exp(current_actions_logits - batch_actions_logits)

                surrogate_1 = ratios * batch_A
                surrogate_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_A

                # negative sign since we are performing gradient **ascent**
                actor_loss = (-torch.min(surrogate_1, surrogate_2)).mean()
                actor_loss -= self.entropy_coefficient * entropy.mean()

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

                critic_loss = nn.MSELoss()(current_V, batch_rewards_to_go)
                self.critic_optimiser.zero_grad()    
                critic_loss.backward()    
                self.critic_optimiser.step()
        
        self.actor.to('cpu')
        self.critic.to('cpu')

    @classmethod
    def load(cls, path: str) -> 'PPOAgent':
        if not path.endswith('.zip'):
            path += '.agent.zip'
        zf = zipfile.ZipFile(path)

        with tempfile.TemporaryDirectory() as tempdir:
            zf.extractall(tempdir)
            
            actor_params = torch.load(os.path.join(tempdir, 'actor.pth'))
            critic_params = torch.load(os.path.join(tempdir, 'critic.pth'))
            
            with open(os.path.join(tempdir, 'state.agent.json'), 'r') as file:
                agent_state = json.load(file)
            agent_state: dict
            buffer_state: dict = agent_state.pop('buffer')
            actor_architecture = cls.get_type(agent_state.pop('actor_architecture'))
            critic_architecture = cls.get_type(agent_state.pop('critic_architecture'))
            rng_state = agent_state.pop('random_state')
            covariance_matrix = torch.tensor(agent_state.pop('__covariance_matrix'))
            action_logit_cache = agent_state.pop('__action_logit_cache')

            del agent_state['actor']
            del agent_state['critic']

            agent = cls(
                actor_architecture = actor_architecture,
                critic_architecture = critic_architecture,
                n_steps = buffer_state['n_steps'],
                n_episodes = buffer_state['n_episodes'],
                **agent_state
            )

            agent.actor.load_state_dict(actor_params)
            agent.actor.eval()

            agent.critic.load_state_dict(critic_params)
            agent.critic.eval()

            agent._rng.bit_generator.state = rng_state
            agent.__covariance_matrix = covariance_matrix
            agent.__action_logit_cache = action_logit_cache

            agent.buffer.states = [np.array(s) for s in buffer_state['states']]
            del buffer_state['states']
            for attribute_name, value in buffer_state.items():
                setattr(agent.buffer, attribute_name, value)

        return agent

    def save(self, path: str) -> str:
        if not path.endswith('.zip'):
            path += '.agent.zip'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with zipfile.ZipFile(path, 'w') as zf:
            actor_temp = tempfile.NamedTemporaryFile(delete=False)
            torch.save(self.actor.state_dict(), actor_temp)

            critic_temp = tempfile.NamedTemporaryFile(delete=False)
            torch.save(self.critic.state_dict(), critic_temp)

            agent_state = {
                'n_actions': self.n_actions,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon,
                'discrete': self.discrete,
                'clip': self.clip,
                'lr': self.lr,
                'entropy_coefficient': self.entropy_coefficient,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'device': str(self.device),
                'actor_architecture': self.get_type_name_full(self.actor_architecture),
                'critic_architecture': self.get_type_name_full(self.critic_architecture),
                'random_state': self._rng.bit_generator.state,
                '__covariance_matrix': self.__covariance_matrix.tolist(),
                '__action_logit_cache': self.__action_logit_cache,
                'epsilon': self.epsilon if type(self.epsilon) is float else self.epsilon.item(),
                'buffer': None,
                'actor': None,
                'critic': None,
            }

            # the state of the buffer should only be saved up to the last full episode
            n_valid_steps = int(np.sum(self.buffer.episode_lengths).item())
            buffer_state = {
                'n_steps': self.buffer.n_steps,
                'n_episodes': self.buffer.n_episodes,
                'steps_counter': n_valid_steps,
                'episode_length_counter': 0,
                'episode_counter': self.buffer.episode_counter,
                'states': [state.tolist() for state in self.buffer.states[:n_valid_steps+1]],
                'actions': [a.item() for a in np.array(self.buffer.actions[:n_valid_steps+1])],
                'actions_logits': self.buffer.actions_logits[:n_valid_steps+1],
                'rewards': [r.item() for r in np.array(self.buffer.rewards[:n_valid_steps+1])],
                'rewards_to_go': self.buffer.rewards_to_go[:n_valid_steps+1],
                'episode_lengths': self.buffer.episode_lengths
            }

            agent_state['buffer'] = buffer_state

            agent_temp = tempfile.NamedTemporaryFile(delete=False, mode='w')
            json.dump(agent_state, agent_temp, indent=4)
            agent_temp.flush()  # otherwise file is empty

            zf.write(actor_temp.name, 'actor.pth')
            zf.write(critic_temp.name, 'critic.pth')
            zf.write(agent_temp.name, 'state.agent.json')

            actor_temp.close()
            critic_temp.close()
            agent_temp.close()

        return os.path.abspath(path)
            
