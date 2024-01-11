import logging
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

_logger = logging.getLogger('academia.agents')

class PPOAgent(Agent):
    """
    Class representing a Proximal Policy Optimization (PPO) agent for reinforcement learning tasks.
    Paper on PPO: https://arxiv.org/pdf/1707.06347.pdf

    Args:
        actor_architecture: Type of neural network architecture to be used for the actor.
        critic_architecture: Type of neural network architecture to be used. for the critic.
        n_actions: Number of possible actions in the environment.
        discrete: Whether the agent's action space is discrete. Defaults to ``True``.
        batch_size: The size of the minibatch used during training. Defaults to 64.
        n_epochs: Number of epochs per training. Defaults to 5.
        n_steps: Minimum number of steps to take between training sessions. Note that if the minimum
            is reached during an episode the episode will still finish and the remaining steps
            will be included in the buffer. If set to None :attr:`n_episodes` will be used instead. 
            Exactly one of :attr:`n_steps` and :attr:`n_episodes` must be not ``None``. 
            Defaults to ``None``.
        n_episodes: Number of episodes to take between training sessions. 
            If set to None :attr:`n_steps` will be used instead. 
            Exactly one of :attr:`n_steps` and :attr:`n_episodes` must be not ``None``. 
            Defaults to 10.
        clip: Clip rate hyperparameter from the PPO algorithm. Defaults to 0.2.
        lr: Learning rate used by (Adam) optimizers. The same value is used for both actor and critic.
            Defaults to ``3e-4``.
        covariance_fill: Value on the diagonal in the covariance matrix used to randomly sample
            continuous actions when :attr:`discrete` is ``False``. Defaults to 0.5.
        entropy_coefficient: Coefficient used to control the impact of entropy on the loss function.
            Defaults to 0.01.
        gamma: Discount factor for future rewards. Defaults to 0.99.
        random_state: Seed for random number generation. Defaults to ``None``.
        device: Device used for computation. Possible values are ``'cuda'`` and ``'cpu'``.
            Defaults to ``'cpu'``.
        
    
    Attributes:
        n_actions (int): Number of possible actions in the environment.
        gamma (float): Discount factor for future rewards.
        discrete (bool): Whether the agent's action space is discrete.
        clip (float): Clip rate hyperparameter from the PPO algorithm.
        lr (float): Learning rate used by (Adam) optimizers.
        entropy_coefficient (float): Coefficient used to control the impact of entropy on the loss function.
        batch_size (int): The size of the minibatch used during training.
        n_epochs (int): Number of epochs per training.
        device (Literal['cpu', 'cuda']): Device used for computation.
        buffer (PPOAgent.PPOBuffer): Trajectory buffer. This object contains all transitions gathered since
            the last training session.
        actor (nn.Module): Actor neural network.
        critic (nn.Module): Critic neural network.
        actor_architecture (Type[nn.Module]): 
            Type of neural network architecture to be used for the actor.
        critic_architecture (Type[nn.Module]): 
            Type of neural network architecture to be used for the critic.

    Examples:
        >>> from academia.agents import PPOAgent
        >>> from academia.environments import LavaCrossing
        >>> from academia.curriculum import LearningTask
        >>> from academia.utils.models import lava_crossing
        >>>
        >>> task = LearningTask(
        >>>     LavaCrossing, 
        >>>     env_args={'difficulty': 0}, 
        >>>     stop_conditions={'max_episodes': 100}
        >>> )
        >>> agent = PPOAgent(
        >>>     actor_architecture=lava_crossing.MLPActor,
        >>>     critic_architecture=lava_crossing.MLPCritic,
        >>>     n_actions=3
        >>> )
        >>> task.run(agent)

    Note:
        - PPOAgent currently does not support legal masks.
        - PPOAgent currently does not provide implementations for :func:`update_exploration` or
          :func:`reset_exploration` methods.
    """

    class PPOBuffer:
        """
        Class representing the buffer of PPOAgent

        Args:
            n_steps: Minimum number of steps to take between training sessions. Note that if the minimum
                is reached during an episode the episode will still finish and the remaining steps
                will be included in the buffer. If set to None :attr:`n_episodes` will be used instead. 
                Exactly one of :attr:`n_steps` and :attr:`n_episodes` must be not ``None``. 
                Defaults to ``None``.
            n_episodes: Number of episodes to take between training sessions. 
                If set to None :attr:`n_steps` will be used instead. 
                Exactly one of :attr:`n_steps` and :attr:`n_episodes` must be not ``None``. 
                Defaults to ``None``.
 
        Attributes:
            n_steps (int): Minimum number of steps to take between training sessions.
            n_episodes (int): Number of episodes to take between training sessions.
            episode_length_counter (int): Length of the currently running episode.
            steps_counter (int): Number of steps stored inside the buffer.
            episode_counter (int): Number of full episodes stored inside the buffer.
            states (list): List containing observed states.
            actions (list): List containing actions taken.
            actions_logits (list): List containing logits of actions taken.
            rewards (list): List of obtained rewards.
            rewards_to_go (list): List of discounted rewards. Note that it is only calculated right
                before the training and is cleared afterwards.
            episode_lengths (list): List containing the lengths of buffered episodes.
        """

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
            """
            Returns ``True`` is enough steps or episodes is stored for training.
            """
            if self.n_episodes is not None and self.episode_counter >= self.n_episodes:
                return True
            if self.n_steps is not None and self.steps_counter >= self.n_steps:
                return True
            return False

        def update(self, 
                    state: Any, 
                    action: Any, 
                    action_logit: float, 
                    reward: float, 
                    is_terminal: bool) -> bool:
            """
            Updates the buffer with the provided transition attriutes.

            Args:
                state: Observed state of the environment.
                action: Action taken by the agent.
                action_logit: Logit of the action taken by the agent.
                reward: Reward obtained by the agent.
                is_terminal: Whether the resulting new state is terminal.

            Returns:
                Whether the buffer is full and the current episode is terminated.
            """
            self.steps_counter += 1
            self.episode_length_counter += 1

            self.states.append(state)
            self.actions.append(action)
            self.actions_logits.append(action_logit)
            self.rewards.append(reward)

            if is_terminal:
                self.episode_counter += 1
                self.episode_lengths.append(self.episode_length_counter)
                self.episode_length_counter = 0

            # we are also checking if the state is terminal to avoid updating the agent
            # before the end of the episode when using n_steps instead of n_episodes
            return is_terminal and self.__is_full()

        def calculate_rewards_to_go(self, gamma: float) -> None:
            """
            Calculates the discounted rewards for each buffered episode.

            Args:
                gamma: Discount factor
            """
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
            """
            Clears the buffer and resets it to the initial state.
            """
            self.episode_length_counter = 0
            self.steps_counter = 0
            self.episode_counter = 0

            self.states = []
            self.actions = []
            self.actions_logits = []
            self.rewards = []
            self.rewards_to_go = []
            self.episode_lengths = []

        def get_tensors(self, device: Literal['cpu', 'cuda']) \
                -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
            """
            Calculates the discounted rewards for each buffered episode

            Args:
                device: Target computation device

            Returns:
                A 4-element tuple containing states, actions, actions logits and discounted rewards
                in that order converted to tensors.
            """
            # converting a list to a tensor is slow; pytorch suggests converting to numpy array first
            states_t = torch.stack(self.states).to(device)
            actions_t = torch.tensor(np.array(self.actions), dtype=torch.float).to(device)
            actions_logits_t = torch.tensor(np.array(self.actions_logits), dtype=torch.float).to(device)
            rewards_to_go_t = torch.tensor(np.array(self.rewards_to_go), dtype=torch.float).to(device)
            return states_t, actions_t, actions_logits_t, rewards_to_go_t

        def __len__(self) -> int:
            """
            Returns:
                Buffer size
            """
            return self.steps_counter

    def __init__(self, 
                 actor_architecture: Type[nn.Module],
                 critic_architecture: Type[nn.Module],
                 n_actions: int,
                 discrete: bool = True,
                 batch_size: int = 64,
                 n_epochs: int = 5,
                 n_steps: Optional[int] = None,
                 n_episodes: Optional[int] = 10,
                 clip: float = 0.2,
                 lr: float = 3e-4,
                 covariance_fill: float = 0.5,
                 entropy_coefficient: float = 0.01,
                 gamma: float = 0.99,
                 random_state: Optional[int] = None,
                 device: Literal['cpu', 'cuda'] = 'cpu'
                 ) -> None:
        super(PPOAgent, self).__init__(
            n_actions=n_actions,
            gamma=gamma, 
            random_state=random_state)
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
        elif device == 'cuda':
            _logger.warning("CUDA device not available. CPU will be used instead")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        if random_state is not None:
            torch.manual_seed(random_state)
        self.__init_networks()
        self.__covariance_matrix = torch.diag(torch.full(size=(self.n_actions,), fill_value=covariance_fill))
        self.__action_logit_cache = 0

    def __init_networks(self) -> None:
        """
        Initializes the actor and critic networks and optimizers
        """
        self.actor = self.actor_architecture()
        self.critic = self.critic_architecture()

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

    def __evaluate(self, states: torch.FloatTensor, actions: torch.FloatTensor) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Evaluates the provided states and actions to obtain state-values
        and actions logits using the current network parameters.
        """
        V = self.critic(states).squeeze(dim=1)
        if self.discrete:
            pi = torch.softmax(self.actor(states), dim=1)
            distribution = Categorical(pi)
        else:
            mean = self.actor(states)
            distribution = MultivariateNormal(mean, self.__covariance_matrix)

        actions_logits = distribution.log_prob(actions)
        return V, actions_logits, distribution.entropy()

    def __get_discrete_action_with_logits(self, 
                                          states: torch.FloatTensor, 
                                          greedy: bool) \
            -> Tuple[npt.NDArray, torch.FloatTensor]:
        """
        Gets an action and its logit for a given state assuming discrete action space.
        """
        pi = torch.softmax(self.actor(states), dim=1)
        distribution = Categorical(pi)
        if greedy:
            action = torch.argmax(pi).detach().numpy().reshape(1,)
            action_logit = torch.full((len(states),), fill_value=1.0)
            return action, action_logit
        action = distribution.sample()
        return action.detach().numpy(), distribution.log_prob(action).detach()

    def __get_continuous_action_with_logits(self, states: torch.FloatTensor, greedy: bool) \
            -> Tuple[npt.NDArray, torch.FloatTensor]:
        """
        Gets an action and its logit for a given state assuming continuous action space.
        """
        mean = self.actor(states)
        distribution = MultivariateNormal(mean, self.__covariance_matrix)
        if greedy:
            return mean.detach().numpy(), distribution.log_prob(mean).detach()
        action = distribution.sample()
        return action.detach().numpy(), distribution.log_prob(action).detach()

    def __get_action_with_logits(self, 
                                 states: torch.FloatTensor, 
                                 greedy: bool = False):
        """
        Gets an action and its logit for a given state.
        """
        with torch.no_grad():
            if self.discrete:
                return self.__get_discrete_action_with_logits(states, greedy)
            else:
                return self.__get_continuous_action_with_logits(states, greedy)

    def get_action(self, 
                   state: Any,
                   legal_mask: npt.NDArray[np.int32] = None,
                   greedy: bool = False) \
            -> Union[float, int]:
        """
        Selects an action based on the current state.

        Args:
            state: The current state representation used to make the action selection decision.
            legal_mask: A binary mask indicating the legality of actions.
                If provided, restricts the agent's choices to legal actions.
                Note that currently PPOAgent does not support legal masks.
            greedy: A boolean flag indicating whether to force a greedy action selection.

        Returns:
            The selected action.
        """
        
        # in `get_action` we will always receive a single state
        # but we prefer to operate on batches of states so we add one dimension
        # to `state`` so that it behaves like a batch with single sample
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), dim=0)
        action, action_logit = self.__get_action_with_logits(state, greedy)

        # however converting the state to a batch means we have to 'unbatch' action (and logits). 
        # Otherwise gym environments return new states as batches which we try to unsqueeze again
        # and this leads to shape errors during inference.
        action = action[0]
        action_logit = action_logit.item()
        self.__action_logit_cache = action_logit
        return action

    def update(self, 
               state: Any,
               action: int,
               reward: float, 
               new_state: Any,
               is_terminal: bool) -> None:
        """
        Updates the PPOAgent by saving the provided transition into its buffer.
        If the buffer is full it will also perform training on the actor and critic networks
        and clear the buffer.

        Args:
            state: Current state of the environment.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            new_state: Next state of the environment after taking the action.
                Note that PPOAgent does not actually use this value when updating.
            is_terminal: A flag indicating whether the new state is a terminal state.
        """
        state = torch.tensor(state, dtype=torch.float)
        buffer_full = self.buffer.update(state, action, self.__action_logit_cache, reward, is_terminal)
        if buffer_full:
            self.buffer.calculate_rewards_to_go(self.gamma)
            self.__train()
            self.buffer.reset()

    def __train(self) -> None:
        """
        Performs training on the actor and critic networks.
        """
        self.actor.to(self.device)
        self.critic.to(self.device)

        states, actions, actions_logits, rewards_to_go = self.buffer.get_tensors(self.device)
        V, _, _ = self.__evaluate(states, actions)
        A = rewards_to_go - V.detach()
        A = (A - A.mean()) / (A.std() + 1e-10)

        for _ in range(self.n_epochs):
            idx_permutation = np.arange(len(self.buffer))
            self._rng.shuffle(idx_permutation)
            n_batches = np.ceil(len(self.buffer) / self.batch_size).astype(np.int32)
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

                self.actor_optimizer.zero_grad()

                # why retain_graph=True?
                # good explanation: https://t.ly/NjM38 (stackoverflow)
                # short explanation: if actor and critic models share paramets (i.e. layers) (not uncommon)
                # and we perform backward pass on actor_loss we will lose the values from the forward pass 
                # in the shared layers (default pytorch behaviour that saves memory)
                # however this would result in an error being thrown when we perform 
                # a backward pass on the critic_loss. 
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                critic_loss = nn.MSELoss()(current_V, batch_rewards_to_go)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()    
                self.critic_optimizer.step()
        
        self.actor.to('cpu')
        self.critic.to('cpu')

    def update_exploration(self):
        """
        Updates the exploration parameter.

        Note:
            ``PPOAgent`` currently does not provide implementation for this method.
        """
        pass

    def reset_exploration(self, value):
        """
        Resets the exploration parameter to the specified value.

        Note:
            ``PPOAgent`` currently does not provide implementation for this method.
        """
        pass

    @classmethod
    def load(cls, path: str) -> 'PPOAgent':
        """
        Loads the state of the agent from the specified file path.

        Args:
            path: Path to a file from which to load the agent state.

        Returns:
            A loaded instance of ``PPOAgent``.
        """
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

            agent.buffer.states = [torch.tensor(s, dtype=torch.float) for s in buffer_state['states']]
            del buffer_state['states']
            for attribute_name, value in buffer_state.items():
                setattr(agent.buffer, attribute_name, value)

        return agent

    def save(self, path: str) -> str:
        """
        Saves the state of the agent to the specified file.

        Args:
            path: Path to a file to which the agent state will be saved.

        Returns:
            An absolute path to the saved file.
        """
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
                'states': [state.tolist() for state in self.buffer.states[:n_valid_steps]],
                'actions': [a.item() for a in np.array(self.buffer.actions[:n_valid_steps])],
                'actions_logits': self.buffer.actions_logits[:n_valid_steps],
                'rewards': [r.item() for r in np.array(self.buffer.rewards[:n_valid_steps])],
                'rewards_to_go': self.buffer.rewards_to_go[:n_valid_steps],
                'episode_lengths': self.buffer.episode_lengths
            }

            agent_state['buffer'] = buffer_state

            agent_temp = tempfile.NamedTemporaryFile(delete=False, mode='w')
            json.dump(agent_state, agent_temp)
            agent_temp.flush()  # otherwise file is empty

            zf.write(actor_temp.name, 'actor.pth')
            zf.write(critic_temp.name, 'critic.pth')
            zf.write(agent_temp.name, 'state.agent.json')

            actor_temp.close()
            critic_temp.close()
            agent_temp.close()
            try:
                if os.path.isfile(actor_temp.name):
                    os.remove(actor_temp.name)
            except OSError:
                _logger.warn("Failed to delete a temporary file while saving agents.")
            try:
                if os.path.isfile(critic_temp.name):
                    os.remove(critic_temp.name)
            except OSError:
                _logger.warn("Failed to delete a temporary file while saving agents.")
            try:
                if os.path.isfile(agent_temp.name):
                    os.remove(agent_temp.name)
            except OSError:
                _logger.warn("Failed to delete a temporary file while saving agents.")

        return os.path.abspath(path)
            
