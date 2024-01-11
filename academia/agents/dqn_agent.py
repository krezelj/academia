from collections import deque, namedtuple
from typing import Type, Optional, Any, Literal
import os
import zipfile
import tempfile
import json
import logging

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import EpsilonGreedyAgent


_logger = logging.getLogger('academia.agents')


class DQNAgent(EpsilonGreedyAgent):
    """
    Class representing a Deep Q-Network (DQN) agent for reinforcement learning tasks.

    The DQNAgent class implements the Deep Q-Network (DQN) algorithm for reinforcement learning tasks.
    It uses a neural network to approximate the Q-values of actions in a given environment. The agent
    learns from experiences stored in a replay memory and performs updates to its Q-values during
    training episodes. The target network is soft updated to stabilize training.

    Args:
        nn_architecture: Type of neural network architecture to be used.
        n_actions: Number of possible actions in the environment.
        gamma: Discount factor for future rewards. Defaults to 0.99.
        epsilon: Initial exploration-exploitation trade-off parameter. Defaults to 1.0.
        epsilon_decay: Decay factor for epsilon over time. Defaults to 0.995.
        min_epsilon: Minimum epsilon value to ensure exploration. Defaults to 0.01.
        batch_size: Size of the mini-batch used for training. Defaults to 64.
        random_state: Seed for random number generation. Defaults to ``None``.
        replay_memory_size: Maximum size of the replay memory. Defaults to 100000.
        lr: Learning rate for the optimizer. Defaults to 0.0005.
        tau: Interpolation parameter for target network soft updates. Defaults to 0.001.
        update_every: Frequency of network updates. Defaults to 3.
        device: Device to use for training. Defaults to ``cpu``.
    
    Attributes:
        nn_architecture (Type[nn.Module]): Type of neural network architecture to be used.
        epsilon (float): Exploration-exploitation trade-off parameter.
        min_epsilon (float): Minimum value for epsilon during exploration.
        epsilon_decay (float): Decay rate for epsilon.
        n_actions (int): Number of possible actions in the environment.
        gamma (float): Discount factor.
        memory (deque): Replay memory used to store experiences for training.
        batch_size (int): Size of the mini-batch used for training.
        network (nn.Module): Neural network used to approximate Q-values.
        target_network (nn.Module): Target network used to stabilize training.
        optimizer (optim.Optimizer): Optimizer used for training.
        experience (namedtuple): Named tuple representing an experience tuple which stores state, action, 
            reward, new_state, and done.
        train_step (int): Counter for the number of training steps performed.
        replay_memory_size (int): Maximum size of the replay memory.
        lr (float): Learning rate for the optimizer.
        tau (float): Interpolation parameter for target network soft updates.
        update_every (int): Frequency of network updates.
        device (Literal['cpu', 'cuda']): Device used for training.

    Examples:
        >>> from academia.agents import DQNAgent
        >>> from academia.environments import DoorKey
        >>> # Import custom neural network architecture
        >>> from academia.utils.models import door_key
        >>>
        >>> # Create an environment:
        >>> env = DoorKey(difficulty=0, append_step_count=True)
        >>> # Create an instance of the DQNAgent class with
        >>> # custom neural network architecture
        >>> dqn_agent = DQNAgent(
        >>>     nn_architecture=door_key.MLPStepDQN,
        >>>     n_actions=DoorKey.N_ACTIONS,
        >>>     gamma=0.99,
        >>>     epsilon=1.0,
        >>>     epsilon_decay=0.99,
        >>>     min_epsilon=0.01,
        >>>     batch_size=64,
        >>> )
        >>> # Training loop: Update the agent using experiences
        >>> # (state, action, reward, new_state, done)
        >>> num_episodes = 100
        >>> for episode in range(num_episodes):
        >>>    state = env.reset()
        >>>    done = False
        >>>    while not done:
        >>>        action = dqn_agent.get_action(state)
        >>>        new_state, reward, terminated = env.step(action)
        >>>        if terminated:
        >>>            done = True 
        >>>        dqn_agent.update(state, action, reward, new_state, done)
        >>>        state = new_state
        >>>
        >>> # Save the agent's state dictionary to a file
        >>> dqn_agent.save('dqn_agent')
        >>>
        >>> # Load the agent's state dictionary from a file
        >>> dqn_agent = DQNAgent.load('dqn_agent')

    Note:
        - Ensure that the custom neural network architecture passed to the constructor inherits 
          from ``torch.nn.Module`` and is appropriate for the task.
        - The agent's exploration-exploitation strategy is based on epsilon-greedy method.
        - The :func:`__soft_update_target` method updates the target network weights from the main network's
          weights based on strategy target_weights = :attr:`tau` * main_weights + (1 - :attr:`tau`) *
          target_weights, where :attr:`tau` << 1.
        - It is recommended to adjust hyperparameters such as gamma, epsilon, epsilon_decay, and batch_size
          based on the specific task and environment.
    """
    def __init__(self, nn_architecture: Type[nn.Module],
                 n_actions: int,
                 gamma: float = 0.99, epsilon: float = 1.,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01,
                 batch_size: int = 64, random_state: Optional[int] = None,
                 replay_memory_size: int = 100000,
                 lr: float = 0.0005,
                 tau: float = 0.001,
                 update_every: int = 3,
                 device: Literal['cpu', 'cuda'] = 'cpu'
                 ):
        super(DQNAgent, self).__init__(epsilon=epsilon, min_epsilon=min_epsilon,
                                       epsilon_decay=epsilon_decay,
                                       n_actions=n_actions, gamma=gamma, random_state=random_state)
        self.replay_memory_size = replay_memory_size
        self.lr = lr
        self.tau = tau
        self.update_every = update_every
        self.memory = deque(maxlen=self.replay_memory_size)
        self.batch_size = batch_size
        self.nn_architecture = nn_architecture
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward",
                                                                "new_state", "done"])
        self.train_step = 0

        if device == 'cuda' and not torch.cuda.is_available():
            _logger.warning('CUDA is not available. Using CPU instead.')
            device = torch.device('cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

        if random_state is not None:
            torch.manual_seed(random_state)
        self.__build_network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-4)

    def __build_network(self):
        """
        Builds the neural network architectures for both the main and target networks.

        The method creates instances of the neural network specified by nn_architecture and
        initializes the optimizer with the Adam optimizer. It also initializes the target
        network with the same architecture and loads its initial weights from the main
        network. The target network is set to evaluation mode during training.

        Note:
            - The neural networks are moved to the appropriate device (CPU or CUDA) using the to(device) method.
            - The target network is initialized with the same architecture as the main network with same weights.
        """
        self.network = self.nn_architecture()
        self.network.to(self.device)

        self.target_network = self.nn_architecture()
        self.target_network.to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def __remember(self, state: Any, action: int, reward: float, new_state: Any, done: bool):
        """
        Stores an experience tuple in the replay memory.

        Args:
            state: Current state of the agent in the environment.
            action: Action taken by the agent in the current state.
            reward: Reward received by the agent after taking the action.
            new_state: Next state of the agent after taking the action.
            done: A boolean flag indicating if the episode has terminated or truncated after the action.
        """
        e = self.experience(state, action, reward, new_state, done)
        self.memory.append(e)

    def get_action(self, state: Any, legal_mask: npt.NDArray[np.int32] = None, greedy: bool = False) -> int:
        """
        Selects an action based on the current state using the epsilon-greedy strategy.

        Args:
            state: The current state representation used to make the action selection decision.
            legal_mask: A binary mask indicating the legality of actions.
                If provided, restricts the agent's choices to legal actions.
            greedy: A boolean flag indicating whether to force a greedy action selection.
                If True, the function always chooses the action with the highest Q-value, ignoring exploration.

        Returns:
            The index of the selected action.
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.network.eval()

        with torch.no_grad():
            q_val_act = self.network(state).to(self.device)
        self.network.train()

        if legal_mask is not None:
            legal_mask = torch.from_numpy(legal_mask).float().to(self.device)
            q_val_act = (q_val_act - torch.min(q_val_act)) * legal_mask \
                + legal_mask
            
        if self._rng.uniform() > self.epsilon or greedy:
            return torch.argmax(q_val_act).item()
        
        elif legal_mask is not None:
            legal_mask_cpu = torch.Tensor.cpu(legal_mask)
            # convert to numpy array to avoid ValueError
            proba = np.array(legal_mask_cpu/legal_mask_cpu.sum())
            return self._rng.choice(np.arange(0, self.n_actions), size=1, p=proba)[0]
        else:
            return self._rng.integers(0, self.n_actions)

    def __soft_update_target(self):
        """
        Updates the target network's weights with the main network's weights.

        It uses soft max strategy so target_weights = :attr:`tau` * main_weights + (1 - :attr:`tau`) * target_weights, where :attr:`tau` << 1.
        Small value of :attr:`tau` still covers the statement that target values supposed to be fixed to prevent moving target problem
        """
        for network_params, target_params in zip(self.network.parameters(), 
                                                 self.target_network.parameters()):
            target_params.data.copy_(self.tau * network_params.data \
                                     + (1.0 - self.tau) * target_params.data)

    def update(self, state: Any, action: int, reward: float, new_state: Any, is_terminal: bool):
        """
        Updates the DQN network weights to better estimate Q-values of every action.

        Args:
            state: Current state of the environment.
            action: Action taken in the current state.
            reward: Reward received after taking the action.
            new_state: Next state of the environment after taking the action.
            is_terminal: A flag indicating whether the new state is a terminal state.
        """
        self.__remember(state=state, action=action, reward=reward, new_state=new_state,
                      done=is_terminal)
        self.train_step = (self.train_step + 1) % self.update_every
        if self.train_step == 0:
            if len(self.memory) >= self.batch_size:
                states, actions, rewards, new_states, dones = self.__replay()
                q_targets_next = self.target_network(new_states).detach().max(1)[0].unsqueeze(1)
                # bellman equation
                q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
                q_expected = self.network(states).gather(1, actions)
                loss = F.mse_loss(q_expected, q_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.__soft_update_target()

    def __replay(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a mini-batch from the replay memory and prepares states, actions, rewards, new_states
        casting them to tensors with appropriate type values and adding to device.

        Returns:
            Tuple containing tensors of states, actions, rewards, new_states, and dones.
        """
        batch_indices = self._rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(self.device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in batch if e is not None]))\
            .float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None])).float().to(self.device)
        return (states, actions, rewards, new_states, dones)

    def save(self, path: str) -> str:
        """
        Saves the state dictionary of the neural network model, target network model and agent parameters to 
        the specified file path.

        Args:
            path: Path to a file (including filename and extension) where the model's state
                dictionary will be saved.

        Returns:
            An absolute path to the saved file.
        """
        if not path.endswith('.zip'):
            path += '.agent.zip'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with zipfile.ZipFile(path, 'w') as zf:
            # network state
            network_temp = tempfile.NamedTemporaryFile(delete=False)
            torch.save(self.network.state_dict(), network_temp)
            # target network state
            target_network_temp = tempfile.NamedTemporaryFile(delete=False)
            torch.save(self.target_network.state_dict(), target_network_temp)
            # agent config
            agent_temp = tempfile.NamedTemporaryFile(delete=False, mode='w')

            memory_save_format = [{"state": exp.state.tolist(), "action": int(exp.action), "reward": float(exp.reward),
                    "new_state": exp.new_state.tolist(), "done": bool(exp.done)} for exp in self.memory]
            
            learner_state_dict = {
                'n_actions': self.n_actions,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon,
                'batch_size': self.batch_size,
                'nn_architecture': self.get_type_name_full(self.nn_architecture),
                'random_state': self._rng.bit_generator.state,
                'memory': memory_save_format,
                'device': str(self.device),
                'train_step': self.train_step,
                'replay_memory_size': self.replay_memory_size,
                'tau': self.tau,
                'update_every': self.update_every,
                'lr': self.lr,
            }
            json.dump(dict(learner_state_dict), agent_temp)
            agent_temp.flush()

            zf.write(network_temp.name, 'network.pth')
            zf.write(agent_temp.name, 'state.agent.json')
            zf.write(target_network_temp.name, 'target_network.pth')

            network_temp.close()
            target_network_temp.close()
            agent_temp.close()
            try:
                if os.path.isfile(network_temp.name):
                    os.remove(network_temp.name)
            except OSError:
                _logger.warn("Failed to delete a temporary file while saving agents.")
            try:
                if os.path.isfile(target_network_temp.name):
                    os.remove(target_network_temp.name)
            except OSError:
                _logger.warn("Failed to delete a temporary file while saving agents.")
            try:
                if os.path.isfile(agent_temp.name):
                    os.remove(agent_temp.name)
            except OSError:
                _logger.warn("Failed to delete a temporary file while saving agents.")

        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> 'DQNAgent':
        """
        Loads the state dictionary of the neural network model, target network model and agent parameters 
        from the specified file.

        Args:
            path: Path to a file from which to load the model's state dictionary.

        Returns:
            A loaded instance of ``PPOAgent``.
        """
        if not path.endswith('.zip'):
            path += '.agent.zip'
        zf = zipfile.ZipFile(path)
        with tempfile.TemporaryDirectory() as tempdir:
            zf.extractall(tempdir)
            # network state
            network_params = torch.load(os.path.join(tempdir, 'network.pth'))
            target_network_params = torch.load(os.path.join(tempdir, 'target_network.pth'))
            # agent config
            with open(os.path.join(tempdir, 'state.agent.json'), 'r') as file:
                params = json.load(file)

        nn_architecture = cls.get_type(params['nn_architecture'])
        del params['nn_architecture']
        rng_state = params.pop('random_state')
        memory = params.pop('memory')
        train_step = params.pop('train_step')
        experience = namedtuple("Experience", field_names=["state", "action", "reward",
                                                                    "new_state", "done"])
        replay_memory_size = params.pop('replay_memory_size')
        restored_memory = deque(maxlen=replay_memory_size)
        agent = cls(nn_architecture=nn_architecture, **params)
        agent._rng.bit_generator.state = rng_state
        for exp_dict in memory:
            state = np.array(exp_dict["state"], dtype=np.float32) 
            action = int(exp_dict["action"])
            reward = float(exp_dict["reward"])
            new_state = np.array(exp_dict["new_state"], dtype=np.float32)
            done = bool(exp_dict["done"])
            restored_memory.append(experience(state, action, reward, new_state, done))
        agent.memory = restored_memory
        agent.network.load_state_dict(network_params)
        agent.target_network.load_state_dict(target_network_params)
        agent.train_step = train_step
        return agent
