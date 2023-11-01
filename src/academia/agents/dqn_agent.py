from collections import deque, namedtuple
from typing import Type, Optional
import os
import zipfile
import tempfile
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from academia.agents.base import Agent

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class DQNAgent(Agent):
    """
    Class: DQNAgent

    This class represents a Deep Q-Network (DQN) agent used for reinforcement learning tasks.

    Attributes:
        - `REPLAY_MEMORY_SIZE`: Maximum size of the replay memory.
        - `nn_architecture (Type[nn.Module])`: Type of neural network architecture to be used.
        - `n_actions (int)`: Number of possible actions in the environment.
        - `gamma (float)`: Discount factor for future rewards.
        - `epsilon (float)`: Exploration-exploitation trade-off parameter.
        - `epsilon_decay (float)`: Decay factor for epsilon over time.
        - `min_epsilon (float)`: Minimum epsilon value to ensure exploration.
        - `batch_size (int)`: Size of the batch used for training the DQN.
        - `network (nn.Module)`: Main DQN neural network used for estimating Q-values.
        - `target_network (nn.Module)`: Target DQN neural network used for computing target Q-values.
        - `optimizer (torch.optim)`: Optimizer used for training the neural network.
        - `memory (deque)`: Replay memory to store experiences for training. From there batches are
        sampled.

    Usage:
        ```python
        # Example Usage of DQNAgent

        from models import CartPoleMLP  # Import custom neural network architecture
        
        # Create an instance of the DQNAgent class with custom neural network architecture

        dqn_agent = DQNAgent(nn_architecture=CartPoleMLP, n_actions=2, gamma=0.99, epsilon=1.0,
                             epsilon_decay=0.99, min_epsilon=0.01, batch_size=64)
        
        # Training loop: Update the agent using experiences (state, action, reward, next_state, done)
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = dqn_agent.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    done = True 
                dqn_agent.update(state, action, reward, next_state, done)
                state = next_state
        ```
        
    Description:
        The `DQNAgent` class implements the Deep Q-Network (DQN) algorithm 
        for reinforcement learning tasks. It uses a neural network to 
        approximate the Q-values of actions in a given environment. 
        The agent learns from experiences stored in a replay memory 
        and performs updates to its Q-values during training episodes. 
        The target network is periodically updated to stabilize training.

    Notes:
        - Ensure that the custom neural network architecture passed to the constructor inherits 
        from `torch.nn.Module` and is appropriate for the task.
        - The agent's exploration-exploitation strategy is based on epsilon-greedy method.
        - The `update_target` method updates the target network weights from the main network's
        weights.
        - It is recommended to adjust hyperparameters such as `gamma`, `epsilon`, `epsilon_decay`
        and `batch_size` based on the specific task and environment.

    """

    REPLAY_MEMORY_SIZE = 100000
    LR = 0.0005
    TAU = 0.001  # interpolation parameter
    UPDATE_EVERY = 3

    def __init__(self, nn_architecture: Type[nn.Module],
                 n_actions: int,
                 gamma: float = 0.99, epsilon: float = 1.,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01,
                 batch_size: int = 64, random_state: Optional[int] = None
                 ):
        """
        Constructor method initializing the DQNAgent.

        Parameters:
            - `nn_architecture (Type[nn.Module])`: Type of neural network architecture to be used.
            - `n_actions (int)`: Number of possible actions in the environment.
            - `gamma (float)`: Discount factor for future rewards (default: 0.99).
            - `epsilon (float)`: Initial exploration-exploitation trade-off parameter (default: 1.0).
            - `epsilon_decay (float)`: Decay factor for epsilon over time (default: 0.99).
            - `min_epsilon (float)`: Minimum epsilon value to ensure exploration (default: 0.01).
            - `batch_size (int)`: Size of the mini-batch used for training the DQN (default: 64).

        """
        super(DQNAgent, self).__init__(epsilon=epsilon, min_epsilon=min_epsilon,
                                       epsilon_decay=epsilon_decay,
                                       n_actions=n_actions, gamma=gamma, random_state=random_state)
        self.memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.batch_size = batch_size
        self.nn_architecture = nn_architecture
        self.experience = namedtuple("Experience", field_names=["state","action",
                                                                "reward", "next_state", "done"])
        self.train_step = 0
        
        if random_state is not None:
            torch.manual_seed(random_state)
        self.__build_network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-4)

    def __build_network(self):
        """
        Builds the neural network architectures for both the main and target networks.

        Parameters:
            None
        
        Returns:
            None

        Description:
             The method creates instances of the neural network specified by `nn_architecture` and
             initializes the optimizer with the Adam optimizer. It also initializes the target
             network with the same architecture and loads its initial weights from the main
             network. The target network is set to evaluation mode during training.

        Notes:
            - The neural networks are moved to the appropriate device (CPU or CUDA) using the
            `to(device)` method.
            - The target network is initialized with the same architecture as the main network and
            its weights are loaded from the main network using `load_state_dict`.
            - The target network is set to evaluation mode (`eval()`) to ensure consistent target
            Q-value computations during training.

        """
        self.network = self.nn_architecture()
        self.network.to(device)

        self.target_network = self.nn_architecture()
        self.target_network.to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.LR)

    def __remember(self, state, action, reward, next_state, done):
        """
        Function: remember

        This function stores an experience tuple in the replay memory of the DQNAgent.

        Parameters:
            - `state`: Current state of the agent in the environment.
            - `action`: Action taken by the agent in the current state.
            - `reward`: Reward received by the agent after taking the action.
            - `next_state`: Next state of the agent after taking the action.
            - `done`: A boolean flag indicating if the episode has terminated or truncated after
            the action.

        Returns:
            None

        Description:
            The `remember` function appends the experience tuple `(state, action, reward,
            next_state, done)` to the replay memory of the DQNAgent. The replay memory is a data
            structure used to store and sample past experiences, allowing the agent to learn from
            previous interactions with the environment. If the replay memory exceeds its maximum
            size (`REPLAY_MEMORY_SIZE`), the function removes the oldest experience to make room
            for the new one.

        Notes:
            - `state` and `next_state` should be representations of the agent's observations in the
            environment.
            - `action` represents the action taken by the agent based on the current state.
            - `reward` is the numerical reward received by the agent after taking the specified
            action.
            - `done` is a boolean flag indicating whether the episode has terminated or truncated
            (`True`) or not (`False`).

        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def get_action(self, state, legal_mask=None, greedy=False) -> int:
        """
        This function selects an action based on the current state using the epsilon-greedy strategy.

        Parameters:
            - `state (array-like)`: The current state representation used to make the action
            selection decision.
            - `legal_mask (array-like, optional)`: A binary mask indicating the legality of actions.
            If provided, restricts the agent's choices to legal actions.
            - `greedy (bool, optional)`: A boolean flag indicating whether to force a greedy action
            selection. If True, the function always chooses the action with the highest Q-value,
            ignoring exploration.

        Returns:
            - `int`: The index of the selected action.

        Description:
            The `get_action` function selects an action for the agent based on the current state
            using the epsilon-greedy strategy. If the `legal_mask` is provided, it restricts the
            agent's choices to legal actions, ensuring that illegal actions are not considered.
            The `greedy` flag allows forcing a purely exploitative behavior, where the action with
            the highest Q-value is always chosen, ignoring exploration. If `greedy` is False or the
            exploration condition is met (based on epsilon value), the function chooses a random
            action with a probability of epsilon or explores other legal actions based on the
            provided `legal_mask`. If no legal mask is provided, the function considers all
            possible actions.

        Notes:
            - Ensure that the `state` input is compatible with the input size expected by the
            agent's neural network.
            - If the `legal_mask` is not provided, all actions are considered during action
            selection.
            - The `greedy` flag allows controlling the agent's exploration-exploitation behavior.
        """
        state = torch.from_numpy(state).float().to(device)
        self.network.eval()

        with torch.no_grad():
            q_val_act = self.network(state).to(device)
        self.network.train()

        if legal_mask is not None:
            legal_mask = torch.from_numpy(legal_mask).float().to(device)
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

        Parameters:
             None
         
        Returns:
             None

        Description:
            Updates the target network's weights with the main network's weights. This function
            synchronizes the parameters of the target network to match those of the main network,
            ensuring consistency between the two networks during the training process. It is called
            periodically to stabilize the training of the DQNAgent.

        Note:
            It is essential to call this function periodically, especially after a certain number of
            training steps, to ensure that the target network's weights are in line with the main
            network. Keeping the target network updated is crucial for stable and effective training
            of the DQN agent.

        """
        for network_params, target_params in zip(self.network.parameters(), 
                                                 self.target_network.parameters()):
            target_params.data.copy_(self.TAU * network_params.data \
                                     + (1.0 - self.TAU) * target_params.data)

    def update(self, state, action, reward: float, new_state, is_terminal: bool):
        """
        Parameters:
            - `state`: Current state of the environment.
            - `action`: Action taken in the current state.
            - `reward (float)`: Reward received after taking the action.
            - `new_state`: Next state of the environment after taking the action.
            - `is_terminal (bool)`: A flag indicating whether the new state is a terminal state.

        Returns:
            None

        Description:
            The `update` function is responsible for updating the DQN agent's Q-values based on the
            provided experience tuple. It adds the experience to the replay memory and performs a
            mini-batch update if the replay memory size reaches the specified batch size. The agent
            uses the Mean Squared Error (MSE) loss between predicted Q-values and target Q-values
            to update its neural network weights.

            If the `is_terminal` flag is `True`, the function increments the update counter and
            checks if it's time to update the target network weights. The target network is updated
            periodically to stabilize the learning process.

        Notes:
            - Ensure that the DQN agent has been properly initialized with the necessary parameters
            before calling this function.
            - The function performs an update only if the replay memory size reaches the specified
            batch size (`batch_size`).
        """
        self.__remember(state=state, action=action, reward=reward, next_state=new_state,
                      done=is_terminal)
        self.train_step = (self.train_step + 1) % self.UPDATE_EVERY
        if self.train_step == 0:
            if len(self.memory) >= self.batch_size:
                states, actions, rewards, next_states, dones = self.__replay()
                q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
                # bellman equation
                q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
                q_expected = self.network(states).gather(1, actions)
                loss = F.mse_loss(q_expected, q_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.__soft_update_target()

    def __replay(self) -> (torch.Tensor, torch.Tensor):
        """
        This function samples a mini-batch from the replay memory and prepares states and
        corresponding target Q-values for optimization.

        Returns:
            - `states (torch.Tensor)`: Tensor containing the states sampled from the replay memory.
                Shape: (batch_size, state_features)
            - `targets (torch.Tensor)`: Tensor containing the corresponding target Q-values for the
            sampled states and actions.
                Shape: (batch_size, num_actions)

        Description:
            The `replay` function is responsible for generating a mini-batch of experiences from the
            replay memory and computing target Q-values for each state-action pair. For each sampled
            experience tuple (state, action, reward, next_state, done), it calculates the target
            Q-value as follows:
                - If the episode is not done, the target Q-value is computed as the sum of the
                immediate reward and the discounted maximum Q-value of the next state according
                to the target network.
                - If the episode is done, the target Q-value is set equal to the immediate reward.

            After computing the target Q-values, the function constructs tensors of states and
            target Q-values, both of which are moved to the appropriate device (CPU or CUDA)
            for optimization. The states tensor represents the sampled states, and the targets
            tensor contains the corresponding target Q-values.

        Notes:
            - Ensure that the `replay` function is called when the replay memory contains enough
            experiences to form a mini-batch (i.e., when `len(self.memory) >= self.batch_size`).
            - The function constructs tensors from the sampled states and target Q-values, which
            can be directly used for training the neural network.
            - The `target_network` is used to estimate the maximum Q-value for the next state during
            target computation.

        """
        batch_indices = self._rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None]))\
            .float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None])).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def save(self, path: str) -> str:
        """
        Saves the state dictionary of the neural network model to the 
        specified file path.

        Parameters:
            - `path (str)`: The file path (including filename and extension) where the model's
            state dictionary will be saved.

        Returns:
            None

        Description:
              This method allows the user to save the learned
              parameters of the model, enabling later use or further training without 
              the need to retrain the network from scratch.

        Notes:
            - Ensure that the file path provided in `save_path` is writable and has appropriate
            permissions.
            - The saved file can be loaded later using the 'load' method to restore the model's
            state for inference or further training.

        """
        if not path.endswith('.zip'):
            path += '.agent.zip'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with zipfile.ZipFile(path, 'w') as zf:
            # network state
            network_temp = tempfile.NamedTemporaryFile()
            torch.save(self.network.state_dict(), network_temp)
            # agent config
            agent_temp = tempfile.NamedTemporaryFile()
            learner_state_dict = {
                'n_actions': self.n_actions,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon,
                'batch_size': self.batch_size,
                'nn_architecture': self.get_type_name_full(self.nn_architecture),
                'random_state': self._rng.bit_generator.state
            }
            with open(agent_temp.name, 'w') as file:
                json.dump(dict(learner_state_dict), file, indent=4)
            # zip both
            zf.write(network_temp.name, 'network.pth')
            zf.write(agent_temp.name, 'state.agent.json')
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> 'DQNAgent':
        """
        Loads the state dictionary of the neural network model from the specified file path. After
        loading the model, it sets the network to evaluation mode (`eval()`) to disable gradient
        computation, making the model ready for inference.

        Parameters:
            - `path (str)`: The file path from which to load the model's state dictionary.

        Returns:
            None

        Description:
            The `load` method allows loading a pre-trained neural network model's state dictionary
            from a specified file path. This method is particularly useful when you want to continue
            training from a pre-trained model or use a pre-trained model for making predictions in a
            real-time environment. After loading the model, it is set to evaluation mode, ensuring
            that gradients are not computed during inference, which reduces memory consumption and
            computation time.

        Notes:
            - Ensure that the `model_path` parameter points to a valid saved model file with the
            appropriate model architecture and state dictionary.
            - The method assumes that the model architecture and structure match the one used during
            the initial training.
            - It's recommended to call this method after initializing an instance of the `DQNAgent`
            class to load pre-trained weights before using the agent for inference or further training.
        """
        if not path.endswith('.zip'):
            path += '.agent.zip'
        zf = zipfile.ZipFile(path)
        with tempfile.TemporaryDirectory() as tempdir:
            zf.extractall(tempdir)
            # network state
            network_params = torch.load(os.path.join(tempdir, 'network.pth'))
            # agent config
            with open(os.path.join(tempdir, 'state.agent.json'), 'r') as file:
                params = json.load(file)

        nn_architecture = cls.get_type(params['nn_architecture'])
        del params['nn_architecture']
        rng_state = params.pop('random_state')
        agent = cls(nn_architecture=nn_architecture, **params)
        agent._rng.bit_generator.state = rng_state
        agent.network.load_state_dict(network_params)
        agent.network.eval()
        agent.__soft_update_target()
        return agent
