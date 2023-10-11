from .base import Agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Type

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class DQNAgent(Agent):
    """
    Class: DQNAgent

    This class represents a Deep Q-Network (DQN) agent used for reinforcement learning tasks.

    Attributes:
        - `REPLAY_MEMORY_SIZE`: Maximum size of the replay memory.
        - `UPDATE_TARGET_FREQ`: Frequency of updating the target network.
        - `nn_architecture (Type[nn.Module])`: Type of neural network architecture to be used.
        - `n_actions (int)`: Number of possible actions in the environment.
        - `gamma (float)`: Discount factor for future rewards.
        - `epsilon (float)`: Exploration-exploitation trade-off parameter.
        - `epsilon_decay (float)`: Decay factor for epsilon over time.
        - `min_epsilon (float)`: Minimum epsilon value to ensure exploration.
        - `batch_size (int)`: Size of the batch used for training the DQN.
        - `update_counter (int)`: Counter to keep track of when to update the target network.
        - `network (nn.Module)`: Main DQN neural network used for estimating Q-values.
        - `target_network (nn.Module)`: Target DQN neural network used for computing target Q-values.
        - `optimizer (torch.optim)`: Optimizer used for training the neural network.
        - `memory (deque)`: Replay memory to store experiences for training. From there batches are sampled.

    Methods:
        - `__init__(self, nn_architecture: Type[nn.Module], n_actions: int, gamma: float, epsilon: float, epsilon_decay: float, min_epsilon: float, batch_size: int)`: 
        Constructor method initializing the DQNAgent.
        - `__build_network_(self)`: Builds the neural network architectures for both the main and target networks.
        - `remember(self, state, action, reward, next_state, done)`: Stores an experience tuple in the replay memory.
        - `get_action(self, state, legal_mask=None, greedy=False) -> int`: Selects an action based on the current state using epsilon-greedy strategy.
        - `update_target(self)`: Updates the target network's weights with the main network's weights.
        - `update(self, state, action, reward: float, new_state, is_terminal: bool)`: Adds (state, action, reward, next_state, done) to replay memory. 
        Performs a single step of optimization for the DQN. Every UPDATE_TARGET_FREQ steps runs update_target() function
        - `replay(self) -> (torch.Tensor, torch.Tensor)`: Samples a batch from the replay memory and computes Q-value targets.
        - `save(self, save_path: str)`: Saves the model's state dictionary to the specified file path.
        - `load(self, model_path: str)`: Loads the model's state dictionary from the specified file path.

    Usage:
        ```python
        # Example Usage of DQNAgent
        from models import CartPoleMLP  # Import custom neural network architecture
        
        # Create an instance of the DQNAgent class with custom neural network architecture
        dqn_agent = DQNAgent(nn_architecture=CartPoleMLP, n_actions=2, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01, batch_size=64)
        
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
        - The `update_target` method updates the target network weights from the main network's weights.
        - It is recommended to adjust hyperparameters such as `gamma`, `epsilon`, `epsilon_decay`, and `batch_size` 
        based on the specific task and environment.

    """

    REPLAY_MEMORY_SIZE = 50000
    UPDATE_TARGET_FREQ = 10

    def __init__(self, nn_architecture: Type[nn.Module],
                 n_actions: int,
                 gamma: float =0.99, epsilon: float =1.,
                 epsilon_decay: float =0.99,
                 min_epsilon: float =0.01,
                 batch_size: int =64
                 ):
        super(DQNAgent, self).__init__(epsilon, min_epsilon, epsilon_decay)
        self.memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_counter = 0
        self.nn_architecture = nn_architecture
        self.n_actions = n_actions
        self.__build_network_()

    def __build_network_(self):
        self.network = self.nn_architecture()
        self.network.to(device)

        self.target_network = self.nn_architecture()
        self.target_network.to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.REPLAY_MEMORY_SIZE:
            self.memory = self.memory[1:]

    def get_action(self, state, legal_mask=None, greedy=False) -> int:
        q_val_act = self.network(torch.Tensor(state)).to(device)
        if legal_mask is not None:
            q_val_act = (q_val_act - torch.min(q_val_act)) * torch.Tensor(legal_mask) + torch.Tensor(legal_mask)
        if np.random.uniform() > self.epsilon or greedy:
            return torch.argmax(q_val_act).item()
        elif legal_mask is not None:
            return np.random.choice(np.arange(0, self.n_actions), size=1, p=legal_mask/legal_mask.sum())[0]
        else:
            return np.random.randint(0, self.n_actions)
        
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def update(self, state, action, reward: float, new_state, is_terminal: bool):
        self.remember(state=state, action=action, reward=reward, next_state=new_state, done=is_terminal)
        if len(self.memory) >= self.batch_size:
            states, targets = self.replay()
            self.optimizer.zero_grad()
            q_values = self.network(states).to(device)
            loss = F.mse_loss(q_values, targets)
            loss.backward()
            self.optimizer.step()
        if is_terminal:
            self.update_counter += 1
        if self.update_counter > self.UPDATE_TARGET_FREQ:
            self.update_target()
            self.update_counter = 0

    def replay(self) -> (torch.Tensor, torch.Tensor):
        batch_indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_network(torch.Tensor(next_state)))
            target_value = self.network(torch.Tensor(state)).to(device)
            target_value[action] = target
            states.append(state)
            targets.append(target_value)
        #To allow faster transformation to tensor is preffered numpy array
        states = torch.Tensor(np.array(states)).to(device)
        targets = torch.stack(targets).to(device)
        return states, targets
    
    def save(self, save_path):
        torch.save(self.network.state_dict(), save_path)

    def load(self, model_path):
        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()
