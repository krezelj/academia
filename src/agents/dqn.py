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

    REPLAY_MEMORY_SIZE = 50000
    UPDATE_TARGET_FREQ = 10

    def __init__(self, nn_architecture: Type[nn.Module],
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

    def get_action(self, state, legal_mask=None, greedy=False):
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

    def replay(self):
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
