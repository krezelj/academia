import numpy as np
import torch.nn as nn
import torch
from .base import Agent
from typing import Literal
import torch.optim as optim
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    #zastanowic sie nad konwencja
    def __init__(self, state_size, n_actions, use_convolutions=False):
        super(DQNNetwork, self).__init__()
        self.use_convolutions = use_convolutions
        if not use_convolutions:
            self.network = nn.Sequential(
                nn.Linear(state_size, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, n_actions),
                #zastanowić się jak w przypadku przestrzeni akcji która również jest ciągła
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )
    def forward(self, x):
        #tutaj dla conv zmienic
        return self.network(x)

class DQN(Agent):
    def __init__(self, state_size: int, 
                 policy_type: Literal['CnnPolicy', 'MlpPolicy'],
                 n_actions: int, gamma: float =1., 
                 epsilon: float =1., epsilon_decay: float =0.99,
                 min_epsilon: float =0.01, learning_rate: float =0.01
                 ):
        self.state_size = state_size
        self.n_actions = n_actions
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.policy_type = policy_type
        self._build_network_()

    def _build_network_(self):
        if self.policy_type == "CnnPolicy":
            self.network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=True)
            self.target_network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=True)
        else:
            self.network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=False)
            self.target_network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=False)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1024:
            self.memory = self.memory[-1024:]

    def get_action(self, state, legal_mask=None, greedy=False):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_val_act = self.network(torch.Tensor(state))
            return torch.argmax(q_val_act).item()
        
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def update(self, state, action, reward: float, new_state, is_terminal: bool):
        #pomyslec nad tym
        return super().update(state, action, reward, new_state, is_terminal)
    
    def replay(self, batch_size):
        batch_indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_network(torch.Tensor(next_state)))
            target_value = self.network(torch.Tensor(state))
            target_value[action] = target
            states.append(state)
            targets.append(target_value)
        
        states = torch.Tensor(states)
        targets = torch.stack(targets)
        self.optimizer.zero_grad()
        q_values = self.network(states)
        loss = F.mse_loss(q_values, targets)
        loss.backward()
        self.optimizer.step()
        self._update_epsilon()
    
    def train_agent(self, env, num_episodes, batch_size, update_target_frequency = 100, learn_frequency = 10):
        step = 0
        n_updates = 0
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, __ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1
                if len(self.memory) > batch_size and step % learn_frequency == 0:
                    self.replay(batch_size)
                    n_updates += 1
                if n_updates % update_target_frequency == 0:
                    self.update_target()
            if episode % 50 == 0:
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}")



        print("Training finished.")
