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
        if use_convolutions:
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

class DQN(Agent):
    def __init__(self, state_size: int, 
                 policy_type: Literal['CnnPolicy', 'MlpPolicy'],
                 n_actions: int, gamma: float =1., 
                 epsilon: float =1., epsilon_decay: float =0.99,
                 epsilon_min: float =0.01, learning_rate: float =0.01
                 ):
        self.state_size = state_size
        self.n_actions = n_actions
        self.memory = np.array([])
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.policy_type = policy_type
        self._build_network_()

    def _build_network_(self):
        if self.policy_type == "CnnPolicy":
            self.network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=True)
            self.target_network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=True)
            self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        else:
            self.network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=False)
            self.target_network = DQNNetwork(self.state_size, self.n_actions, use_convolutions=False)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

    def remember(self, state, action, reward, next_state, done):
        np.append(self.memory, (state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_val_act = self.network(torch.Tensor(state))
            return torch.argmax(q_val_act).item()
        
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def update():
        #zaimplementowac
        pass

    def replay(self, batch_size):
        batch_indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_network(torch.Tensor(next_state)))
            target_value = self.network(torch.Tensor(state))
            target_value[0][action] = target
            states.append(state)
            targets.append(target_value)
        
        states = torch.Tensor(states)
        targets = torch.stack(targets)
        self.optimizer.zero_grad()
        q_values = self.network(states)
        loss = F.mse_loss(q_values, targets)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        
    def train_agent(agent, env, episodes=1000, max_time_steps=500, batch_size=32):
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])
            for time_step in range(max_time_steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, agent.state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("Episode: {}/{}, Score: {}, Epsilon: {:.2f}".format(
                        episode, episodes, time_step, agent.epsilon))
                    break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                agent.update_target_model()