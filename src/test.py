import logging
from academia.agents import PPOAgent
from academia.curriculum.learning_task import LearningTask
from academia.environments import LavaCrossing, LunarLander
from academia.utils.agent_debugger import AgentDebugger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    # filename='all.log',
)


class LCActor(nn.Module):

    def __init__(self):
        super(LCActor, self).__init__()

        self.layer1 = nn.Linear(50, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif obs.dtype != torch.float32:
            obs = obs.float()
 
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = self.layer3(activation2)
        output = self.softmax(activation3)

        return output
    
class LCCritic(nn.Module):

    def __init__(self):
        super(LCCritic, self).__init__()

        self.layer1 = nn.Linear(50, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif obs.dtype != torch.float32:
            obs = obs.float()
 
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

class LLActor(nn.Module):

    def __init__(self):
        super(LLActor, self).__init__()

        self.layer1 = nn.Linear(8, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif obs.dtype != torch.float32:
            obs = obs.float()
 
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = self.layer3(activation2)
        output = self.softmax(activation3)

        return output
    
class LLCritic(nn.Module):

    def __init__(self):
        super(LLCritic, self).__init__()

        self.layer1 = nn.Linear(8, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif obs.dtype != torch.float32:
            obs = obs.float()
 
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

# agent = PPOAgent(
#     discrete=True,actor_architecture=LCActor,critic_architecture=LCCritic,
#     batch_size=64, n_epochs=5, n_actions=3, n_episodes=10
# )
# task = LearningTask(LavaCrossing, {'difficulty': 0}, {'max_episodes': 1})
# task.run(agent, verbose=5)
# print("done learning")

# env = LavaCrossing(difficulty=0, render_mode='human')
# ad = AgentDebugger(agent, env, run=True, key_action_map={
#   'w': 2,
#   'a': 0,
#   'd': 1,  
# })

agent = PPOAgent(
    discrete=True,actor_architecture=LLActor,critic_architecture=LLCritic,
    batch_size=64, n_epochs=5, n_actions=4, n_episodes=10
)
env = LunarLander(difficulty=0, render_mode="human")
ad = AgentDebugger(agent, env, start_paused=True, run=True, key_action_map={
  's': 0, # nothing
  'a': 3, # left engine
  'w': 2, # main engine
  'd': 1, # right engine
})