import sys
import os
import numpy as np

sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./src/experiments"))

from agents import QLAgent
from environments import BridgeBuilding
from curriculum import *

MIN_TARGET_REWARD = 75
MAX_EPISODES = 100_000
ENVIRONMENT_NAME = "BridgeBuilding"

task_1_params = TaskParameters(
    environment_name=ENVIRONMENT_NAME,
    environment_params={"n_boulders_placed": 3},
    max_episodes=MAX_EPISODES,
    greedy_evaluation_frequency=100,
    min_target_reward=MIN_TARGET_REWARD,
    episodes_to_early_stop=10
)
task_1 = Task(task_1_params)

task_2_params = TaskParameters(
    environment_name=ENVIRONMENT_NAME,
    environment_params={"n_boulders_placed": 2},
    max_episodes=MAX_EPISODES,
    greedy_evaluation_frequency=100,
    min_target_reward=MIN_TARGET_REWARD,
    episodes_to_early_stop=10
)
task_2 = Task(task_2_params)

task_3_params = TaskParameters(
    environment_name=ENVIRONMENT_NAME,
    environment_params={"n_boulders_placed": 1},
    max_episodes=MAX_EPISODES,
    greedy_evaluation_frequency=100,
    min_target_reward=MIN_TARGET_REWARD,
    episodes_to_early_stop=10
)
task_3 = Task(task_3_params)

task_final_params = TaskParameters(
    environment_name=ENVIRONMENT_NAME,
    environment_params={"n_boulders_placed": 0},
    max_episodes=MAX_EPISODES,
    greedy_evaluation_frequency=100,
    min_target_reward=MIN_TARGET_REWARD,
    episodes_to_early_stop=10
)
task_final = Task(task_final_params)

full_curriculum = Curriculum([task_1, task_2, task_3, task_final])
partial_curriculum = Curriculum([task_3, task_final])
no_curriculum = Curriculum([task_final])

ql_agent = QLAgent(8, alpha=0.2, gamma=0.95, epsilon_decay=0.99995)
full_curriculum.run_curriculum(ql_agent)

ql_agent = QLAgent(8, alpha=0.2, gamma=0.95, epsilon_decay=0.99995)
partial_curriculum.run_curriculum(ql_agent)

ql_agent = QLAgent(8, alpha=0.2, gamma=0.95, epsilon_decay=0.99995)
no_curriculum.run_curriculum(ql_agent)
