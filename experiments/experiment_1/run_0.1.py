from academia.curriculum import Curriculum, LearningTask, LearningStats
from academia.environments import LavaCrossing
from academia.agents import DQNAgent
from academia.utils.models.lava_crossing import MLPStepDQN

import logging


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='experiments/experiment_1/run_0.1.log',
)

N_RUNS = 10

task_lvl_0 = LearningTask(env_type=LavaCrossing, 
                          env_args={'difficulty': 0, 'append_step_count': True}, 
                          stop_conditions={'min_evaluation_score' : 0.8},
                          exploration_reset_value=0.1)

task_lvl_1 = LearningTask(env_type=LavaCrossing, 
                          env_args={'difficulty': 1, 'append_step_count': True}, 
                          stop_conditions={'min_evaluation_score' : 0.8},
                          exploration_reset_value=0.1)

task_lvl_2 = LearningTask(env_type=LavaCrossing, 
                          env_args={'difficulty': 2, 'append_step_count': True}, 
                          stop_conditions={'min_evaluation_score' : 0.8},
                          exploration_reset_value=0.1)

with open('experiments/experiment_1/meta.txt', 'r') as file:
    iter = sum(1 for line in file)

for i in range(N_RUNS - iter):
    agent = DQNAgent(model=MLPStepDQN, 
                     n_actions=LavaCrossing.N_ACTIONS,
                     gamma=0.99,
                     epsilon=1.0,
                     epsilon_decay=0.999,
                     epsilon_min=0.01,
                     batch_size=128,
                     random_state=123)
    
    curriculum = Curriculum(tasks=[task_lvl_0, task_lvl_1, task_lvl_2],
                            output_dir=f'experiments/experiment_1/outputs/eps=0.1/curriculum_iter={iter}')
    curriculum.save('experiments/experiment_1/configs/eps=0.1/curriculum_iter={iter}')
    
    with open('experiments/experiment_1/meta.txt', 'a') as f:
        f.write(f'Curriculum no. {iter} has been done.\n')

    iter += 1
