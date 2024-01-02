import logging
import json
import os
import sys

# script should be run at the experiments/experiment_1 directory level
sys.path.append('..\\..')

from academia.curriculum import Curriculum, LearningTask
from academia.environments import LavaCrossing
from academia.agents import DQNAgent
from academia.utils.models.lava_crossing import MLPStepDQN

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='run.log',
)

is_empty = os.stat('meta.json').st_size == 0

if is_empty:
    params_rounds = [
        {"epsilon_reset_value": 1.0, "epsilon_decay": 0.999, "round": 0},
        {"epsilon_reset_value": 0.6, "epsilon_decay": 0.9994, "round": 0},
        {"epsilon_reset_value": 0.3, "epsilon_decay": 0.9995, "round": 0},
        {"epsilon_reset_value": 0.1, "epsilon_decay": 0.9996, "round": 0},
        {"epsilon_reset_value": 0.03, "epsilon_decay": 0.9997, "round": 0},
    ]
    with open('meta.json', 'w') as file:
        json.dump(params_rounds, file, indent=4)


def run_experiment(n_rounds):
    with open('meta.json', 'r') as meta_file:
        params_sets = json.load(meta_file)

    for i in range(n_rounds):
        selected_params, idx = next(
            ((params_set, index) for index, params_set in enumerate(params_sets) if params_set["round"] < 10),
            (None, None))
        if selected_params is None:
            logging.info("All params are exhausted.")
            return
        agent = DQNAgent(nn_architecture=MLPStepDQN,
                         n_actions=LavaCrossing.N_ACTIONS,
                         gamma=0.99,
                         epsilon=1.0,
                         epsilon_decay=selected_params['epsilon_decay'],
                         min_epsilon=0.01,
                         batch_size=128,
                         random_state=selected_params['round'] + int(selected_params['epsilon_decay'] * 10000))
        list_of_tasks = []
        for difficulty in range(3):
            list_of_tasks.append(LearningTask(env_type=LavaCrossing,
                                              env_args={'difficulty': difficulty,
                                                        'append_step_count': True,
                                                        'random_state': difficulty},
                                              stop_conditions={'min_evaluation_score': 0.8},
                                              evaluation_count=25,
                                              exploration_reset_value=selected_params['epsilon_reset_value']))

        curriculum = Curriculum(tasks=list_of_tasks,
                                output_dir=(
                                    f"outputs/eps={selected_params['epsilon_reset_value']}/"
                                    f"curriculum_iter={selected_params['round'] + 1}")
                                )
        if selected_params['round'] == 0:
            curriculum.save(f"configs/curriculum_eps={selected_params['epsilon_reset_value']}")
        curriculum.run(agent, verbose=2)

        params_sets[idx]['round'] += 1
        with open('meta.json', 'w') as meta_file:
            json.dump(params_sets, meta_file, indent=4)


run_experiment(1)
