import logging
import json
import os
import sys

# script should be run at the experiments/experiment_1 directory level
sys.path.append('..\\..')

from academia.curriculum import Curriculum, LearningTask
from academia.environments import DoorKey
from academia.agents import PPOAgent
from academia.utils.models import door_key


_logger = logging.getLogger('experiments')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='run.log',
)

is_empty = os.stat('meta.json').st_size == 0

if is_empty:
    params_rounds = [
        {"episodes_task1": 500, "type_of_impact": 'time', "round": 0},
        {"episodes_task1": 750, "type_of_impact": 'time', "round": 0},
        {"episodes_task1": 1000, "type_of_impact": 'time', "round": 0},
        {"episodes_task1": 1250, "type_of_impact": 'time', "round": 0},
        {"episodes_task1": 1500, "type_of_impact": 'time', "round": 0},
        {"episodes_task1": 500, "type_of_impact": 'eval', "round": 0},
        {"episodes_task1": 750, "type_of_impact": 'eval', "round": 0},
        {"episodes_task1": 1000, "type_of_impact": 'eval', "round": 0},
        {"episodes_task1": 1250, "type_of_impact": 'eval', "round": 0},
        {"episodes_task1": 1500, "type_of_impact": 'eval', "round": 0},
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
            _logger.info("All params are exhausted.")
            return

        agent = PPOAgent(
            n_actions=DoorKey.N_ACTIONS,
            actor_architecture=door_key.MLPStepActor,
            critic_architecture=door_key.MLPStepCritic,
            random_state=selected_params['round'] + idx * 10000,
            n_episodes=10,
            n_epochs=10,
        )
        if selected_params['type_of_impact'] == 'time':
            stop_condition = {'min_evaluation_score': 0.8}
            eval_count = 25
            eval_interval = 100
        else:
            stop_condition = {'max_episodes': 1500}
            eval_count = 100
            eval_interval = 1500

        task1 = LearningTask(env_type=DoorKey,
                             env_args={'difficulty': 1,
                                       'append_step_count': True,
                                       'random_state': selected_params['round']},
                             stop_conditions={'max_episodes': selected_params['episodes_task1']},
                             evaluation_count=25)

        task2 = LearningTask(env_type=DoorKey,
                             env_args={'difficulty': 2,
                                       'append_step_count': True,
                                       'random_state': selected_params['round']},
                             stop_conditions=stop_condition,
                             evaluation_count=eval_count,
                             evaluation_interval=eval_interval)

        curriculum = Curriculum(tasks=[task1, task2],
                                output_dir=(
                                    f"outputs/{selected_params['type_of_impact']}/episodes_{selected_params['episodes_task1']}"
                                    f"/curriculum_iter_{selected_params['round'] + 1}")
                                )

        curriculum.save(
            f"configs/{selected_params['type_of_impact']}/episodes_{selected_params['episodes_task1']}"
            f"/curriculum_iter_{selected_params['round'] + 1}")
        curriculum.run(agent, verbose=2)

        params_sets[idx]['round'] += 1
        with open('meta.json', 'w') as meta_file:
            json.dump(params_sets, meta_file, indent=4)


if __name__ == '__main__':
    run_experiment(1)
