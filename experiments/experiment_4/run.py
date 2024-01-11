import logging
import json
import os
import sys

# script should be run at the experiments/experiment_4 directory level
sys.path.append('..\\..')

from academia.curriculum import Curriculum, LearningTask
from academia.environments import LavaCrossing
from academia.agents import DQNAgent
from academia.utils.models import lava_crossing


_logger = logging.getLogger('experiments')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='run.log',
)


def get_agent(selected_params: dict, random_state_offset: int):
    agent = DQNAgent(nn_architecture=lava_crossing.MLPStepDQN,
                        n_actions=LavaCrossing.N_ACTIONS,
                        gamma=0.99,
                        epsilon=1.0,
                        epsilon_decay=0.999,
                        min_epsilon=0.01,
                        batch_size=128,
                        random_state=selected_params['round'] + random_state_offset * 10000)
    return agent

def run_first_task(n_rounds):
    is_empty = os.stat('meta_task1.json').st_size == 0

    if is_empty:
        params_rounds = [
            {"episodes_task1": 1500, "round": 0},
            {"episodes_task1": 2500, "round": 0},
            {"episodes_task1": 3500, "round": 0},
            {"episodes_task1": 4500, "round": 0},
            {"episodes_task1": 5500, "round": 0},
        ]
        with open('meta_task1.json', 'w') as file:
            json.dump(params_rounds, file, indent=4)
        params_sets=params_rounds
    else:
        with open('meta_task1.json', 'r') as meta_file:
            params_sets = json.load(meta_file)
    
    for i in range(n_rounds):
        selected_params, idx = next(
            ((params_set, index) for index, params_set in enumerate(params_sets) if params_set["round"] < 10),
            (None, None))
        if selected_params is None:
            _logger.info("All params are exhausted.")
            return
        agent = get_agent(selected_params, idx)

        task1 = LearningTask(env_type=LavaCrossing,
                             env_args={'difficulty': 0,
                                       'append_step_count': True,
                                       'random_state': selected_params['round']},
                             stop_conditions={'max_episodes': selected_params['episodes_task1']},
                             evaluation_count=25,
                             agent_save_path=f"outputs/episodes_{selected_params['episodes_task1']}/agents/run_{selected_params['round']}/task1",
                             stats_save_path=f"outputs/episodes_{selected_params['episodes_task1']}/stats/run_{selected_params['round']}/task1",
                             exploration_reset_value=0.3)
        task1.run(agent, verbose=2)

        params_sets[idx]['round'] += 1
        with open('meta_task1.json', 'w') as meta_file:
            json.dump(params_sets, meta_file, indent=4)
    

def run_second_task(n_rounds):
    is_empty = os.stat('meta_task2.json').st_size == 0

    if is_empty:
        params_rounds = [
            {"episodes_task1": 1500, "type_of_impact": 'time', "round": 0},
            {"episodes_task1": 2500, "type_of_impact": 'time', "round": 0},
            {"episodes_task1": 3500, "type_of_impact": 'time', "round": 0},
            {"episodes_task1": 4500, "type_of_impact": 'time', "round": 0},
            {"episodes_task1": 5500, "type_of_impact": 'time', "round": 0},
            {"episodes_task1": 1500, "type_of_impact": 'eval', "round": 0},
            {"episodes_task1": 2500, "type_of_impact": 'eval', "round": 0},
            {"episodes_task1": 3500, "type_of_impact": 'eval', "round": 0},
            {"episodes_task1": 4500, "type_of_impact": 'eval', "round": 0},
            {"episodes_task1": 5500, "type_of_impact": 'eval', "round": 0},
        ]
        with open('meta_task2.json', 'w') as file:
            json.dump(params_rounds, file, indent=4)
        params_sets=params_rounds
    
    else:
        with open('meta_task2.json', 'r') as meta_file:
            params_sets = json.load(meta_file)
    
    for i in range(n_rounds):
        selected_params, idx = next(
            ((params_set, index) for index, params_set in enumerate(params_sets) if params_set["round"] < 10),
            (None, None))
        if selected_params is None:
            _logger.info("All params are exhausted.")
            return
        agent = DQNAgent.load(f"outputs/episodes_{selected_params['episodes_task1']}/agents/run_{selected_params['round']}/task1")
        agent.epsilon_decay = 0.9995

        if selected_params['type_of_impact'] == 'time':
            stop_condition = {'min_evaluation_score': 0.8}
            eval_count = 25
            eval_interval = 100
        else:
            max_episodes_task2 = 2000
            stop_condition = {'max_episodes': max_episodes_task2}
            eval_count = 100
            eval_interval = max_episodes_task2
        
        task2 = LearningTask(env_type=LavaCrossing,
                                env_args={'difficulty': 1,
                                        'append_step_count': True,
                                        'random_state': selected_params['round']},
                                stop_conditions=stop_condition,
                                evaluation_count=eval_count,
                                evaluation_interval=eval_interval,
                                agent_save_path=f"outputs/episodes_{selected_params['episodes_task1']}/agents/run_{selected_params['round']}/task2",
                                stats_save_path=f"outputs/episodes_{selected_params['episodes_task1']}/stats/run_{selected_params['round']}/task2",) 
        task2.run(agent, verbose=2)

        params_sets[idx]['round'] += 1
        with open('meta_task2.json', 'w') as meta_file:
            json.dump(params_sets, meta_file, indent=4)

if __name__ == '__main__':
    run_second_task(100)
