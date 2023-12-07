import argparse
import sys
import json
import os
import logging
from typing import Literal, Optional

import numpy as np

sys.path.append('../..')

from academia.environments import BridgeBuilding
from academia.agents import PPOAgent
from academia.curriculum import LearningTask, Curriculum
from academia.utils.models import bridge_building
from academia.utils import Stopwatch


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='experiment_8.log',
)

_logger = logging.getLogger('experiments')


def load_meta():
    if not os.path.exists('meta.json'):
        meta = {
            'n_runs': 10,
            'sparse': {
                'curr_runs': 0,
                'curr_steps_sum': 0,
                'nocurr_runs': 0
            },
            'dense': {
                'curr_runs': 0,
                'curr_steps_sum': 0,
                'nocurr_runs': 0
            }
        }
    else:
        with open('meta.json', 'r') as f:
            meta = json.load(f)
    return meta


def save_meta(meta: dict):
    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=4)


def get_min_evaluation(
        reward_density: Literal['sparse', 'dense'], 
        difficulty: int, 
        default: float):
    if reward_density == 'sparse':
        return default
    return default + 0.5 * difficulty


def get_task(
        reward_density: Literal['dense', 'sparse'],
        difficulty: int = 2,
        min_evaluation_score: int = 1.8,
        max_steps: int = np.inf,
        save_path: Optional[str] = None,
        random_state: Optional[int] = None,
        max_episodes: int = np.inf):
    task = LearningTask(
        BridgeBuilding,
        env_args={
            'difficulty': difficulty, 
            'random_state': random_state, 
            'append_step_count': False,
            'reward_density': reward_density,
            'append_step_count': True,
            'max_steps': 500},
        stop_conditions={
            'min_evaluation_score': get_min_evaluation(reward_density, difficulty, min_evaluation_score),
            'max_steps': max_steps,
            'max_episodes': max_episodes
            },
        greedy_evaluation=False,
        evaluation_count=25,
        stats_save_path=save_path,
        agent_save_path=save_path
    )
    return task


def get_curriculum(reward_density: Literal['sparse', 'dense'], output_dir: str, random_state: int):
    tasks = [
        get_task(reward_density=reward_density, difficulty=0, random_state=random_state),
        get_task(reward_density=reward_density, difficulty=1, random_state=random_state+1000),
        get_task(reward_density=reward_density, difficulty=2, random_state=random_state+2000)
    ]
    curriculum = Curriculum(
        tasks,
        output_dir=output_dir
    )
    return curriculum


def get_runnable(
        runnable_type: Literal['curr', 'nocurr'], 
        reward_density: Literal['sparse', 'dense'],
        i: int,
        max_steps: int,
        random_state: int,):
    if runnable_type == 'curr':
        return get_curriculum(
            reward_density=reward_density,
            output_dir=f'./outputs/{reward_density}/curriculum_{i}',
            random_state=random_state,
        )
    elif runnable_type == 'nocurr':
        return get_task(
            reward_density=reward_density,
            save_path=f'./outputs/{reward_density}/nocurriculum_{i}/nocurr',
            max_steps=max_steps,
            random_state=random_state,
        )
    raise ValueError(f"Invalid runnable type: {runnable_type}")


def get_agent(random_state: int):
    return PPOAgent(
        bridge_building.MLPStepActor,
        bridge_building.MLPStepCritic,
        n_actions=4,
        n_episodes=10,
        n_epochs=5,
        gamma=0.99,
        device='cpu',
        lr=0.0003,
        entropy_coefficient=0.01,
        random_state=random_state
    )


def determine_next_run(
        meta: dict, 
        reward_density: Optional[Literal['sparse', 'dense']] = None,
        allow_curr: bool = True,
        allow_nocurr: bool = True):
    n_runs = meta['n_runs']
    can_use_dense = reward_density is None or reward_density == 'dense'
    can_use_sparse = reward_density is None or reward_density == 'sparse'

    # dense curr
    if can_use_dense and \
            allow_curr and \
            meta['dense']['curr_runs'] < n_runs:
        return (
            'dense', 
            'curr', 
            meta['dense']['curr_runs'],
            meta['dense']['curr_runs'] + 0 * n_runs)
    
    # dense nocurr
    elif can_use_dense and \
            allow_nocurr and \
            meta['dense']['curr_runs'] == n_runs and \
            meta['dense']['nocurr_runs'] < n_runs:
        return (
            'dense', 
            'nocurr', 
            meta['dense']['nocurr_runs'],
            meta['dense']['nocurr_runs'] + 1 * n_runs)
    
    # sparse curr
    if can_use_sparse and \
            allow_curr and \
            meta['sparse']['curr_runs'] < n_runs:
        return (
            'sparse', 
            'curr', 
            meta['sparse']['curr_runs'],
            meta['sparse']['curr_runs'] + 2 * n_runs)
    
    # sparse nocurr
    elif can_use_sparse and \
            allow_nocurr and \
            meta['sparse']['curr_runs'] == n_runs and \
            meta['sparse']['nocurr_runs'] < n_runs:
        return (
            'sparse', 
            'nocurr', 
            meta['sparse']['nocurr_runs'],
            meta['sparse']['nocurr_runs'] + 3 * n_runs)
    return (None, None, None, None)


def run_experiment(
        n_runs: int = 1_000_000_000, 
        wall_time: float = 1_000_000_000,
        force_reward_density: Optional[Literal['sparse', 'dense']] = None,
        allow_curr: bool = True,
        allow_nocurr: bool = True,
        verbose: int = 2
        ):
    
    meta = load_meta()
    runs = 0
    sw = Stopwatch()
    while True:
        runs += 1
        if runs > n_runs:
            _logger.info("Reached runs limit. Stopping experiment.")
            break
        if sw.peek_time()[0] >= wall_time:
            _logger.info("Reached time limit. Stopping experiment.")
            break

        reward_density, runnable_type, i, random_state = determine_next_run(
            meta, force_reward_density, allow_curr, allow_nocurr)
        if reward_density is None:
            _logger.info("No allowed configurations left. Stopping experiment.")
            break

        agent = get_agent(random_state)

        if runnable_type == 'nocurr':
            max_steps = 2 * meta[reward_density]['curr_steps_sum'] / meta['n_runs']
        else:
            max_steps = np.inf

        runnable = get_runnable(runnable_type, reward_density, i, max_steps=max_steps, random_state=random_state)
        runnable.save(f'./configs/{reward_density}_{runnable_type}_{i}')

        try:
            _logger.info(f"Starting {runnable_type} for {reward_density} (run {i+1})")
            runnable.run(agent, verbose=verbose)
            _logger.info(f"Finished {runnable_type} for {reward_density} (run {i+1})")
        except:
            break

        meta[reward_density][f'{runnable_type}_runs'] += 1
        if runnable_type == 'curr':
            runnable: Curriculum
            steps_this_curriculum = 0
            for task_stats in runnable.stats.values():
                steps_this_curriculum += np.sum(task_stats.step_counts)
            meta[reward_density]['curr_steps_sum'] += steps_this_curriculum

    save_meta(meta)


def parse_options():
    _argparser = argparse.ArgumentParser()
    _argparser.add_argument('-v', '--verbose', action='store', default=2,
                            help='Verbosity')
    _argparser.add_argument('-t', '--time', action='store', default=1_000_000_000,
                            help='Maximum wall time')
    _argparser.add_argument('-r', '--runs', action='store', default=1_000_000_000,
                            help='Maximum runs')
    _argparser.add_argument('-d', '--density', action='store',
                            help='Reward density')
    _argparser.add_argument('-dc', '--discurriculum', action='store_false',
                            help='Disallow curriculum runs')
    _argparser.add_argument('-dn', '--disnocurriculum', action='store_false',
                            help='Disallow nocurriculum runs')
    args = _argparser.parse_args()
    return {
        'force_reward_density': args.density,
        'wall_time': float(args.time),
        'n_runs': int(args.runs),
        'allow_curr': args.discurriculum,
        'allow_nocurr': args.disnocurriculum,
        'verbose': int(args.verbose)
    }


def main():
    kwargs = parse_options()
    run_experiment(**kwargs)
    

if __name__ == '__main__':
    main()