import argparse
import sys
import json
import os
import logging
from typing import Literal, Optional

import numpy as np

sys.path.append('../..')

from academia.environments import MsPacman
from academia.agents import DQNAgent, PPOAgent
from academia.curriculum import LearningTask, Curriculum
from academia.utils.models import ms_pacman
from academia.utils import Stopwatch


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='experiment_7.log',
)

_logger = logging.getLogger('experiments')


def load_meta(run_offset):
    if not os.path.exists('meta.json'):
        max_runs_per_machine = 5
        meta = {
            'n_runs': max_runs_per_machine + run_offset,
            'ppo': {
                'curr_runs': run_offset,
                'nocurr_runs': run_offset
            },
            'dqn': {
                'curr_runs': run_offset,
                'nocurr_runs': run_offset
            }
        }
    else:
        with open('meta.json', 'r') as f:
            meta = json.load(f)
    return meta


def save_meta(meta: dict):
    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=4)


def get_task(
        difficulty: int = 3,
        greedy_evaluation: bool = True,
        save_path: Optional[str] = None,
        max_steps: int = 6_000_000,
        random_state: Optional[int] = None):
    task = LearningTask(
        MsPacman,
        env_args={
            'difficulty': difficulty, 
            'random_state': random_state, 
            'append_step_count': True, 
            'obs_type': 'ram'},
        stop_conditions={
            'max_steps': max_steps,
            },
        evaluation_count=25,
        greedy_evaluation=greedy_evaluation,
        exploration_reset_value=0.6,
        stats_save_path=save_path,
        agent_save_path=save_path
    )
    return task


def get_curriculum(greedy_evaluation: bool, output_dir: str, random_state: int):
    tasks = [
        get_task(0, max_steps=2_000_000, greedy_evaluation=greedy_evaluation, random_state=random_state),
        get_task(1, max_steps=2_000_000, greedy_evaluation=greedy_evaluation, random_state=random_state+1000),
        get_task(3, max_steps=2_000_000, greedy_evaluation=greedy_evaluation, random_state=random_state+2000),
    ]
    curriculum = Curriculum(
        tasks,
        output_dir=output_dir
    )
    return curriculum


def get_runnable(
        runnable_type: Literal['curr', 'nocurr'], 
        agent_type: Literal['ppo', 'dqn'],
        i: int,
        random_state: int,):
    if runnable_type == 'curr':
        return get_curriculum(
            greedy_evaluation=(agent_type=='dqn'),
            output_dir=f'./outputs/{agent_type}/curriculum_{i}',
            random_state=random_state,
        )
    elif runnable_type == 'nocurr':
        return get_task(
            save_path=f'./outputs/{agent_type}/nocurriculum_{i}/nocurr',
            random_state=random_state,
        )
    raise ValueError(f"Invalid runnable type: {runnable_type}")


def get_agent(agent_type: Literal['dqn', 'ppo'], random_state: int):
    if agent_type == 'dqn':
        return DQNAgent(
            ms_pacman.MLPStepDQN,
            n_actions=9,
            epsilon_decay=0.999,
            batch_size=128,
            min_epsilon=0.03, # let it explore a bit more
            device='cuda',
            random_state=random_state
        )
    if agent_type == 'ppo':
        return PPOAgent(
            actor_architecture=ms_pacman.MLPStepActor,
            critic_architecture=ms_pacman.MLPStepCritic,
            n_actions=9,
            n_episodes=10,
            n_epochs=5,
            gamma=0.99,
            lr=0.0001,
            entropy_coefficient=0.01,
            device='cuda',
            random_state=random_state
        )


def determine_next_run(
        meta: dict, 
        force_agent_type: Optional[Literal['dqn', 'ppo']] = None,
        allow_curr: bool = True,
        allow_nocurr: bool = True):
    n_runs = meta['n_runs']
    can_use_ppo = force_agent_type is None or force_agent_type == 'ppo'
    can_use_dqn = force_agent_type is None or force_agent_type == 'dqn'

    # ppo curr
    if can_use_ppo and \
            allow_curr and \
            meta['ppo']['curr_runs'] < n_runs:
        return (
            'ppo', 
            'curr', 
            meta['ppo']['curr_runs'],
            meta['ppo']['curr_runs'] + 0 * n_runs)
    
    # ppo nocurr
    elif can_use_ppo and \
            allow_nocurr and \
            meta['ppo']['nocurr_runs'] < n_runs:
        return (
            'ppo', 
            'nocurr', 
            meta['ppo']['nocurr_runs'],
            meta['ppo']['nocurr_runs'] + 1 * n_runs)
    
    # dqn curr
    if can_use_dqn and \
            allow_curr and \
            meta['dqn']['curr_runs'] < n_runs:
        return (
            'dqn', 
            'curr', 
            meta['dqn']['curr_runs'],
            meta['dqn']['curr_runs'] + 2 * n_runs)
    
    # dqn nocurr
    elif can_use_dqn and \
            allow_nocurr and \
            meta['dqn']['nocurr_runs'] < n_runs:
        return (
            'dqn', 
            'nocurr', 
            meta['dqn']['nocurr_runs'],
            meta['dqn']['nocurr_runs'] + 3 * n_runs)
    return (None, None, None, None)
    

def run_experiment(
        n_runs: int = 1_000_000_000, 
        wall_time: float = 1_000_000_000,
        force_agent_type: Optional[Literal['ppo', 'dqn']] = None,
        allow_curr: bool = True,
        allow_nocurr: bool = True,
        run_offset: int = 0,
        verbose: int = 2
        ):
    
    meta = load_meta(run_offset)
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

        agent_type, runnable_type, i, random_state = determine_next_run(
            meta, force_agent_type, allow_curr, allow_nocurr)
        if agent_type is None:
            _logger.info("No allowed configurations left. Stopping experiment.")
            break

        agent = get_agent(agent_type, random_state)
        runnable = get_runnable(runnable_type, agent_type, i, random_state=random_state)

        try:
            _logger.info(f"Starting {runnable_type} for {agent_type} (run {i+1})")
            runnable.run(agent, verbose=verbose)
            _logger.info(f"Finished {runnable_type} for {agent_type} (run {i+1})")
        except:
            break

        meta[agent_type][f'{runnable_type}_runs'] += 1

    save_meta(meta)


def parse_options():
    _argparser = argparse.ArgumentParser()
    _argparser.add_argument('-v', '--verbose', action='store', default=2,
                            help='Verbosity')
    _argparser.add_argument('-t', '--time', action='store', default=1_000_000_000,
                            help='Maximum wall time')
    _argparser.add_argument('-r', '--runs', action='store', default=1_000_000_000,
                            help='Maximum runs')
    _argparser.add_argument('-a', '--agent', action='store',
                            help='Agent type ("dqn"/"ppo")')
    _argparser.add_argument('-dc', '--discurriculum', action='store_false',
                            help='Disallow curriculum runs')
    _argparser.add_argument('-dn', '--disnocurriculum', action='store_false',
                            help='Disallow nocurriculum runs')
    _argparser.add_argument('-o', '--offset', action='store', required=True,
                            help='Run offset')
    args = _argparser.parse_args()
    return {
        'force_agent_type': args.agent,
        'wall_time': float(args.time),
        'n_runs': int(args.runs),
        'allow_curr': args.discurriculum,
        'allow_nocurr': args.disnocurriculum,
        'verbose': int(args.verbose),
        'run_offset': int(args.offset),
    }
        

def main():
    kwargs = parse_options()
    run_experiment(**kwargs)
    

if __name__ == '__main__':
    main()

