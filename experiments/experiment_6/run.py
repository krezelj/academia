import argparse
import sys
import json
import os
import logging
from typing import Literal, Optional

import numpy as np

sys.path.append('../..')

from academia.environments import LunarLander
from academia.agents import DQNAgent, PPOAgent
from academia.curriculum import LearningTask, Curriculum
from academia.utils.models import lunar_lander
from academia.utils import Stopwatch

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='experiment_6.log',
)

_logger = logging.getLogger('experiments')


def load_meta():
    if not os.path.exists('meta.json'):
        meta = {
            'n_runs': 10,
            'ppo': {
                'curr_1_runs': 10,
                'curr_2_runs': 0,
            },
            'dqn': {
                'curr_1_runs': 10,
                'curr_2_runs': 0,
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
        difficulty: int = 4,
        min_evaluation_score: int = 200,
        max_steps: int = np.inf,
        greedy_evaluation: bool = True,
        save_path: Optional[str] = None,
        random_state: Optional[int] = None,
        max_episodes: int = np.inf):
    task = LearningTask(
        LunarLander,
        env_args={'difficulty': difficulty, 'random_state': random_state, 'append_step_count': False},
        stop_conditions={
            'min_evaluation_score': min_evaluation_score,
            'max_steps': max_steps,
            'max_episodes': max_episodes
        },
        evaluation_count=50,
        greedy_evaluation=greedy_evaluation,
        exploration_reset_value=0.3,
        stats_save_path=save_path,
        agent_save_path=save_path
    )
    return task


def get_curriculum_1(greedy_evaluation: bool, output_dir: str, random_state: int):
    tasks = [
        get_task(0, 200, greedy_evaluation=greedy_evaluation, random_state=random_state),
        get_task(4, 200, greedy_evaluation=greedy_evaluation, random_state=random_state + 4000)
    ]
    curriculum = Curriculum(
        tasks,
        output_dir=output_dir
    )
    return curriculum


def get_curriculum_2(greedy_evaluation: bool, output_dir: str, random_state: int):
    tasks = [
        get_task(0, 200, greedy_evaluation=greedy_evaluation, random_state=random_state),
        get_task(3, 200, greedy_evaluation=greedy_evaluation, random_state=random_state + 3000),
        get_task(4, 200, greedy_evaluation=greedy_evaluation, random_state=random_state + 4000)
    ]
    curriculum = Curriculum(
        tasks,
        output_dir=output_dir
    )
    return curriculum


def get_runnable(
        runnable_type: Literal['curr_1', 'curr_2'],
        agent_type: Literal['ppo', 'dqn'],
        i: int,
        random_state: int, ):
    if runnable_type == 'curr_1':
        return get_curriculum_1(
            greedy_evaluation=True,
            output_dir=f'./outputs/{agent_type}/{runnable_type}/curriculum_{i}',
            random_state=random_state,
        )
    if runnable_type == 'curr_2':
        return get_curriculum_2(
            greedy_evaluation=True,
            output_dir=f'./outputs/{agent_type}/{runnable_type}/curriculum_{i}',
            random_state=random_state,
        )
    raise ValueError(f"Invalid runnable type: {runnable_type}")


def get_agent(agent_type: Literal['dqn', 'ppo'], random_state: int):
    if agent_type == 'dqn':
        return DQNAgent(
            lunar_lander.MLPDQN,
            n_actions=4,
            epsilon_decay=0.9995,
            device='cpu',
            random_state=random_state
        )
    if agent_type == 'ppo':
        return PPOAgent(
            lunar_lander.MLPActor,
            lunar_lander.MLPCritic,
            n_actions=4,
            n_episodes=10,
            n_epochs=4,
            gamma=0.99,
            device='cpu',
            lr=0.0003,
            entropy_coefficient=0,
            random_state=random_state
        )


def determine_next_run(
        meta: dict,
        force_agent_type: Optional[Literal['dqn', 'ppo']] = None):
    n_runs = meta['n_runs']
    can_use_ppo = force_agent_type is None or force_agent_type == 'ppo'
    can_use_dqn = force_agent_type is None or force_agent_type == 'dqn'

    # dqn curr
    if can_use_dqn and \
            meta['dqn']['curr_1_runs'] < n_runs:
        return (
            'dqn',
            'curr_1',
            meta['dqn']['curr_1_runs'],
            meta['dqn']['curr_1_runs'] + 0 * n_runs)
    if can_use_dqn and \
            meta['dqn']['curr_2_runs'] < n_runs:
        return (
            'dqn',
            'curr_2',
            meta['dqn']['curr_2_runs'],
            meta['dqn']['curr_2_runs'] + 1 * n_runs) 

    # ppo curr
    if can_use_ppo and \
            meta['ppo']['curr_1_runs'] < n_runs:
        return (
            'ppo',
            'curr_1',
            meta['ppo']['curr_1_runs'],
            meta['ppo']['curr_1_runs'] + 2 * n_runs) # TODO change offset (x * n_runs)
    if can_use_ppo and \
            meta['ppo']['curr_2_runs'] < n_runs:
        return (
            'ppo',
            'curr_1',
            meta['ppo']['curr_2_runs'],
            meta['ppo']['curr_2_runs'] + 3 * n_runs) # TODO change offset (x * n_runs)

    return (None, None, None, None)


def run_experiment(
        n_runs: int = 1_000_000_000,
        wall_time: float = 1_000_000_000,
        force_agent_type: Optional[Literal['ppo', 'dqn']] = None,
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

        agent_type, runnable_type, i, random_state = determine_next_run(meta, force_agent_type)
        if agent_type is None:
            _logger.info("No allowed configurations left. Stopping experiment.")
            break

        agent = get_agent(agent_type, random_state)
        runnable = get_runnable(runnable_type, agent_type, i, random_state=random_state)

        try:
            _logger.info(f"Starting {runnable_type} for {agent_type} (run {i + 1})")
            runnable.run(agent, verbose=verbose)
            _logger.info(f"Finished {runnable_type} for {agent_type} (run {i + 1})")
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
    args = _argparser.parse_args()
    return {
        'force_agent_type': args.agent,
        'wall_time': float(args.time),
        'n_runs': int(args.runs),
        'verbose': int(args.verbose)
    }


def main():
    kwargs = parse_options()
    run_experiment(**kwargs)


if __name__ == '__main__':
    main()
