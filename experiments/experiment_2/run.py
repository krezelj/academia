"""This script needs to be run from its directory"""
import sys

sys.path.append('../..')

import logging
import json
import os
import argparse

from academia.curriculum import LearningTask, Curriculum
from academia.environments import DoorKey
from academia.agents import DQNAgent, PPOAgent
from academia.utils.models import door_key


META_PATH = './meta.json'
CONFIGS_DIR = './configs'
OUTPUTS_DIR = './outputs'


def _get_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, 'r') as file:
        return json.load(file)


def _update_meta(meta_, key, value):
    meta_[key] = value
    with open(META_PATH, 'w') as file:
        json.dump(meta_, file, indent=2)


def _run_curr(run_no: int, agent_):
    # load configs
    curriculum = Curriculum.load(os.path.join(CONFIGS_DIR, 'curriculum.yml'))
    # setup output dirs
    agent_name = agent_.__class__.__qualname__
    output_dir = os.path.join(OUTPUTS_DIR, agent_name, f'curriculum_{run_no}')
    curriculum.output_dir = output_dir
    # run
    _logger.info(f'Starting curriculum run {run_no} for agent {agent_name}.')
    curriculum.run(agent_, verbose=4)
    _logger.info(f'Completed curriculum run {run_no} for agent {agent_name}.')


def _run_no_curr(run_no: int, agent_):
    # load configs
    nocurriculum = LearningTask.load(os.path.join(CONFIGS_DIR, 'nocurr.task.yml'))
    # setup output dirs
    agent_name = agent_.__class__.__qualname__
    output_dir = os.path.join(OUTPUTS_DIR, agent_name, f'nocurriculum_{run_no}')
    nocurriculum.agent_save_path = os.path.join(output_dir, 'agent')
    nocurriculum.stats_save_path = os.path.join(output_dir, 'stats')
    # run
    _logger.info(f'Starting nocurriculum run {run_no} for agent {agent_name}.')
    nocurriculum.run(agent_, verbose=4)
    _logger.info(f'Completed nocurriculum run {run_no} for agent {agent_name}.')


if __name__ == '__main__':
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='./outputs/experiment2.log',
    )
    _logger = logging.getLogger('experiments')

    _argparser = argparse.ArgumentParser()
    _argparser.add_argument('-c', '--curriculum', action='store_true',
                            help='Perform curriculum runs')
    _argparser.add_argument('-n', '--nocurriculum', action='store_true',
                            help='Perform nocurriculum runs')
    args = _argparser.parse_args()
    if not args.curriculum and not args.nocurriculum:
        _logger.warning(
            'No arguments specified, so no experiments will be run. If you wish to run everything, '
            'run the script with -nc. See --help for more details')
        sys.exit(0)

    meta = _get_meta()
    max_runs = 10

    # DQN curriculum
    if args.curriculum:
        print(1)
        runs_done = meta.get('runs_done_dqn_curr', 0)
        while runs_done < max_runs:
            agent = DQNAgent(
                n_actions=DoorKey.N_ACTIONS,
                nn_architecture=door_key.MLPStepDQN,
                random_state=runs_done,
            )
            _run_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_dqn_curr', runs_done)

    # DQN no_curriculum
    if args.nocurriculum:
        print(2)
        runs_done = meta.get('runs_done_dqn_nocurr', 0)
        while runs_done < max_runs:
            agent = DQNAgent(
                n_actions=DoorKey.N_ACTIONS,
                nn_architecture=door_key.MLPStepDQN,
                random_state=runs_done + max_runs,
            )
            _run_no_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_dqn_nocurr', runs_done)

    # PPO curriculum
    if args.curriculum:
        runs_done = meta.get('runs_done_ppo_curr', 0)
        while runs_done < max_runs:
            agent = PPOAgent(
                n_actions=DoorKey.N_ACTIONS,
                actor_architecture=door_key.MLPStepActor,
                critic_architecture=door_key.MLPStepCritic,
                random_state=runs_done + 2*max_runs,
                n_episodes=10,
            )
            _run_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_ppo_curr', runs_done)

    # PPO no_curriculum
    if args.nocurriculum:
        runs_done = meta.get('runs_done_ppo_nocurr', 0)
        while runs_done < max_runs:
            agent = PPOAgent(
                n_actions=DoorKey.N_ACTIONS,
                actor_architecture=door_key.MLPStepActor,
                critic_architecture=door_key.MLPStepCritic,
                random_state=runs_done + 3*max_runs,
                n_episodes=10,
            )
            _run_no_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_ppo_nocurr', runs_done)
