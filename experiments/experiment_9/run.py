"""This script needs to be run from its directory"""
import sys

sys.path.append('../..')

import logging
import json
import os
import argparse

from academia.curriculum import Curriculum
from academia.environments import DoorKey
from academia.agents import PPOAgent
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


def _run_full(run_no: int, agent_):
    # load configs
    config_path = os.path.join(CONFIGS_DIR, 'full.curriculum.yml')
    curriculum_full = Curriculum.load(config_path)
    # setup output dirs
    output_dir = os.path.join(OUTPUTS_DIR, f'full_{run_no}')
    curriculum_full.output_dir = output_dir
    # run
    _logger.info(f'Starting full run {run_no}.')
    curriculum_full.run(agent_, verbose=2)
    _logger.info(f'Completed full run {run_no}.')


def _run_skip(run_no: int, agent_):
    # load configs
    config_path = os.path.join(CONFIGS_DIR, 'skip.curriculum.yml')
    curriculum_skip = Curriculum.load(config_path)
    # setup output dirs
    output_dir = os.path.join(OUTPUTS_DIR, f'skip_{run_no}')
    curriculum_skip.output_dir = output_dir
    # run
    _logger.info(f'Starting skip run {run_no}.')
    curriculum_skip.run(agent_, verbose=2)
    _logger.info(f'Completed skip run {run_no}.')


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser()
    _argparser.add_argument('-r', '--runs', action='store', default=10,
                            help='Maximum runs')
    args = _argparser.parse_args()
    max_runs = int(args.runs)

    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)-19s] [%(levelname)-8s] %(name)s: %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='./outputs/experiment9.log',
    )
    _logger = logging.getLogger('experiments')

    meta = _get_meta()

    # full curriculum
    runs_done = meta.get('runs_done_full', 0)
    while runs_done < max_runs:
        agent = PPOAgent(
            n_actions=DoorKey.N_ACTIONS,
            actor_architecture=door_key.MLPStepActor,
            critic_architecture=door_key.MLPStepCritic,
            random_state=runs_done + 9000,
            n_episodes=10,
            n_epochs=10,
        )
        _run_full(runs_done + 1, agent)
        runs_done += 1
        _update_meta(meta, 'runs_done_full', runs_done)

    # curriculum no level 1
    runs_done = meta.get('runs_done_skip', 0)
    while runs_done < max_runs:
        agent = PPOAgent(
            n_actions=DoorKey.N_ACTIONS,
            actor_architecture=door_key.MLPStepActor,
            critic_architecture=door_key.MLPStepCritic,
            random_state=runs_done + 90000,
            n_episodes=10,
            n_epochs=10,
        )
        _run_skip(runs_done + 1, agent)
        runs_done += 1
        _update_meta(meta, 'runs_done_skip', runs_done)
