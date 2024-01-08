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
    agent_name = agent_.__class__.__qualname__
    # load configs
    config_path = os.path.join(CONFIGS_DIR, agent_name, 'curriculum.yml')
    curriculum = Curriculum.load(config_path)
    # setup output dirs
    output_dir = os.path.join(OUTPUTS_DIR, agent_name, f'curriculum_{run_no}')
    curriculum.output_dir = output_dir
    # run
    _logger.info(f'Starting curriculum run {run_no} for agent {agent_name}.')
    curriculum.run(agent_, verbose=2)
    _logger.info(f'Completed curriculum run {run_no} for agent {agent_name}.')


def _run_no_curr(run_no: int, agent_):
    agent_name = agent_.__class__.__qualname__
    # load configs
    config_path = os.path.join(CONFIGS_DIR, agent_name, 'nocurr.task.yml')
    nocurriculum = LearningTask.load(config_path)
    # setup output dirs
    output_dir = os.path.join(OUTPUTS_DIR, agent_name, f'nocurriculum_{run_no}')
    nocurriculum.agent_save_path = os.path.join(output_dir, 'agent')
    nocurriculum.stats_save_path = os.path.join(output_dir, 'stats')
    # run
    _logger.info(f'Starting nocurriculum run {run_no} for agent {agent_name}.')
    nocurriculum.run(agent_, verbose=2)
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
    _argparser.add_argument('-d', '--dqn', action='store_true',
                            help='Perform DQN runs')
    _argparser.add_argument('-p', '--ppo', action='store_true',
                            help='Perform PPO runs')
    args = _argparser.parse_args()
    if not args.curriculum and not args.nocurriculum:
        _logger.warning(
            'Neither curriculum nor nocurriculum arguments have been specified, so no experiments will be '
            'run. If you wish to run everything, run the script with -cndp options. '
            'See --help for more details')
        sys.exit(0)
    if not args.dqn and not args.ppo:
        _logger.warning(
            'Neither dqn nor ppo arguments have been specified, so no experiments will be '
            'run. If you wish to run everything, run the script with -cndp options. '
            'See --help for more details')
        sys.exit(0)

    meta = _get_meta()
    # Max runs were split between algorithms. This is because more runs were needed for DQN due
    # to unconclusive results
    max_runs_dqn = 20
    max_runs_ppo = 10

    # DQN curriculum
    if args.curriculum and args.dqn:
        runs_done = meta.get('runs_done_dqn_curr', 0)
        while runs_done < max_runs_dqn:
            # Seed offset is needed because as of creating this script only 10 runs were assumed for DQN.
            # Since eventually we did 20 we need this to avoid seed overlapping between different runs
            seed_offset = 0
            if runs_done >= 10:
                seed_offset = 100

            agent = DQNAgent(
                n_actions=DoorKey.N_ACTIONS,
                nn_architecture=door_key.MLPStepDQN,
                batch_size=128,
                random_state=runs_done + seed_offset,
                epsilon_decay=0.9995,
            )
            _run_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_dqn_curr', runs_done)

    # DQN no_curriculum
    if args.nocurriculum and args.dqn:
        runs_done = meta.get('runs_done_dqn_nocurr', 0)
        while runs_done < max_runs_dqn:
            # Seed offset is needed because as of creating this script only 10 runs were assumed for DQN.
            # Since eventually we did 20 we need this to avoid seed overlapping between different runs
            seed_offset = 0
            if runs_done >= 10:
                seed_offset = 100

            agent = DQNAgent(
                n_actions=DoorKey.N_ACTIONS,
                nn_architecture=door_key.MLPStepDQN,
                batch_size=128,
                random_state=runs_done + 10 + seed_offset,
                epsilon_decay=0.9995,
            )
            _run_no_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_dqn_nocurr', runs_done)

    # PPO curriculum
    if args.curriculum and args.ppo:
        runs_done = meta.get('runs_done_ppo_curr', 0)
        while runs_done < max_runs_ppo:
            agent = PPOAgent(
                n_actions=DoorKey.N_ACTIONS,
                actor_architecture=door_key.MLPStepActor,
                critic_architecture=door_key.MLPStepCritic,
                random_state=runs_done + 2*max_runs_ppo,
                n_episodes=10,
                n_epochs=10,
            )
            _run_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_ppo_curr', runs_done)

    # PPO no_curriculum
    if args.nocurriculum and args.ppo:
        runs_done = meta.get('runs_done_ppo_nocurr', 0)
        while runs_done < max_runs_ppo:
            agent = PPOAgent(
                n_actions=DoorKey.N_ACTIONS,
                actor_architecture=door_key.MLPStepActor,
                critic_architecture=door_key.MLPStepCritic,
                random_state=runs_done + 3*max_runs_ppo,
                n_episodes=10,
                n_epochs=10,
            )
            _run_no_curr(runs_done + 1, agent)
            runs_done += 1
            _update_meta(meta, 'runs_done_ppo_nocurr', runs_done)
