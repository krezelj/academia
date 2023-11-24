import unittest

import torch

from academia.utils.models import ms_pacman
from academia.environments import MsPacman


class TestMsPacmanModels(unittest.TestCase):

    def test_mlpdqn(self):
        sut = ms_pacman.MLPDQN()
        env = MsPacman(
            difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1, obs_type='ram',
        )
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match MsPacman actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match MsPacman state size')

    def test_mlpstepdqn(self):
        sut = ms_pacman.MLPStepDQN()
        env = MsPacman(
            difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1, obs_type='ram',
        )
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match MsPacman actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match MsPacman state size')

    def test_mlpactor(self):
        sut = ms_pacman.MLPActor()
        env = MsPacman(
            difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1, obs_type='ram',
        )
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match MsPacman actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match MsPacman state size')

    def test_mlpcritic(self):
        sut = ms_pacman.MLPCritic()
        env = MsPacman(
            difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1, obs_type='ram',
        )
        state = env.observe()
        try:
            sut(torch.tensor(state, dtype=torch.float))
        except RuntimeError:
            self.fail('Input layer neuron count does not match MsPacman state size')

    def test_mlpstepactor(self):
        sut = ms_pacman.MLPStepActor()
        env = MsPacman(
            difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1, obs_type='ram',
        )
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match MsPacman actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match MsPacman state size')

    def test_mlpstepcritic(self):
        sut = ms_pacman.MLPStepCritic()
        env = MsPacman(
            difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1, obs_type='ram',
        )
        state = env.observe()
        try:
            sut(torch.tensor(state, dtype=torch.float))
        except RuntimeError:
            self.fail('Input layer neuron count does not match MsPacman state size')


if __name__ == '__main__':
    unittest.main()
