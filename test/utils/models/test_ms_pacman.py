import unittest

import torch

from academia.utils.models import ms_pacman
from academia.environments import MsPacman


class TestMsPacmanModels(unittest.TestCase):

    def test_mlpactor(self):
        sut = ms_pacman.MLPActor()
        env = MsPacman(
            difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1, obs_type='ram',
        )
        state = env.observe()
        try:
            out = sut(torch.unsqueeze(torch.tensor(state, dtype=torch.float), dim=0))[0]
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


if __name__ == '__main__':
    unittest.main()
