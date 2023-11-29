import unittest

import torch

from academia.utils.models import lava_crossing
from academia.environments import LavaCrossing


class TestLavaCrossingModels(unittest.TestCase):

    def test_mlpdqn(self):
        sut = lava_crossing.MLPDQN()
        env = LavaCrossing(difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LavaCrossing actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LavaCrossing state size')

    def test_mlpstepdqn(self):
        sut = lava_crossing.MLPStepDQN()
        env = LavaCrossing(difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LavaCrossing actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LavaCrossing state size')

    def test_mlpactor(self):
        sut = lava_crossing.MLPActor()
        env = LavaCrossing(difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LavaCrossing actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LavaCrossing state size')

    def test_mlpcritic(self):
        sut = lava_crossing.MLPCritic()
        env = LavaCrossing(difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1)
        state = env.observe()
        try:
            sut(torch.tensor(state, dtype=torch.float))
        except RuntimeError:
            self.fail('Input layer neuron count does not match LavaCrossing state size')

    def test_mlpstepactor(self):
        sut = lava_crossing.MLPStepActor()
        env = LavaCrossing(difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LavaCrossing actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LavaCrossing state size')

    def test_mlpstepcritic(self):
        sut = lava_crossing.MLPStepCritic()
        env = LavaCrossing(difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1)
        state = env.observe()
        try:
            sut(torch.tensor(state, dtype=torch.float))
        except RuntimeError:
            self.fail('Input layer neuron count does not match LavaCrossing state size')


if __name__ == '__main__':
    unittest.main()
