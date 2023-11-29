import unittest

import torch

from academia.utils.models import lunar_lander
from academia.environments import LunarLander


class TestLunarLanderModels(unittest.TestCase):

    def test_mlpdqn(self):
        sut = lunar_lander.MLPDQN()
        env = LunarLander(difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LunarLander actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LunarLander state size')

    def test_mlpstepdqn(self):
        sut = lunar_lander.MLPStepDQN()
        env = LunarLander(difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LunarLander actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LunarLander state size')

    def test_mlpactor(self):
        sut = lunar_lander.MLPActor()
        env = LunarLander(difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LunarLander actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LunarLander state size')

    def test_mlpcritic(self):
        sut = lunar_lander.MLPCritic()
        env = LunarLander(difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1)
        state = env.observe()
        try:
            sut(torch.tensor(state, dtype=torch.float))
        except RuntimeError:
            self.fail('Input layer neuron count does not match LunarLander state size')

    def test_mlpstepactor(self):
        sut = lunar_lander.MLPStepActor()
        env = LunarLander(difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match LunarLander actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match LunarLander state size')

    def test_mlpstepcritic(self):
        sut = lunar_lander.MLPStepCritic()
        env = LunarLander(difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1)
        state = env.observe()
        try:
            sut(torch.tensor(state, dtype=torch.float))
        except RuntimeError:
            self.fail('Input layer neuron count does not match LunarLander state size')


if __name__ == '__main__':
    unittest.main()
