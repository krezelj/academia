import unittest

import torch

from academia.utils.models import bridge_building
from academia.environments import BridgeBuilding


class TestBridgeBuildingModels(unittest.TestCase):

    def test_mlpactor(self):
        sut = bridge_building.MLPActor()
        env = BridgeBuilding(
            difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1, obs_type='array',
        )
        state = env.observe()
        try:
            out = sut(torch.unsqueeze(torch.tensor(state, dtype=torch.float), dim=0))[0]
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match BridgeBuilding actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match BridgeBuilding state size')

    def test_mlpcritic(self):
        sut = bridge_building.MLPCritic()
        env = BridgeBuilding(
            difficulty=0, n_frames_stacked=1, append_step_count=False, random_state=1, obs_type='array',
        )
        state = env.observe()
        try:
            sut(torch.tensor(state, dtype=torch.float))
        except RuntimeError:
            self.fail('Input layer neuron count does not match BridgeBuilding state size')


if __name__ == '__main__':
    unittest.main()
