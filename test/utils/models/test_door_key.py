import unittest

import torch

from academia.utils.models import door_key
from academia.environments import DoorKey


class TestDoorKeyModels(unittest.TestCase):

    def test_mlpstepdqn(self):
        sut = door_key.MLPStepDQN()
        env = DoorKey(difficulty=0, n_frames_stacked=1, append_step_count=True, random_state=1)
        state = env.observe()
        try:
            out = sut(torch.tensor(state, dtype=torch.float))
            self.assertEqual(env.N_ACTIONS, len(out),
                             'Output layer neuron count does not match DoorKey actions count')
        except RuntimeError:
            self.fail('Input layer neuron count does not match DoorKey state size')


if __name__ == '__main__':
    unittest.main()
