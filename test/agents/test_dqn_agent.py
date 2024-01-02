import tempfile
from typing import Type
import unittest
from unittest import mock

import numpy as np
import torch

from academia.agents import DQNAgent
from academia.utils.models import lava_crossing


class MockModel(torch.nn.Module):

    def __init__(self):
        super(MockModel, self).__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(3, 1))
    
    def forward(self, state):
        return torch.Tensor([1, -1, 2])


class TestDQNAgent(unittest.TestCase):

    def __simulate_updates(self, agent: Type[DQNAgent], n_samples: int):
        for i in range(n_samples):
            # force some terminal states and some non terminal states after that
            force_terminal = i == n_samples - 2
            force_non_terminal = i == n_samples - 1
            is_terminal = (force_terminal or np.random.random() > 0.9) and not force_non_terminal 
            agent.update(state=np.random.randint(0, 10, size=51),
                         action=np.random.randint(0, 3),
                         reward=np.random.random(),
                         new_state=np.random.randint(0, 10, size=51),
                         is_terminal=is_terminal)

    def __is_model_equal(self, expected: Type[torch.nn.Module], returned: Type[torch.nn.Module]):
        for expected_params, loaded_params in zip(expected.parameters(), returned.parameters()):
            if expected_params.data.ne(loaded_params.data).sum() > 0:
                return False
        return True

    def __is_memory_equal(self, expected, returned):
        for expected_transition, returned_transition in zip(expected, returned):
            for expected_element, returned_element in zip(expected_transition, returned_transition):
                if type(expected_element) is np.ndarray:
                    if not np.all(expected_element == returned_element):
                        return False
                else:
                    if expected_element != returned_element:
                        return False
        return True

    def __assert_agents_equal(self, expected: DQNAgent, returned: DQNAgent):
        self.assertTrue(self.__is_model_equal(expected.network, returned.network))
        self.assertTrue(self.__is_model_equal(expected.target_network, returned.target_network))
        self.assertTrue(self.__is_memory_equal(expected.memory, returned.memory))

        ignored_attributes = [\
            'network', 'target_network', 'memory',\
            '_rng', 'experience', 'optimizer']
        for attribute_name in expected.__dict__.keys():
            if attribute_name not in ignored_attributes:
                self.assertEqual(
                    getattr(expected, attribute_name), 
                    getattr(returned, attribute_name),
                    msg=f"Attribute '{attribute_name}' not equal")

    def test_memory(self):
        # arrange
        sut = DQNAgent(lava_crossing.MLPStepDQN, 3, replay_memory_size=10)
        memory_size = sut.replay_memory_size

        # act/assert
        self.assertEqual(0, len(sut.memory))

        self.__simulate_updates(sut, memory_size)
        self.assertEqual(memory_size, len(sut.memory))

        self.__simulate_updates(sut, 1)
        self.assertEqual(memory_size, len(sut.memory))

    def test_train_step(self):
        # arrange
        sut = DQNAgent(lava_crossing.MLPStepDQN, 3)
        update_every = sut.update_every

        # act/assert
        self.assertEqual(0, sut.train_step)

        self.__simulate_updates(sut, update_every - 1)
        self.assertEqual(update_every - 1, sut.train_step)

        self.__simulate_updates(sut, 1)
        self.assertEqual(0, sut.train_step)

    def test_get_greedy_action(self):
        # arrange
        sut = DQNAgent(
            nn_architecture=MockModel,
            n_actions=3
        )
        # act
        action = sut.get_action(np.zeros(1), greedy=True)
        # assert
        expected_action = 2
        self.assertEqual(expected_action, action)

    def test_legal_mask(self):
        # arrange
        sut = DQNAgent(
            nn_architecture=MockModel,
            n_actions=3
        )
        # act
        action = sut.get_action(np.zeros(1), legal_mask=np.array([0, 1, 0]), greedy=True)
        # assert
        expected_action = 1
        self.assertEqual(expected_action, action)

    def test_network_seeding(self):
        # arrange
        sut_1 = DQNAgent(
            nn_architecture=lava_crossing.MLPStepDQN,
            n_actions=3,
            random_state=42
        )
        sut_2 = DQNAgent(
            nn_architecture=lava_crossing.MLPStepDQN,
            n_actions=3,
            random_state=42
        )
        sut_3 = DQNAgent(
            nn_architecture=lava_crossing.MLPStepDQN,
            n_actions=3,
            random_state=0
        )
        # assert
        self.assertTrue(self.__is_model_equal(sut_1.network, sut_2.network))
        self.assertTrue(self.__is_model_equal(sut_1.target_network, sut_2.target_network))

        self.assertFalse(self.__is_model_equal(sut_1.network, sut_3.network))
        self.assertFalse(self.__is_model_equal(sut_1.target_network, sut_3.target_network))

    def test_device_cuda(self):
        with mock.patch('torch.cuda.is_available', return_value=True):
            def mock_build_network(self: DQNAgent):
                self.network = MockModel()
            with mock.patch.object(DQNAgent, '_DQNAgent__build_network', new=mock_build_network):
                sut = DQNAgent(
                    nn_architecture=lava_crossing.MLPStepDQN, 
                    n_actions=3, 
                    device='cuda')
                self.assertEqual(torch.device('cuda'), sut.device)

    def test_device_cpu(self):
        sut = DQNAgent(
            nn_architecture=lava_crossing.MLPStepDQN, 
            n_actions=3, 
            device='cpu')
        self.assertEqual(torch.device('cpu'), sut.device)

    def test_device_cuda_not_available(self):
        with mock.patch('torch.cuda.is_available', return_value=False):
            sut = DQNAgent(
                nn_architecture=lava_crossing.MLPStepDQN, 
                n_actions=3, 
                device='cuda')
            self.assertNotEqual(torch.device('cuda'), sut.device)       

    def test_file_path_suffixed(self):
        # arrange
        sut = DQNAgent(
            nn_architecture=lava_crossing.MLPStepDQN,
            n_actions=3
        )
        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.zip', delete=False)
        returned_path = sut.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(tmpfile.name, returned_path)

    def test_file_path_unsuffixed(self):
        # arrange
        sut = DQNAgent(
            nn_architecture=lava_crossing.MLPStepDQN,
            n_actions=3
        )
        # act 
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        returned_path = sut.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(tmpfile.name + '.agent.zip', returned_path)
        
    def test_saving_loading(self):
        # arrange
        agent = DQNAgent(
            lava_crossing.MLPStepDQN,
            n_actions=3,
            gamma=0.8,
            epsilon=0.9,
            epsilon_decay=0.95,
            min_epsilon=0.03,
            batch_size=128,
            random_state=0,
            device='cpu'
        )
        self.__simulate_updates(agent, 50)

        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.zip', delete=False)
        agent.save(tmpfile.name)
        sut = DQNAgent.load(tmpfile.name)
        tmpfile.close()

        # assert
        self.__assert_agents_equal(agent, sut)


if __name__ == '__main__':
    unittest.main()