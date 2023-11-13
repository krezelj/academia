import tempfile
from typing import Type
import unittest
from unittest import mock

import numpy as np
import torch

from academia.agents import PPOAgent
from academia.utils.models import ms_pacman

class MockModel(torch.nn.Module):

    def __init__(self):
        super(MockModel, self).__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(3, 1))
    
    def forward(self, state):
        return torch.Tensor([0.3, 0.1, 0.6])

class TestPPOAgent(unittest.TestCase):

    def __simulate_updates(self, agent: Type[PPOAgent], n_samples: int):
        # disable training during testing
        with mock.patch.object(PPOAgent.PPOBuffer, '_PPOBuffer__is_full', return_value=False):
            for i in range(n_samples):
                # force some terminal states and some non terminal states after that
                force_terminal = i == n_samples - 2
                force_non_terminal = i == n_samples - 1
                is_terminal = (force_terminal or np.random.random() > 0.9) and not force_non_terminal 
                agent.update(state=np.random.randint(0, 10, size=128),
                            action=np.random.randint(0, 3),
                            reward=np.random.random(),
                            new_state=np.random.randint(0, 10, size=128),
                            is_terminal=is_terminal)

    def __is_model_equal(self, expected: Type[torch.nn.Module], returned: Type[torch.nn.Module]):
        for expected_params, loaded_params in zip(expected.parameters(), returned.parameters()):
            if expected_params.data.ne(loaded_params.data).sum() > 0:
                return False
        return True

    def __is_buffer_equal(self, expected, returned):
        """
        Checks is two buffers are the same ignoring unfinished episodes.
        """
        # for expected_transition, loaded_transition in zip(expected, loaded):
        #     for expected_element, loaded_element in zip(expected_transition, loaded_transition):
        #         if type(expected_element) is np.ndarray:
        #             if not np.all(expected_element == loaded_element):
        #                 return False
        #         else:
        #             if expected_element != loaded_element:
        #                 return False
        return True

    def __assert_agents_equal(self, expected, returned):
        self.assertTrue(self.__is_model_equal(expected.actor, returned.actor))
        self.assertTrue(self.__is_model_equal(expected.critic, returned.critic))
        self.assertTrue(self.__is_buffer_equal(expected.buffer, returned.buffer))

        agent_cm = getattr(expected, '_PPOAgent__covariance_matrix')
        loaded_agent_cm = getattr(returned, '_PPOAgent__covariance_matrix')
        self.assertTrue(torch.all(agent_cm == loaded_agent_cm))

        ignored_attributes = [\
            'actor', 'critic', 'buffer', '_rng', 'actor_optimiser', 'critic_optimiser',\
            '_PPOAgent__covariance_matrix']
        for attribute_name in expected.__dict__.keys():
            if attribute_name not in ignored_attributes:
                self.assertEqual(
                    getattr(expected, attribute_name), 
                    getattr(returned, attribute_name),
                    msg=f"Attribute '{attribute_name}' not equal")

    def test_get_greedy_action(self):
        # arrange
        agent = PPOAgent(
            actor_architecture=MockModel,
            critic_architecture=MockModel,
            n_episodes=5,
            n_actions=3
        )
        # act
        action = agent.get_action(np.zeros(1), greedy=True)
        # assert
        expected_action = 2
        self.assertEqual(action, expected_action)

    def test_network_seeding(self):
        # arrange
        agent_1 = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_episodes=5,
            random_state=42
        )
        agent_2 = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_episodes=5,
            random_state=42
        )
        agent_3 = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_episodes=5,
            random_state=0
        )
        # assert
        self.assertTrue(self.__is_model_equal(agent_1.actor, agent_2.actor))
        self.assertTrue(self.__is_model_equal(agent_1.critic, agent_2.critic))

        self.assertFalse(self.__is_model_equal(agent_1.actor, agent_3.actor))
        self.assertFalse(self.__is_model_equal(agent_1.critic, agent_3.critic))

    def test_device_cuda(self):
        with mock.patch('torch.cuda.is_available', return_value=True):
            agent = PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9,
                n_episodes=5, 
                device='cuda')
            self.assertEqual(agent.device, torch.device('cuda'))

    def test_device_cpu(self):
        agent = PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9,
                n_episodes=5,
                device='cpu')
        self.assertEqual(agent.device, torch.device('cpu'))

    def test_device_cuda_not_available(self):
        with mock.patch('torch.cuda.is_available', return_value=False):
            agent = PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9,
                n_episodes=5,
                device='cuda')
            self.assertNotEqual(agent.device, torch.device('cuda'))

    def test_no_steps_no_episodes_raises_error(self):
        with self.assertRaises(ValueError):
            PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9)
            
    def test_file_path_suffixed(self):
        # arrange
        agent = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_steps=5,
        )
        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.zip', delete=False)
        returned_path = agent.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(returned_path, tmpfile.name)

    def test_file_path_unsuffixed(self):
        # arrange
        agent = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_steps=5,
        )
        # act 
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        returned_path = agent.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(returned_path, tmpfile.name + '.agent.zip')

    def test_saving_loading(self):
        # configuration 1
        # arrange
        agent = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            discrete=True,
            batch_size=128,
            n_epochs=5,
            n_steps=10,
            clip=0.1,
            gamma=0.8,
            epsilon=0.9,
            epsilon_decay=0.95,
            min_epsilon=0.03,
            random_state=0,
            device='cpu'
        )
        self.__simulate_updates(agent, 50)

        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.zip', delete=False)
        agent.save(tmpfile.name)
        loaded_agent = PPOAgent.load(tmpfile.name)
        tmpfile.close()

        # assert
        self.__assert_agents_equal(agent, loaded_agent)

        # configuration 2
        # arrange
        agent = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            discrete=False,
            batch_size=128,
            n_epochs=5,
            n_episodes=5,
            clip=0.1,
            gamma=0.8,
            epsilon=0.9,
            epsilon_decay=0.95,
            min_epsilon=0.03,
            random_state=0,
            device='cpu'
        )
        self.__simulate_updates(agent, 50)

        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.zip', delete=False)
        agent.save(tmpfile.name)
        loaded_agent = PPOAgent.load(tmpfile.name)
        tmpfile.close()

        # assert
        self.__assert_agents_equal(agent, loaded_agent)


class TestPPOBuffer(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()