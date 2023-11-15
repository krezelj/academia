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

    def __is_buffer_equal(self, expected: PPOAgent.PPOBuffer, returned: PPOAgent.PPOBuffer):
        """
        Checks if two buffers are the same ignoring unfinished episodes.
        """

        # we can only check up to the last finished episode
        n_valid_steps = int(np.sum(expected.episode_lengths).item())

        self.assertEqual(expected.episode_counter, returned.episode_counter)
        self.assertEqual(len(expected.episode_lengths), len(returned.episode_lengths))
        self.assertEqual(n_valid_steps, returned.steps_counter)
        self.assertEqual(expected.n_steps, returned.n_steps)
        self.assertEqual(expected.n_episodes, returned.n_episodes)
        
        for expected_transitions, returned_transitions \
                in zip(expected.get_tensors('cpu'), returned.get_tensors('cpu')):
            for i, (expected_element, returned_element) \
                    in enumerate(zip(expected_transitions, returned_transitions)):
                if i >= n_valid_steps:
                    break
                if type(expected_element) is np.ndarray:
                    if not np.all(expected_element == returned_element):
                        return False
                if type(expected_element) is torch.Tensor:
                    if not torch.all(expected_element == returned_element):
                        return False
                else:
                    if expected_element != returned_element:
                        return False
        return True

    def __assert_agents_equal(self, expected: PPOAgent, returned: PPOAgent):
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
        sut = PPOAgent(
            actor_architecture=MockModel,
            critic_architecture=MockModel,
            n_episodes=5,
            n_actions=3
        )
        # act
        action = sut.get_action(np.zeros(1), greedy=True)
        # assert
        expected_action = 2
        self.assertEqual(expected_action, action)

    def test_network_seeding(self):
        # arrange
        sut_1 = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_episodes=5,
            random_state=42
        )
        sut_2 = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_episodes=5,
            random_state=42
        )
        sut_3 = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_episodes=5,
            random_state=0
        )
        # assert
        self.assertTrue(self.__is_model_equal(sut_1.actor, sut_2.actor))
        self.assertTrue(self.__is_model_equal(sut_1.critic, sut_2.critic))

        self.assertFalse(self.__is_model_equal(sut_1.actor, sut_3.actor))
        self.assertFalse(self.__is_model_equal(sut_1.critic, sut_3.critic))

    def test_device_cuda(self):
        with mock.patch('torch.cuda.is_available', return_value=True):
            sut = PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9,
                n_episodes=5, 
                device='cuda')
            self.assertEqual(torch.device('cuda'), sut.device)

    def test_device_cpu(self):
        sut = PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9,
                n_episodes=5,
                device='cpu')
        self.assertEqual(torch.device('cpu'), sut.device)

    def test_device_cuda_not_available(self):
        with mock.patch('torch.cuda.is_available', return_value=False):
            sut = PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9,
                n_episodes=5,
                device='cuda')
            self.assertNotEqual(torch.device('cuda'), sut.device)

    def test_no_steps_no_episodes_raises_error(self):
        with self.assertRaises(ValueError):
            PPOAgent(
                actor_architecture=ms_pacman.MLPActor,
                critic_architecture=ms_pacman.MLPCritic,
                n_actions=9)
            
    def test_file_path_suffixed(self):
        # arrange
        sut = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_steps=5,
        )
        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.agent.zip', delete=False)
        returned_path = sut.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(tmpfile.name, returned_path)

    def test_file_path_unsuffixed(self):
        # arrange
        sut = PPOAgent(
            actor_architecture=ms_pacman.MLPActor,
            critic_architecture=ms_pacman.MLPCritic,
            n_actions=9,
            n_steps=5,
        )
        # act 
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        returned_path = sut.save(tmpfile.name)
        tmpfile.close()

        # assert
        self.assertEqual(tmpfile.name + '.agent.zip', returned_path)

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
        sut = PPOAgent.load(tmpfile.name)
        tmpfile.close()

        # assert
        self.__assert_agents_equal(agent, sut)

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
        sut = PPOAgent.load(tmpfile.name)
        tmpfile.close()

        # assert
        self.__assert_agents_equal(agent, sut)

class TestPPOBuffer(unittest.TestCase):
    
    def __fill_buffer(self, buffer: PPOAgent.PPOBuffer, n_samples: int, terminal: bool=False) -> bool:
        """
        Fills the buffer with mock data and returns the value returned by the last update
        """
        return_flag = None
        for _ in range(n_samples):
            return_flag = buffer.update(
                np.random.randint(0, 10, size=128),
                np.random.randint(0, 3),
                np.random.random(),
                np.random.random(),
                is_terminal = terminal
            )
        return return_flag

    def test_episode_counter(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=1000)
        # act/assert
        self.assertEqual(0, sut.episode_counter)
        self.__fill_buffer(sut, 5, terminal=False)
        self.assertEqual(0, sut.episode_counter)
        self.__fill_buffer(sut, 1, terminal=True)
        self.assertEqual(1, sut.episode_counter)
        self.__fill_buffer(sut, 1, terminal=True)
        self.assertEqual(2, sut.episode_counter)
        self.__fill_buffer(sut, 1, terminal=False)
        self.assertEqual(2, sut.episode_counter)

    def test_episode_length_counter(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=1000)
        # act/assert
        self.assertEqual(0, sut.episode_length_counter)
        self.__fill_buffer(sut, 5, terminal=False)
        self.assertEqual(5, sut.episode_length_counter)
        self.__fill_buffer(sut, 1, terminal=True)
        self.assertEqual(0, sut.episode_length_counter)
        self.__fill_buffer(sut, 1, terminal=False)
        self.assertEqual(1, sut.episode_length_counter)

    def test_episode_lengths(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=1000)
        # act/assert
        self.assertEqual(0, len(sut.episode_lengths))
        self.__fill_buffer(sut, 5, terminal=False)
        self.assertEqual(0, len(sut.episode_lengths))
        self.__fill_buffer(sut, 1, terminal=True)
        self.assertEqual(1, len(sut.episode_lengths))
        self.assertEqual(6, sut.episode_lengths[-1])

        self.__fill_buffer(sut, 1, terminal=True)
        self.assertEqual(2, len(sut.episode_lengths))
        self.assertEqual(1, sut.episode_lengths[-1])

        self.__fill_buffer(sut, 5, terminal=False)
        self.assertEqual(2, len(sut.episode_lengths))

    def test_update(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=1000)

        # act/assert
        self.assertEqual(0, len(sut.states))
        self.assertEqual(0, len(sut.actions))
        self.assertEqual(0, len(sut.actions_logits))
        self.assertEqual(0, len(sut.rewards))
        self.assertEqual(0, sut.steps_counter)

        self.__fill_buffer(sut, 10)

        self.assertEqual(10, len(sut.states))
        self.assertEqual(10, len(sut.actions))
        self.assertEqual(10, len(sut.actions_logits))
        self.assertEqual(10, len(sut.rewards))
        self.assertEqual(10, sut.steps_counter)

    def test_update_return_when_using_n_episodes(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_episodes=3)

        # act/assert
        self.assertFalse(self.__fill_buffer(sut, 1, terminal=True))
        self.assertFalse(self.__fill_buffer(sut, 1, terminal=True))
        self.assertTrue(self.__fill_buffer(sut, 1, terminal=True))
    
    def test_update_return_when_using_n_steps(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=10)

        # act/assert
        self.assertFalse(self.__fill_buffer(sut, 9, terminal=False))
        self.assertTrue(self.__fill_buffer(sut, 1, terminal=True))

        self.assertFalse(self.__fill_buffer(sut, 10, terminal=False))
        self.assertFalse(self.__fill_buffer(sut, 1, terminal=False))
        self.assertTrue(self.__fill_buffer(sut, 1, terminal=True))

    def test_calculate_rewards_to_go(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=10)
        
        # act
        self.__fill_buffer(sut, 2, terminal=False)
        self.__fill_buffer(sut, 1, terminal=True)
        self.__fill_buffer(sut, 2, terminal=False)
        self.__fill_buffer(sut, 1, terminal=True)
        sut.calculate_rewards_to_go(gamma=0.8)

        # assert
        self.assertEqual(6, len(sut.rewards_to_go))

        # only test the oldest ones as they depend on other
        # rewards-to-go-as well so it implictly tests
        # them as well
        self.assertAlmostEqual(sut.rewards[2] * 0.8**2\
                        +sut.rewards[1] * 0.8\
                        +sut.rewards[0], 
                        sut.rewards_to_go[0], 5)
        self.assertAlmostEqual(sut.rewards[-1] * 0.8**2\
                        +sut.rewards[-2] * 0.8\
                        +sut.rewards[-3],
                        sut.rewards_to_go[3], 5)
        
    def test_reset(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=10)

        # act/assert
        self.assertEqual(0, len(sut.states))
        self.assertEqual(0, len(sut.actions))
        self.assertEqual(0, len(sut.actions_logits))
        self.assertEqual(0, len(sut.rewards))
        self.assertEqual(0, len(sut.episode_lengths))
        self.assertEqual(0, sut.steps_counter)
        self.assertEqual(0, sut.episode_counter)
        self.assertEqual(0, sut.episode_length_counter)

        self.__fill_buffer(sut, 10)
        sut.reset()

        self.assertEqual(0, len(sut.states))
        self.assertEqual(0, len(sut.actions))
        self.assertEqual(0, len(sut.actions_logits))
        self.assertEqual(0, len(sut.rewards))
        self.assertEqual(0, len(sut.episode_lengths))
        self.assertEqual(0, sut.steps_counter)
        self.assertEqual(0, sut.episode_counter)
        self.assertEqual(0, sut.episode_length_counter)

    def test_get_tensors(self):
        # arrange
        sut = PPOAgent.PPOBuffer(n_steps=1000)
        self.__fill_buffer(sut, 9)
        self.__fill_buffer(sut, 1, terminal=True)

        # act
        sut.calculate_rewards_to_go(gamma=0.8)
        states, actions, logits, rtgs = sut.get_tensors('cpu')

        # assert
        self.assertEqual(torch.Tensor, type(states))
        self.assertEqual((10, 128), states.shape)

        self.assertEqual(torch.Tensor, type(actions))
        self.assertEqual((10,), actions.shape)

        self.assertEqual(type(logits), torch.Tensor)
        self.assertEqual((10,), logits.shape)

        self.assertEqual(type(rtgs), torch.Tensor)
        self.assertEqual((10,), rtgs.shape)

if __name__ == '__main__':
    unittest.main()