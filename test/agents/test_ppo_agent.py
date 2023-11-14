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
        buffer = PPOAgent.PPOBuffer(n_steps=1000)
        # act/assert
        self.assertEqual(buffer.episode_counter, 0)
        self.__fill_buffer(buffer, 5, terminal=False)
        self.assertEqual(buffer.episode_counter, 0)
        self.__fill_buffer(buffer, 1, terminal=True)
        self.assertEqual(buffer.episode_counter, 1)
        self.__fill_buffer(buffer, 1, terminal=True)
        self.assertEqual(buffer.episode_counter, 2)
        self.__fill_buffer(buffer, 1, terminal=False)
        self.assertEqual(buffer.episode_counter, 2)

    def test_episode_length_counter(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_steps=1000)
        # act/assert
        self.assertEqual(buffer.episode_length_counter, 0)
        self.__fill_buffer(buffer, 5, terminal=False)
        self.assertEqual(buffer.episode_length_counter, 5)
        self.__fill_buffer(buffer, 1, terminal=True)
        self.assertEqual(buffer.episode_length_counter, 0)
        self.__fill_buffer(buffer, 1, terminal=False)
        self.assertEqual(buffer.episode_length_counter, 1)

    def test_episode_lengths(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_steps=1000)
        # act/assert
        self.assertEqual(len(buffer.episode_lengths), 0)
        self.__fill_buffer(buffer, 5, terminal=False)
        self.assertEqual(len(buffer.episode_lengths), 0)
        self.__fill_buffer(buffer, 1, terminal=True)
        self.assertEqual(len(buffer.episode_lengths), 1)
        self.assertEqual(buffer.episode_lengths[-1], 6)

        self.__fill_buffer(buffer, 1, terminal=True)
        self.assertEqual(len(buffer.episode_lengths), 2)
        self.assertEqual(buffer.episode_lengths[-1], 1)

        self.__fill_buffer(buffer, 5, terminal=False)
        self.assertEqual(len(buffer.episode_lengths), 2)

    def test_update(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_steps=1000)

        # act/assert
        self.assertEqual(len(buffer.states), 0)
        self.assertEqual(len(buffer.actions), 0)
        self.assertEqual(len(buffer.actions_logits), 0)
        self.assertEqual(len(buffer.rewards), 0)
        self.assertEqual(buffer.steps_counter, 0)

        self.__fill_buffer(buffer, 10)

        self.assertEqual(len(buffer.states), 10)
        self.assertEqual(len(buffer.actions), 10)
        self.assertEqual(len(buffer.actions_logits), 10)
        self.assertEqual(len(buffer.rewards), 10)
        self.assertEqual(buffer.steps_counter, 10)

    def test_update_return_when_using_n_episodes(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_episodes=3)

        # act/assert
        self.assertFalse(self.__fill_buffer(buffer, 1, terminal=True))
        self.assertFalse(self.__fill_buffer(buffer, 1, terminal=True))
        self.assertTrue(self.__fill_buffer(buffer, 1, terminal=True))
    
    def test_update_return_when_using_n_steps(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_steps=10)

        # act/assert
        self.assertFalse(self.__fill_buffer(buffer, 9, terminal=False))
        self.assertTrue(self.__fill_buffer(buffer, 1, terminal=True))

        self.assertFalse(self.__fill_buffer(buffer, 10, terminal=False))
        self.assertFalse(self.__fill_buffer(buffer, 1, terminal=False))
        self.assertTrue(self.__fill_buffer(buffer, 1, terminal=True))

    def test_calculate_rewards_to_go(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_steps=10)
        
        # act
        self.__fill_buffer(buffer, 2, terminal=False)
        self.__fill_buffer(buffer, 1, terminal=True)
        self.__fill_buffer(buffer, 2, terminal=False)
        self.__fill_buffer(buffer, 1, terminal=True)
        buffer.calculate_rewards_to_go(gamma=0.8)

        # assert
        self.assertEqual(len(buffer.rewards_to_go), 6)

        # only test the oldest ones as they depend on other
        # rewards-to-go-as well so it implictly tests
        # them as well
        self.assertAlmostEqual(buffer.rewards_to_go[0],
                         buffer.rewards[2] * 0.8**2\
                        +buffer.rewards[1] * 0.8\
                        +buffer.rewards[0], 5)
        self.assertAlmostEqual(buffer.rewards_to_go[3],
                         buffer.rewards[-1] * 0.8**2\
                        +buffer.rewards[-2] * 0.8\
                        +buffer.rewards[-3], 5)
        
    def test_reset(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_steps=10)

        # act/assert
        self.assertEqual(len(buffer.states), 0)
        self.assertEqual(len(buffer.actions), 0)
        self.assertEqual(len(buffer.actions_logits), 0)
        self.assertEqual(len(buffer.rewards), 0)
        self.assertEqual(len(buffer.episode_lengths), 0)
        self.assertEqual(buffer.steps_counter, 0)
        self.assertEqual(buffer.episode_counter, 0)
        self.assertEqual(buffer.episode_length_counter, 0)

        self.__fill_buffer(buffer, 10)
        buffer.reset()

        self.assertEqual(len(buffer.states), 0)
        self.assertEqual(len(buffer.actions), 0)
        self.assertEqual(len(buffer.actions_logits), 0)
        self.assertEqual(len(buffer.rewards), 0)
        self.assertEqual(len(buffer.episode_lengths), 0)
        self.assertEqual(buffer.steps_counter, 0)
        self.assertEqual(buffer.episode_counter, 0)
        self.assertEqual(buffer.episode_length_counter, 0)

    def test_get_tensors(self):
        # arrange
        buffer = PPOAgent.PPOBuffer(n_steps=1000)
        self.__fill_buffer(buffer, 9)
        self.__fill_buffer(buffer, 1, terminal=True)

        # act
        buffer.calculate_rewards_to_go(gamma=0.8)
        states, actions, logits, rtgs = buffer.get_tensors('cpu')

        # assert
        self.assertEqual(type(states), torch.Tensor)
        self.assertEqual(states.shape, (10, 128))

        self.assertEqual(type(actions), torch.Tensor)
        self.assertEqual(actions.shape, (10,))

        self.assertEqual(type(logits), torch.Tensor)
        self.assertEqual(logits.shape, (10,))

        self.assertEqual(type(rtgs), torch.Tensor)
        self.assertEqual(rtgs.shape, (10,))

if __name__ == '__main__':
    unittest.main()