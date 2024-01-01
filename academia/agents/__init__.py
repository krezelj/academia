"""
This module contains implementations of reinforcement learning algorithms, including tabular and those based
on neural networks.

Exported classes:

- :class:`QLAgent`
- :class:`SarsaAgent`
- :class:`DQNAgent`
- :class:`PPOAgent`

Note:
    :class:`DQNAgent` and :class:`PPOAgent` need to be provided network architectures when initializing.
    These network architectures should be subclasses of ``torch.nn.Module``. Example architectures
    can be found in :mod:`academia.utils.models`.
"""
from .ql_agent import QLAgent
from .sarsa_agent import SarsaAgent
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent

__all__ = [
    'QLAgent',
    'SarsaAgent',
    'DQNAgent',
    'PPOAgent'
]
