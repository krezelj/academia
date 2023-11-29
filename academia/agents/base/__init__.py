"""
Base classes for all reinforcement learning algorithms available in this package. All user-defined algorithms
should inherit from one of these classes.

Exported classes:

- :class:`Agent`
- :class:`EpsilonGreedyAgent`
- :class:`TabularAgent`
"""
from .agent import Agent
from .epsilon_greedy_agent import EpsilonGreedyAgent
from .tabular_agent import TabularAgent

__all__ = [
    'Agent',
    'EpsilonGreedyAgent',
    'TabularAgent',
]
