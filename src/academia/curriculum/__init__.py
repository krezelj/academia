"""
This module contains utilities for agents training. It basically controls interactions between agents
and environments.

Exported classes:

- :class:`LearningTask`
- :class:`Curriculum`
"""
from .learning_task import LearningTask, LearningStats
from .curriculum import Curriculum

__all__ = [
    'LearningTask',
    'Curriculum',
    'LearningStats',
]
