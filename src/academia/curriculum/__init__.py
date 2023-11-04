"""
This module contains utilities for agents training. It basically controls interactions between agents
and environments.

Exported classes:

- :class:`LearningTask`
- :class:`Curriculum`
- :class:`LearningStats`

In this module logging is used, which is handled using built-in ``logging`` library. Besides standard logging
configuration, in certain methods such as :func:`LearningTask.run` and :func:`Curriculum.run` a user can
specify verbosity level which can be used to filter out some of the logs. These verbosity levels are common
throughout the entire module and they are as follows:

+-----------------+-------------------------------------------+
| Verbosity level | Description                               |
+=================+===========================================+
| 0               | no logging (except for errors)            |
+-----------------+-------------------------------------------+
| 1               | Task finished/Task interrupted + warnings |
+-----------------+-------------------------------------------+
| 2               | Mean evaluation reward at each iteration  |
+-----------------+-------------------------------------------+
| 3               | Each evaluation is logged                 |
+-----------------+-------------------------------------------+
| 4               | Each episode is logged                    |
+-----------------+-------------------------------------------+
"""
from .learning_task import LearningTask, LearningStats
from .curriculum import Curriculum

__all__ = [
    'LearningTask',
    'Curriculum',
    'LearningStats',
]
