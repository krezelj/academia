"""
Base classes for all environments available in this package. All user-defined environments
should inherit from one of these classes (see :ref:`custom-envs` for more information on that)

Exported classes:

- :class:`ScalableEnvironment`
- :class:`GenericMiniGridWrapper`
- :class:`GenericGymnasiumWrapper`
- :class:`GenericAtariWrapper`
"""
from .scalable_env import ScalableEnvironment
from .generic_gymnasium_wrapper import GenericGymnasiumWrapper
from .generic_minigrid_wrapper import GenericMiniGridWrapper
from .generic_atari_wrapper import GenericAtariWrapper

__all__ = [
    'ScalableEnvironment',
    'GenericMiniGridWrapper',
    'GenericGymnasiumWrapper',
    'GenericAtariWrapper',
]
