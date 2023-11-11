"""
This module contains environments that agents can be trained on.

Most of these environments are wrappers for environments from `MiniGrid` or `gymnasium` packages
(:class:`BridgeBuilding` is the exception). The main purpose of these wrappers is to provide an easy and
uniform interface for scaling environments difficulty. Scalability of environments makes them suitable for
`Curriculum Learning`.

Exported classes:

- :class:`BridgeBuilding`
- :class:`LavaCrossing`
- :class:`DoorKey`
- :class:`LunarLander`
- :class:`MsPacman`
"""
from .bridge_building import BridgeBuilding
from .lava_crossing import LavaCrossing
from .door_key import DoorKey
from .lunar_lander import LunarLander
from .ms_pacman import MsPacman

__all__ = [
    'BridgeBuilding',
    'LavaCrossing',
    'DoorKey',
    'LunarLander',
    'MsPacman',
]