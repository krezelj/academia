"""
Miscellaneous classes and functions that don't belong anywhere else or are used by multiple modules.

The idea behind ``academia.utils`` is that other modules depend on it, but it itself does not depend on
any other module. That is different to :mod:`academia.tools` which works the other way around.

Exported classes:

- :class:`SavableLoadable`
- :class:`Stopwatch`
"""
from .saving_loading import SavableLoadable
from .stopwatch import Stopwatch


__all__ = [
    'SavableLoadable',
    'Stopwatch',
]
