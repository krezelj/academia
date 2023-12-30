from abc import ABC, abstractmethod
from typing import Type
import importlib
import os


class SavableLoadable(ABC):
    """
    An interface for classes which instances can be saved or loaded.
    """

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """
        Loads an object.

        Returns:
            A loaded object.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> str:
        """
        Saves an object.

        Returns:
             An absolute path where the object was saved.
        """
        pass

    @staticmethod
    def get_type(type_name_full: str) -> Type:
        """
        Args:
            type_name_full: type's import name (e.g. "academia.agents.SarsaAgent")

        Returns:
             A type based on its full name (e.g. ``academia.agents.SarsaAgent``).
        """
        module_name, _, qualname = type_name_full.rpartition('.')
        module = importlib.import_module(module_name)
        type_ = getattr(module, qualname)
        return type_

    @staticmethod
    def get_type_name_full(type_: Type) -> str:
        """
        Returns:
             A full type name (e.g. "academia.agents.SarsaAgent").
        """
        module_name = type_.__module__
        qualname = type_.__qualname__
        return module_name + '.' + qualname

    @staticmethod
    def _prep_save_file(specified_path: str, interrupted: bool) -> str:
        """
        Creates parent directories if they're missing and, if
        ``interrupted=True``, prepends 'backup_' to the
        file name in the specified path.

        Returns:
            Final path.
        """
        dirname, filename = os.path.split(specified_path)
        if interrupted:
            # prefix to let user know that the process has been interrupted
            filename = f'backup_{filename}'
        os.makedirs(dirname, exist_ok=True)
        full_path = os.path.join(dirname, filename)
        return full_path
