from abc import ABC, abstractmethod
from typing import Type
import importlib


class SavableLoadable(ABC):

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        pass

    @abstractmethod
    def save(self, path: str) -> str:
        """
        :return: an absolute path where the object was saved
        """
        pass

    @staticmethod
    def get_type(type_name_full: str) -> Type:
        """Returns a type based on its full name (e.g. academia.agents.Sarsa)"""
        module_name, _, qualname = type_name_full.rpartition('.')
        module = importlib.import_module(module_name)
        type_ = getattr(module, qualname)
        return type_

    @staticmethod
    def get_type_name_full(type_: Type) -> str:
        """Returns the full type name (e.g. academia.agents.Sarsa)"""
        module_name = type_.__module__
        qualname = type_.__qualname__
        return module_name + '.' + qualname
