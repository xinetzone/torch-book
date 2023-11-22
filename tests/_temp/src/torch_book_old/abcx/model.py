from abc import ABC, abstractmethod
from _collections_abc import _check_methods


class ModelABC(ABC):
    @abstractmethod
    def train_step(self, *args, **kwargs):
        ...

    @abstractmethod
    def valid_step(self, *args, **kwargs):
        ...

    @abstractmethod
    def configure(self, *args, **kwargs):
        ...

    @classmethod
    def __subclasshook__(cls, C):
        if cls is ModelABC:
            return _check_methods(C, "init")
        return NotImplemented