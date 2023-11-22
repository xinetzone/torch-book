from abc import ABC, abstractmethod
from _collections_abc import _check_methods


class LoaderABC(ABC):
    @abstractmethod
    def init(self, mode):
        ...
        
    def train(self):
        return self.init(mode="train")

    def valid(self):
        return self.init(mode="valid")
    
    @classmethod
    def __subclasshook__(cls, C):
        if cls is LoaderABC:
            return _check_methods(C, "init")
        return NotImplemented