from abc import ABC, abstractmethod

class Sketcher(ABC):
    @abstractmethod
    def compress(self, updates):
        pass
    @abstractmethod
    def uncompress(self, compressed_updates):
        pass

class VoidSketcher(Sketcher):
    def compress(self, updates):
        return updates
    def uncompress(self, compressed_updates):
        return compressed_updates
