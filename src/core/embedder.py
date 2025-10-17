from abc import ABC, abstractmethod

class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str):
        pass