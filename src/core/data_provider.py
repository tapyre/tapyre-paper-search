from abc import ABC, abstractmethod

class DataProvider(ABC):
    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def hasNext(self) -> bool:
        pass