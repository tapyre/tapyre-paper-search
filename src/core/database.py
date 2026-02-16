from abc import ABC, abstractmethod

class Database(ABC):
    def __init__(self):
        self.connect()

    @abstractmethod
    def connect(self):
        pass

    def __del__(self):
        self.disconnect()

    @abstractmethod
    def disconnect(self):
        pass