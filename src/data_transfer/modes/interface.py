from abc import ABC, abstractmethod

class ModeInterface(ABC):
    """
    Interface for all transmission modes.
    """

    @abstractmethod
    def send(self, data: bytes):
        """
        Send `data` via the transmission mode. 
        """
        pass

    @abstractmethod
    def receive(self):
        """
        Receive one packet of data and return raw bytes. 
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close any open sockets, subprocesses, or connections.
        """
        pass
