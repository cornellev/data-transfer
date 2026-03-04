import socket
from .interface import ModeInterface

class UDPMode(ModeInterface):
    """
    Mode that uses UDP to send/receive data via Starlink. 
    """

    def __init__(self, ip="localhost", port=5005, bind_socket=False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, port)
        if bind_socket:
            self.sock.bind(self.addr)

    def send(self, data: bytes):
        self.sock.sendto(data, self.addr)

    def receive(self):
        data, _ = self.sock.recvfrom(4096)
        return data

    def close(self): 
        self.sock.close()