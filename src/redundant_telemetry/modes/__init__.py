from ..config import BAUD, UDP_HOST, UDP_PORT
from .interface import ModeInterface
from .modem_mode import ModemMode
from .udp_mode import UDPMode

def get_mode(
        mode_name: str,
        baud: int | None = None,
        bind_socket: bool = False,
        ip: str | None = None,
        port: int | None = None,
) -> ModeInterface:
    baud = BAUD if baud is None else baud
    ip = UDP_HOST if ip is None else ip
    port = UDP_PORT if port is None else port

    if mode_name == "modem":
        return ModemMode(baud=baud)
    elif mode_name == "udp":
        return UDPMode(ip=ip, port=port, bind_socket=bind_socket)
    else:
        raise ValueError(f"Unsupported mode: {mode_name}")
