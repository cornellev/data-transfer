from __future__ import annotations

import os

def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc

BAUD = _get_int("DATA_TRANSFER_BAUD", 256)
START = b"\x02"
END = b"\x03"

# UDP mode config 
UDP_HOST = os.getenv("DATA_TRANSFER_UDP_HOST", "localhost")
UDP_PORT = _get_int("DATA_TRANSFER_UDP_PORT", 5005)

# Modem mode config 
SENDER_NUMBER = os.getenv("DATA_TRANSFER_SENDER_NUMBER", "")
RECEIVER_NUMBER = os.getenv("DATA_TRANSFER_RECEIVER_NUMBER", "")
MODEM_SERIAL_PORT = os.getenv("DATA_TRANSFER_MODEM_PORT", "/dev/ttyS0")
MODEM_POWER_KEY = _get_int("DATA_TRANSFER_MODEM_POWER_KEY", 6)
MODEM_BAUD = _get_int("DATA_TRANSFER_MODEM_BAUD", 115200)