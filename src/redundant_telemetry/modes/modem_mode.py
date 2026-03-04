import subprocess, time
from ..config import START, END, BAUD
from .interface import ModeInterface

class ModemMode(ModeInterface):
    """
    Mode that uses minimodem to send/receive data via audio tones.
    """

    def __init__(self, baud=BAUD) -> None:
        self.baud = baud

    def send(self, data: bytes) -> None:
        proc = subprocess.Popen(
            ["minimodem", "--tx", str(self.baud)],
            stdin=subprocess.PIPE,
            text=False
        )
        try:
            seconds_in_message = (len(data)) * 8 / self.baud
            t_time = max(0, 1 - seconds_in_message)

            proc.stdin.write(data)
            proc.stdin.flush()
            time.sleep(t_time)

        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

    def receive(self) -> bytes | None:
        proc = subprocess.Popen(
            ["minimodem", "--rx", "-8", str(self.baud)],
            stdout=subprocess.PIPE
        )

        raw = b""
        while True:
            byte = proc.stdout.read(1)
            if not byte:
                break
            raw += byte

            if START in raw and END in raw:
                start = raw.find(START) + 1
                end = raw.find(END, start)
                packet = raw[start:end]
                raw = raw[end+1:]
                return packet

    def close(self) -> None:
        """
        No persistent resources to close. 
        """
        pass
