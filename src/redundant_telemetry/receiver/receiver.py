import argparse, socket, time
from ..schema import data_pb2
from ..config import BAUD, END, MODEM_BAUD, MODEM_POWER_KEY, MODEM_SERIAL_PORT, START
from ..modes import get_mode

UDP_SOCKET_TIMEOUT_SECONDS = 0.5
UDP_IDLE_EXIT_SECONDS = 2.0

def process_packet(raw: bytes) -> None:
    """
    Decode and print a protobuf packet. 
    """
    if not raw:
        return
    # Packets may arrive already extracted (Minimodem) or as framed bytes (UDP).
    start_pos = raw.find(START)
    if start_pos == -1:
        packet = raw
    else:
        start = start_pos + 1
        end = raw.find(END, start)
        if end == -1:
            print("Malformed frame: START marker found without END marker.")
            return
        packet = raw[start:end]
    try: 
        msg = data_pb2.Sensors()
        msg.ParseFromString(packet)
        print(msg)
    except Exception as e:
        print("Failed to parse packet:", e)

def main() -> None:
    parser = argparse.ArgumentParser(description="Receive data via Minimodem or UDP")
    parser.add_argument(
        '--mode', 
        choices=['udp', 'modem'], 
        required=True, 
        help="Transmission mode: 'udp' or 'modem'"
    )
    args = parser.parse_args()

    mode = get_mode(args.mode, baud=BAUD, bind_socket=True)

    if args.mode == 'modem':
        from ..hardware.cellular_modem import CellularModem
        modem = CellularModem(
            power_key=MODEM_POWER_KEY,
            port=MODEM_SERIAL_PORT,
            baud=MODEM_BAUD,
        )
        try:
            modem.power_on()
            if modem.answer_call():
                print("Call answered. Packets received:")
                while True:
                    packet = mode.receive()
                    if not packet:
                        break
                    process_packet(packet)
                    time.sleep(0.05)
                modem.hangup()
        finally:
            modem.power_down()
            modem.close()
            mode.close()
    elif args.mode == 'udp':
        sock = getattr(mode, "sock", None)
        if sock is not None:
            # Periodic timeout lets us exit after sender stops.
            sock.settimeout(UDP_SOCKET_TIMEOUT_SECONDS)

        last_rx_time = time.monotonic()
        try:
            while True:
                try:
                    packet = mode.receive()
                except socket.timeout:
                    # Exit once we've been idle for long enough (e.g. if the dummy sender finished).
                    if time.monotonic() - last_rx_time >= UDP_IDLE_EXIT_SECONDS:
                        print("No UDP packets received recently; exiting receiver.")
                        break
                    continue

                if not packet:
                    continue

                last_rx_time = time.monotonic()
                process_packet(packet)
                time.sleep(0.001)
        finally:
            mode.close()

if __name__ == "__main__":
    main()
