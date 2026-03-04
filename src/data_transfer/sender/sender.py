import argparse, time
from ..config import (
    BAUD,
    END,
    MODEM_BAUD,
    MODEM_POWER_KEY,
    MODEM_SERIAL_PORT,
    RECEIVER_NUMBER,
    START,
)
from ..schema import data_pb2
from ..modes import get_mode

def dummy_packet(count: int):
    """
    Return one dummy protobuf packet as bytes with START/END markers. 
    """
    msg = data_pb2.Sensors(
                voltage=1.0 + count,
                draw=2.0 + count,
                gps_lat=3.0 + count,
                gps_long=4.0 + count,
                velocity=5.0 + count,
                throttle=6.0 + count,
            )
    return START + msg.SerializeToString() + END

def main():
    parser = argparse.ArgumentParser(description="Send data via UDP or Modem")
    parser.add_argument(
        '--mode', 
        choices=['udp', 'modem'], 
        required=True, 
        help="Transmission mode: 'udp' or 'modem'"
    )
    args = parser.parse_args()

    mode = get_mode(args.mode, baud=BAUD, bind_socket=False)
    n = 10 # number of dummy protobuf packets to send 

    def send_loop():
        """
        Send each packet as they are being generated.
        """
        for count in range(n):
            packet = dummy_packet(count)
            mode.send(packet)
            print(f'Packet #{count} was sent.')
            time.sleep(0.001)
    
    if args.mode == 'modem':
        from ..hardware.cellular_modem import CellularModem
        modem = CellularModem(
            power_key=MODEM_POWER_KEY,
            port=MODEM_SERIAL_PORT,
            baud=MODEM_BAUD,
        )
        try:
            modem.power_on()
            if modem.dial(RECEIVER_NUMBER):
                print("Call connected. Sending packets:")
                send_loop()
                modem.hangup()
        finally:
            modem.power_down()
            modem.close()
            mode.close()
    elif args.mode == 'udp':
        send_loop()
        mode.close()   

if __name__ == "__main__":
    main()






