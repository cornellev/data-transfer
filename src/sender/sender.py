import argparse, time
from config import (
    BAUD,
    END,
    MODEM_BAUD,
    MODEM_POWER_KEY,
    MODEM_SERIAL_PORT,
    RECEIVER_NUMBER,
    START,
)
from schema import data_pb2
from modes import get_mode
from uc26_sensor_reader.read_shm import SensorShmReader

def _build_message_from_dict(sensor_data: dict) -> data_pb2.Sensors:
    return data_pb2.Sensors(
        seq=sensor_data["seq"],
        global_ts=sensor_data["global_ts"],
        power=data_pb2.Power(
            ts=sensor_data["power"]["ts"],
            current=sensor_data["power"]["current"],
            voltage=sensor_data["power"]["voltage"],
        ),
        steering=data_pb2.Steering(
            ts=sensor_data["steering"]["ts"],
            brake_pressure=sensor_data["steering"]["brake_pressure"],
            turn_angle=sensor_data["steering"]["turn_angle"],
        ),
        rpm_front=data_pb2.RpmAxle(
            ts=sensor_data["rpm_front"]["ts"],
            rpm_left=sensor_data["rpm_front"]["rpm_left"],
            rpm_right=sensor_data["rpm_front"]["rpm_right"],
        ),
        rpm_back=data_pb2.RpmAxle(
            ts=sensor_data["rpm_back"]["ts"],
            rpm_left=sensor_data["rpm_back"]["rpm_left"],
            rpm_right=sensor_data["rpm_back"]["rpm_right"],
        ),
        gps=data_pb2.Gps(
            ts=sensor_data["gps"]["ts"],
            gps_lat=sensor_data["gps"]["gps_lat"],
            gps_long=sensor_data["gps"]["gps_long"],
        ),
        motor=data_pb2.Motor(
            ts=sensor_data["motor"]["ts"],
            rpm=sensor_data["motor"]["rpm"],
            throttle=sensor_data["motor"]["throttle"],
        ),
    )

def packet_from_sensor_dict(sensor_data: dict) -> bytes:
    """
    Serialize one sensor payload with START/END framing markers.
    """
    msg = _build_message_from_dict(sensor_data)
    return START + msg.SerializeToString() + END

def main() -> None:
    parser = argparse.ArgumentParser(description="Send data via Minimodem or UDP")
    parser.add_argument(
        '--mode', 
        choices=['udp', 'modem'], 
        required=True, 
        help="Transmission mode: 'udp' or 'modem'"
    )
    args = parser.parse_args()

    mode = get_mode(args.mode, baud=BAUD, bind_socket=False)

    shm_reader = SensorShmReader()
    if not shm_reader.available:
        raise RuntimeError(
            "Sensor SHM reader is unavailable. Ensure uc26_sensor_reader is initialized and producing data."
        )

    def send_loop() -> None:
        """
        Stream SHM packets.
        """
        while True:
            sensor_data = shm_reader.read_snapshot_dict()
            if sensor_data is None:
                # Sleep briefly when no SHM snapshot is ready.
                time.sleep(0.001)
                continue
            packet = packet_from_sensor_dict(sensor_data)
            mode.send(packet)
            print(f"SHM packet seq={sensor_data['seq']} sent.")
            time.sleep(0.001)
    
    if args.mode == 'modem':
        from hardware.cellular_modem import CellularModem
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
            if shm_reader is not None:
                shm_reader.close()
            modem.power_down()
            modem.close()
            mode.close()
    elif args.mode == 'udp':
        try:
            send_loop()
        finally:
            if shm_reader is not None:
                shm_reader.close()
            mode.close()

if __name__ == "__main__":
    main()
