import argparse, time
import importlib.util
from pathlib import Path
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

# If the SHM reader is unavailable, send this many dummy packets before exiting.
MAX_DUMMY_PACKETS = 10

REPO_ROOT = Path(__file__).resolve().parents[3]
SHM_READER_FILE = REPO_ROOT / "lib" / "uc26_sensor_reader" / "read_shm.py"

try:
    if not SHM_READER_FILE.exists():
        raise FileNotFoundError(f"{SHM_READER_FILE} not found")

    shm_spec = importlib.util.spec_from_file_location("uc26_sensor_reader.read_shm", SHM_READER_FILE)
    if shm_spec is None or shm_spec.loader is None:
        raise ImportError(f"Could not load module spec from {SHM_READER_FILE}")

    shm_module = importlib.util.module_from_spec(shm_spec)
    shm_spec.loader.exec_module(shm_module)
    SensorShmReader = shm_module.SensorShmReader
    SHM_IMPORT_ERROR = None
except Exception as e:
    # If the submodule reader is unavailable, fall back to dummy data.
    SensorShmReader = None
    SHM_IMPORT_ERROR = e

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

def _dummy_sensor_dict(count: int) -> dict:
    base = float(count)
    return {
        "seq": count,
        "global_ts": int(time.time_ns()),
        "power": {"ts": count, "current": 2.0 + base, "voltage": 1.0 + base},
        "steering": {"ts": count, "brake_pressure": 3.0 + base, "turn_angle": 4.0 + base},
        "rpm_front": {"ts": count, "rpm_left": 5.0 + base, "rpm_right": 6.0 + base},
        "rpm_back": {"ts": count, "rpm_left": 7.0 + base, "rpm_right": 8.0 + base},
        "gps": {"ts": count, "gps_lat": 42.444 + (0.0001 * base), "gps_long": -76.501 + (0.0001 * base)},
        "motor": {"ts": count, "rpm": 9.0 + base, "throttle": 10.0 + base},
    }


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

    shm_reader = None
    use_dummy = True
    if SensorShmReader is None:
        print(f"SHM reader import failed: ({SHM_IMPORT_ERROR}); using dummy data instead.")
    else:
        try:
            shm_reader = SensorShmReader()
            use_dummy = not shm_reader.available
            if use_dummy:
                print("SHM reader is unavailable; using dummy data instead.")
        except Exception as e:
            print(f"Failed to initialize SHM reader: ({e}); using dummy data instead.")
            use_dummy = True

    def send_loop() -> None:
        """
        Stream SHM packets when available; otherwise send finite dummy packets.
        """
        if use_dummy:
            count = 0
            while count < MAX_DUMMY_PACKETS:
                packet = packet_from_sensor_dict(_dummy_sensor_dict(count))
                mode.send(packet)
                print(f'Dummy packet #{count} sent.')
                count += 1
                time.sleep(0.001)
        else: 
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
