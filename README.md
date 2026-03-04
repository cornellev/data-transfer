# Redundant Telemetry Protocol

A backup wireless telemetry protocol that converts vehicle sensor data (JSON) into Protocol Buffer packets, transmits them via the selected communication method, and decodes the data for visualization on the Race Engineer Dashboard.

## Important UDP Note

UDP support remains in this repository for testing and validation only.

For competition, ROS is the primary telemetry method; use UDP only for local testing and development.

Transmission methods:
1. [minimodem](https://www.whence.com/minimodem/): Encodes data into audio tones using FSK modulation.
2. UDP over Starlink (**testing only; not for primary telemetry**)

---

## Installation

Windows users: minimodem does not run natively on Windows. Install WSL2 with Ubuntu and run this project in the WSL terminal for modem mode.

```bash
git clone https://github.com/cornellev/redundant-telemetry.git
cd redundant-telemetry
```

---

## Dependencies

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install minimodem if you plan to use modem mode:
```bash
# macOS
brew install minimodem

# Windows/Linux (Ubuntu/WSL)
sudo apt-get update && sudo apt-get install -y minimodem
```

SHM Reader dependency for `sender.py`:
- Install or include the [UC26 sensor reader repo](https://github.com/cornellev/uc26_sensor_reader/tree/main) in `PYTHONPATH` so sender can import `read_shm.py`.
- This dependency is required for SHM-based telemetry input.

---

## Configuration

Create your local environment file from the template:

```bash
cp .env.example .env
```

Set values in `.env` (or export equivalent environment variables):
- `DATA_TRANSFER_BAUD`
- `DATA_TRANSFER_UDP_HOST`
- `DATA_TRANSFER_UDP_PORT`
- `DATA_TRANSFER_SENDER_NUMBER`
- `DATA_TRANSFER_RECEIVER_NUMBER`
- `DATA_TRANSFER_MODEM_PORT`
- `DATA_TRANSFER_MODEM_POWER_KEY`
- `DATA_TRANSFER_MODEM_BAUD`

---

## How to Run

This repo uses a `src` layout, so set `PYTHONPATH=src` before running modules.

### PowerShell
```powershell
$env:PYTHONPATH="src"
python -m redundant_telemetry.receiver.receiver --mode udp
python -m redundant_telemetry.sender.sender --mode udp
```

### Bash/Zsh
```bash
export PYTHONPATH=src
python -m redundant_telemetry.receiver.receiver --mode udp
python -m redundant_telemetry.sender.sender --mode udp
```

Use `--mode modem` instead of `--mode udp` to transfer data via minimodem.

### Example With External SHM Module (Bash/Zsh)
```bash
export PYTHONPATH=/home/pi/redundant-telemetry/src:/home/pi/uc26-sensor-logger
python -m redundant_telemetry.sender.sender --mode udp
```

---

## Runtime Behavior

Sender behavior:
- Uses external `read_shm.py` (`SensorShmReader`) to read SHM snapshots.
- Current implementation includes a finite dummy fallback (`MAX_DUMMY_PACKETS`) if SHM import/init is unavailable.

Receiver behavior:
- Modem mode: powers on the SIM7600, answers an incoming call, receives/decodes packets in a loop, then hangs up.
- UDP mode (current implementation): uses a `2.0` second idle timeout to exit if no packets are received (this timeout is customizable in `receiver.py`).

---

## Project Structure
```text
redundant-telemetry/
|-- src/
|   `-- redundant_telemetry/
|       |-- config.py                     # Runtime configuration for transmission mode
|       |-- hardware/
|       |   |-- cellular_modem.py         # SIM7600 GPIO + serial control helper
|       |   `-- __init__.py
|       |-- modes/
|       |   |-- interface.py
|       |   |-- modem_mode.py             # minimodem transfer implementation
|       |   |-- udp_mode.py               # UDP transfer implementation
|       |   `-- __init__.py
|       |-- receiver/
|       |   |-- receiver.py               # Receives packets and decodes protobuf frames
|       |   `-- __init__.py
|       |-- sender/
|       |   |-- sender.py                 # Reads SHM (or dummy), serializes, and sends
|       |   `-- __init__.py
|       |-- schema/                       # Generated protobuf Python runtime files
|       |   |-- data_pb2.py
|       |   `-- __init__.py
|       `-- __init__.py
|-- schema/                               # Protobuf source schema directory
|   `-- data.proto
|-- .env.example
|-- requirements.txt
`-- README.md
```

---

## Sensor Protobuf Schema

`Sensors` message fields:
- `seq`
- `global_ts`
- `power { ts, current, voltage }`
- `steering { ts, brake_pressure, turn_angle }`
- `rpm_front { ts, rpm_left, rpm_right }`
- `rpm_back { ts, rpm_left, rpm_right }`
- `gps { ts, gps_lat, gps_long }`
- `motor { ts, rpm, throttle }`

---

## Adding/Editing Protobuf Schemas

This system uses [Google Protocol Buffers](https://protobuf.dev/getting-started/pythontutorial/) to define structured messages.

### Edit Existing Schema

1. Edit `schema/data.proto`.
2. Regenerate Python bindings:

```bash
protoc --python_out=src/redundant_telemetry schema/data.proto
```

### Add New Schema 

1. Add a new `.proto` file under `schema/` (example: `schema/new_data.proto`).
2. Generate Python bindings for the new file:

```bash
protoc --python_out=src/redundant_telemetry schema/new_data.proto
```

3. Commit both:
- the `.proto` source file in `schema/`
- the generated `*_pb2.py` file in `src/redundant_telemetry/schema/`
