# Redundant Telemetry Protocol

A backup wireless telemetry protocol that converts vehicle sensor data (JSON) into Protocol Buffer packets, transmits them via the selected communication method, and decodes the data for visualization on the Race Engineer Dashboard.

Transmission Methods:
1. [minimodem](https://www.whence.com/minimodem/): Encodes the data into audio tones using Frequency-Key Shifting (FSK) modulation.
2. UDP over Starlink (**testing only; not for primary telemetry**)

## Important UDP Note

UDP support remains in this repository for testing and validation only.

For competition, ROS is the primary telemetry method; use UDP only for local testing and development.

---

## Installation

Windows users: minimodem does not run natively on Windows. Please install WSL2 with Ubuntu and run this project inside the WSL terminal for modem mode.

```bash
git clone https://github.com/cornellev/data-transfer.git
cd data-transfer
pip install -r requirements.txt
```

Install minimodem only if you plan to use modem mode:
```bash
# macOS
brew install minimodem

# Windows/Linux (Ubuntu/WSL)
sudo apt-get update && sudo apt-get install -y minimodem
```

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
python -m data_transfer.receiver.receiver --mode udp
python -m data_transfer.sender.sender --mode udp
```

### Bash/Zsh
```bash
export PYTHONPATH=src
python -m data_transfer.receiver.receiver --mode udp
python -m data_transfer.sender.sender --mode udp
```

Use `--mode modem` instead of `--mode udp` to transfer data via minimodem.

---

## Project Structure
```text
data-transfer/
|-- src/                                  
|   `-- data_transfer/                    
|       |-- config.py                     # Runtime configuration for data transfer
|       |-- hardware/                     
|       |   |-- cellular_modem.py         # SIM7600 GPIO + serial control helper
|       |   `-- __init__.py              
|       |-- modes/                       
|       |   |-- interface.py              
|       |   |-- modem_mode.py             # minimodem transfer implementation
|       |   |-- udp_mode.py               # UDP transfer implementation
|       |   `-- __init__.py               
|       |-- receiver/                     
|       |   |-- receiver.py               # Receives and decodes data packets
|       |   `-- __init__.py               
|       |-- sender/                       
|       |   |-- sender.py                 # Builds, serializes, and sends data pakets
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

## Adding/Editing Protobuf Schemas

This system uses [Google Protocol Buffers](https://protobuf.dev/getting-started/pythontutorial/) to define structured messages.

1. Edit `schema/data.proto`.
2. Regenerate Python bindings into the runtime package:

```bash
protoc --python_out=src/data_transfer/schema schema/data.proto
```
