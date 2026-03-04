# Data Transfer | FA25

Wireless data transfer system that serializes structured vehicle sensor data (JSON) using Protocol Buffers, transmits it via the chosen transmission method, then decodes it back into readable data to be visualized on the Race Engineer Dashboard during competition.  

Transmission Methods (as of Nov 2025):
1. [minimodem](https://www.whence.com/minimodem/): Encodes the data into audio tones using Frequency-Key Shifting (FSK) modulation.
2. UDP over Starlink 

---

## Installation

Windows users: minimodem does not run natively on Windows. Please install WSL2 with Ubuntu and run this project inside the WSL terminal.

```bash
git clone https://github.com/cornellev/data-transfer.git
cd data-transfer
```

Install dependencies:
```bash
# macOS
brew install minimodem

# Windows/Linux (Ubuntu)
sudo apt-get install minimodem

pip install -r requirements.txt
```

---

## How to Run
```bash
mode = 'udp' | 'modem' # Choose one 
```

#### Sending Data 
Run the sender module to serialize a message and transmit it:
```bash
python -m sender.sender --mode mode
```

#### Receiving Data 
Run the receiver module to listen for incoming data, decode it, and deserialize it:
```bash
python -m receiver.receiver --mode mode
```

---

## Project Structure
```
DATA-TRANSFER/
│
├── modes/                 # Transmission modes 
│   ├── interface.py
│   ├── modem_mode.py
|   ├── udp_mode.py
│   └── __init__.py
|
├── receiver/              # Receiver code 
│   └── receiver.py
│
├── schema/                # Protobuf schema definitions
│   ├── data.proto
│   ├── data_pb2.py
│   └── __init__.py
│
├── sender/                # Sender code 
│   └── sender.py
│
├── cellular_modem.py      # CellularModem class 
├── config.py              # Data/voice call configurations
├── README.md              
├── requirements.txt       
└── .gitignore             
```
---

## Adding/Editing Protobuf Schemas
This system uses [Protocol Buffers](https://protobuf.dev/getting-started/pythontutorial/) to define structured messages.
To add or edit a schema, modify the `.proto` file in `/schema`, then regenerate the Python bindings:
```bash
protoc --python_out=schema schema/data.proto
```