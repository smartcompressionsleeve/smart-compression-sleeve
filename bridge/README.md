# Serial Bridge

Python script that reads sensor packets from the Feather over USB serial,
runs the trained Random Forest on each completed rep, and broadcasts results
to the Flutter app over WebSocket.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python bridge.py --port /dev/tty.usbmodemXXXX
```

Find the correct serial port:
- **Mac/Linux:** `ls /dev/tty.usbmodem*` or `ls /dev/ttyACM*`
- **Windows:** check Device Manager → Ports (COM & LPT)

The WebSocket server listens on port `8765` by default. The Flutter app
connects to this for live form feedback and rep counts.

## Safety guards

- `GYRO_SANITY_LIMIT_DPS = 250.0` — flags suspicious gyroscope spikes
  (detect-only by default; drop logic can be enabled)
- `BAD_READ` status check on each sensor packet, mirroring the firmware
