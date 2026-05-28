# Firmware

Arduino C++ firmware for the sleeve's microcontroller. Handles I²C multiplexer
configuration, IMU/EMG/PPG sampling, and synchronized USB serial streaming
to the Python bridge.

## Hardware

- **MCU:** Adafruit Feather nRF52840
- **IMUs:** 2× ICM-20948, multiplexed via TCA9548A
  - Upper arm: I²C channel 2
  - Forearm: I²C channel 5
- **EMG:** DFRobot Gravity Analog EMG (SEN0240) on analog pin **A1**
- **PPG:** DFRobot Gravity Heart Rate (SEN0203) on analog pin **A0**

## Required Arduino libraries

Install via the Arduino Library Manager:

- `SparkFun ICM-20948 Arduino Library`
- `Adafruit TCA9548A` (or equivalent I²C mux library)
- Adafruit nRF52 board support (via Boards Manager)

## Flashing

1. Open the `.ino` sketch in the Arduino IDE
2. Select **Tools → Board → Adafruit Feather nRF52840 Express**
3. Select the correct serial port
4. Click **Upload**

## Serial output format

USB serial at **115200 baud**. Each packet contains synchronized samples
across all three sensor modalities (see the firmware source for the exact
packet format).
