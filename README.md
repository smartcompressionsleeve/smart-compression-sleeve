# Smart Compression Sleeve

Real-time exercise form classifier in a wearable compression sleeve.
A senior design project at UC Riverside, Bourns College of Engineering (BIEN, 2026).

🌐 **[Live project site →](https://smartcompressionsleeve.github.io/smartsleeveinfo/)**

---

## What it does

The Smart Compression Sleeve is a multi-sensor wearable that classifies bicep
curl form in real time. Two sensor pods on a single-layer compression sleeve
carry an Adafruit Feather nRF52840, two ICM-20948 IMUs (multiplexed through a
TCA9548A), a DFRobot Gravity EMG sensor, and a DFRobot Gravity PPG sensor.
Data streams over USB serial to a Python bridge that runs a per-exercise
Random Forest classifier, with live feedback shown in a Flutter mobile app
backed by Supabase.

Bicep curl is the primary validated exercise: **96.6% ± 3.5%** four-class
form-classification accuracy across 417 labeled reps (GroupKFold by trial).

---

## Repository layout

| Folder | Purpose | Stack |
|---|---|---|
| [`firmware/`](firmware/) | MCU firmware: I²C mux drivers, sensor sampling loop, USB serial streaming | Arduino C++ |
| [`ml-pipeline/`](ml-pipeline/) | Feature extraction, Random Forest training, GroupKFold cross-validation | Python · scikit-learn |
| [`bridge/`](bridge/) | Serial → WebSocket bridge that streams data from the sleeve to the Flutter app | Python · pyserial · websockets |
| [`app/`](app/) | iOS Flutter companion app: live form feedback, rep counting, session history | Dart · Flutter · Supabase |

Each folder has its own `README.md` with setup and run instructions.

---

## Hardware

- **MCU:** Adafruit Feather nRF52840
- **IMUs:** 2× ICM-20948 (9-DOF), multiplexed on a TCA9548A I²C bus expander
  (upper arm on CH2, forearm on CH5)
- **EMG:** DFRobot Gravity Analog EMG (SEN0240) on analog pin A1
- **PPG:** DFRobot Gravity Heart Rate (SEN0203) on analog pin A0
- **Power:** 3.7 V Li-Po
- **Form factor:** Single-layer nylon-spandex pull-on compression sleeve with
  two strap-mounted sensor pods (upper arm and forearm)
- **Communication:** Wired USB serial

---

## Team

A six-person undergraduate team at UCR BIEN, advised by Dr. Park and Dr. Masoumi.
See the [Team page](https://smartcompressionsleeve.github.io/smartsleeveinfo/team.html)
on the project site for individual contributions.

---

## License

MIT — see [LICENSE](LICENSE).
