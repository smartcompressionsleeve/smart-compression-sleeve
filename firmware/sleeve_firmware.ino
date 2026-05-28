// ============================================================
// SMART COMPRESSION SLEEVE — STABLE PACKET SKETCH (with latency instrumentation)
// Safer serial: 115200 baud, 30 Hz output
// Packet format still matches Python:
//   DATA,t_ms,imu2(6),imu5(6),emg,hr,confidence
// Added: periodic LOOP,<avg_us>,<count> report for Arduino-side latency profiling
// ============================================================

#include <Wire.h>
#include "ICM_20948.h"

// -------------------- Pins --------------------
#define EMG_PIN A2
#define PPG_PIN A1

#define MUX_ADDR 0x70
#define MUX_CH_IMU_UPPER 2
#define MUX_CH_IMU_FOREARM 5

// -------------------- Timing --------------------
const unsigned long SERIAL_BAUD = 115200;
const unsigned long DATA_INTERVAL_US = 33333;     // 30 Hz
const unsigned long PPG_SAMPLE_INTERVAL_MS = 20;  // 50 Hz

// -------------------- Latency instrumentation --------------------
const unsigned long LOOP_REPORT_INTERVAL = 100;   // report avg loop time every 100 iters

// -------------------- PPG --------------------
const int PPG_AVG_WINDOW = 8;
const int PPG_MIN_AMPLITUDE = 20;
const unsigned long MIN_BEAT_INTERVAL_MS = 350;
const unsigned long MAX_BEAT_INTERVAL_MS = 2000;
const unsigned long MINMAX_RESET_INTERVAL_MS = 3000;

const uint16_t BPM_MIN = 40;
const uint16_t BPM_MAX = 180;

// -------------------- Motion --------------------
const float MOTION_GYRO_THRESHOLD_DPS = 80.0f;

// ============================================================
// GLOBAL STATE
// ============================================================

ICM_20948_I2C imuUpper;
ICM_20948_I2C imuForearm;

bool imuUpperOk = false;
bool imuForearmOk = false;

unsigned long lastDataUs = 0;
unsigned long lastPpgSampleMs = 0;

int16_t imu2_ax = 0, imu2_ay = 0, imu2_az = 0;
float imu2_gx = 0, imu2_gy = 0, imu2_gz = 0;

int16_t imu5_ax = 0, imu5_ay = 0, imu5_az = 0;
float imu5_gx = 0, imu5_gy = 0, imu5_gz = 0;

int emgValue = 0;
uint16_t heartRate = 0;
uint8_t ppgConfidence = 0;

// PPG smoothing
int ppgBuffer[PPG_AVG_WINDOW] = {0};
int ppgBufferIdx = 0;
long ppgBufferSum = 0;
int ppgSmoothed = 0;
int ppgPrevSmoothed = 0;

int ppgRunningMin = 1023;
int ppgRunningMax = 0;
unsigned long lastMinMaxResetMs = 0;

unsigned long lastBeatMs = 0;
uint16_t computedBpm = 0;
uint16_t lastStableBpm = 0;
bool inMotion = false;

// Loop time tracking
unsigned long loopTimeAccumUs = 0;
unsigned long loopCount = 0;

// ============================================================
// I2C MUX
// ============================================================

void selectMux(uint8_t channel) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void disableMux() {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

// ============================================================
// IMU
// ============================================================

bool initImu(ICM_20948_I2C &imu, uint8_t channel, const char *label) {
  selectMux(channel);
  delay(30);

  imu.begin(Wire, 0);  // AD0 = GND, address 0x68

  bool ok = (imu.status == ICM_20948_Stat_Ok);

  Serial.print("# ");
  Serial.print(label);
  Serial.println(ok ? " OK" : " FAILED");

  disableMux();
  delay(10);

  return ok;
}

void readImu(
  uint8_t channel,
  ICM_20948_I2C &imu,
  int16_t &ax,
  int16_t &ay,
  int16_t &az,
  float &gx,
  float &gy,
  float &gz
) {
  selectMux(channel);
  delayMicroseconds(300);

  imu.getAGMT();

  // DIAGNOSTIC: if the I2C read did not complete cleanly, flag it and
  // KEEP the previous values rather than writing a partial/garbage frame.
  // The "#" prefix means the Python bridge prints this as an [arduino]
  // comment and ignores it for parsing — no bridge change required.
  if (imu.status != ICM_20948_Stat_Ok) {
    Serial.print("# BAD_READ ch");
    Serial.print(channel);
    Serial.print(" status=");
    Serial.println(imu.status);
    disableMux();
    return;  // leave ax/ay/az/gx/gy/gz at their previous values
  }

  ax = (int16_t)imu.accX();
  ay = (int16_t)imu.accY();
  az = (int16_t)imu.accZ();

  gx = imu.gyrX();
  gy = imu.gyrY();
  gz = imu.gyrZ();

  disableMux();
}

void updateIMUs() {
  if (imuUpperOk) {
    readImu(
      MUX_CH_IMU_UPPER,
      imuUpper,
      imu2_ax,
      imu2_ay,
      imu2_az,
      imu2_gx,
      imu2_gy,
      imu2_gz
    );
  }

  if (imuForearmOk) {
    readImu(
      MUX_CH_IMU_FOREARM,
      imuForearm,
      imu5_ax,
      imu5_ay,
      imu5_az,
      imu5_gx,
      imu5_gy,
      imu5_gz
    );
  }
}

// ============================================================
// EMG
// ============================================================

void updateEMG() {
  emgValue = analogRead(EMG_PIN);
}

// ============================================================
// PPG
// ============================================================

void samplePpgAndDetectBeat(unsigned long nowMs) {
  int raw = analogRead(PPG_PIN);

  ppgBufferSum -= ppgBuffer[ppgBufferIdx];
  ppgBuffer[ppgBufferIdx] = raw;
  ppgBufferSum += raw;
  ppgBufferIdx = (ppgBufferIdx + 1) % PPG_AVG_WINDOW;

  ppgSmoothed = ppgBufferSum / PPG_AVG_WINDOW;

  if (ppgSmoothed < ppgRunningMin) ppgRunningMin = ppgSmoothed;
  if (ppgSmoothed > ppgRunningMax) ppgRunningMax = ppgSmoothed;

  if (nowMs - lastMinMaxResetMs >= MINMAX_RESET_INTERVAL_MS) {
    lastMinMaxResetMs = nowMs;
    ppgRunningMin = (ppgRunningMin + ppgSmoothed) / 2;
    ppgRunningMax = (ppgRunningMax + ppgSmoothed) / 2;
  }

  int amplitude = ppgRunningMax - ppgRunningMin;

  if (amplitude < PPG_MIN_AMPLITUDE) {
    ppgPrevSmoothed = ppgSmoothed;
    return;
  }

  int threshold = ppgRunningMin + (amplitude * 6 / 10);

  bool rising =
      (ppgPrevSmoothed < threshold) &&
      (ppgSmoothed >= threshold);

  if (rising && (nowMs - lastBeatMs) >= MIN_BEAT_INTERVAL_MS) {
    unsigned long beatInterval = nowMs - lastBeatMs;
    lastBeatMs = nowMs;

    if (beatInterval < 5000) {
      uint16_t bpm = (uint16_t)(60000UL / beatInterval);

      if (bpm >= BPM_MIN && bpm <= BPM_MAX) {
        computedBpm = (computedBpm == 0) ? bpm : (computedBpm + bpm) / 2;
      }
    }
  }

  if ((nowMs - lastBeatMs) > MAX_BEAT_INTERVAL_MS) {
    computedBpm = 0;
  }

  ppgPrevSmoothed = ppgSmoothed;
}

// ============================================================
// HR OUTPUT
// ============================================================

void updateMotionGate() {
  float gyroMag = sqrtf(
    imu5_gx * imu5_gx +
    imu5_gy * imu5_gy +
    imu5_gz * imu5_gz
  );

  inMotion = gyroMag > MOTION_GYRO_THRESHOLD_DPS;
}

void updateHrOutput(unsigned long nowMs) {
  if (!inMotion) {
    if (computedBpm > 0) {
      lastStableBpm = computedBpm;
    }

    heartRate = computedBpm;
    ppgConfidence =
        ((nowMs - lastBeatMs) < 1500 && computedBpm > 0) ? 95 : 0;
  } else {
    heartRate = lastStableBpm;
    ppgConfidence = (lastStableBpm > 0) ? 50 : 0;
  }
}

// ============================================================
// DATA PACKET
// ============================================================

void sendDataPacket() {
  Serial.print("DATA,");
  Serial.print(millis());
  Serial.print(",");

  Serial.print(imu2_ax);
  Serial.print(",");
  Serial.print(imu2_ay);
  Serial.print(",");
  Serial.print(imu2_az);
  Serial.print(",");
  Serial.print(imu2_gx, 2);
  Serial.print(",");
  Serial.print(imu2_gy, 2);
  Serial.print(",");
  Serial.print(imu2_gz, 2);
  Serial.print(",");

  Serial.print(imu5_ax);
  Serial.print(",");
  Serial.print(imu5_ay);
  Serial.print(",");
  Serial.print(imu5_az);
  Serial.print(",");
  Serial.print(imu5_gx, 2);
  Serial.print(",");
  Serial.print(imu5_gy, 2);
  Serial.print(",");
  Serial.print(imu5_gz, 2);
  Serial.print(",");

  Serial.print(emgValue);
  Serial.print(",");
  Serial.print(heartRate);
  Serial.print(",");
  Serial.println(ppgConfidence);
}

// ============================================================
// LOOP TIME REPORTER
// ============================================================

void reportLoopTimeIfReady() {
  if (loopCount >= LOOP_REPORT_INTERVAL) {
    unsigned long avgUs = loopTimeAccumUs / loopCount;
    Serial.print("LOOP,");
    Serial.print(avgUs);
    Serial.print(",");
    Serial.println(loopCount);
    loopTimeAccumUs = 0;
    loopCount = 0;
  }
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(SERIAL_BAUD);

  while (!Serial) {
    delay(10);
  }

  delay(1000);

  Serial.println("# Smart Sleeve STABLE packet sketch (with latency profiling)");
  Serial.println("# Baud: 115200");
  Serial.println("# Output: DATA,t_ms,imu2(6),imu5(6),emg,hr,confidence @ 30Hz");
  Serial.println("# Also: LOOP,<avg_us>,<count> every 100 iterations");

  Wire.begin();
  Wire.setClock(400000);

  pinMode(EMG_PIN, INPUT);
  pinMode(PPG_PIN, INPUT);

  imuUpperOk = initImu(
    imuUpper,
    MUX_CH_IMU_UPPER,
    "IMU2 upper CH2"
  );

  imuForearmOk = initImu(
    imuForearm,
    MUX_CH_IMU_FOREARM,
    "IMU5 forearm CH5"
  );

  Serial.println("# Ready.");
}

// ============================================================
// LOOP
// ============================================================

void loop() {
  unsigned long loopStartUs = micros();

  unsigned long nowUs = loopStartUs;
  unsigned long nowMs = millis();

  if (nowMs - lastPpgSampleMs >= PPG_SAMPLE_INTERVAL_MS) {
    lastPpgSampleMs = nowMs;
    samplePpgAndDetectBeat(nowMs);
  }

  if (nowUs - lastDataUs >= DATA_INTERVAL_US) {
    lastDataUs = nowUs;

    updateIMUs();
    updateEMG();
    updateMotionGate();
    updateHrOutput(nowMs);

    sendDataPacket();
  }

  // Track loop time and emit periodic report
  loopTimeAccumUs += (micros() - loopStartUs);
  loopCount++;
  reportLoopTimeIfReady();
}
