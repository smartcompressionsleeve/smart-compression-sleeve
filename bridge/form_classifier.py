"""
Smart Compression Sleeve — Form Classifier Inference
=====================================================
Loads a trained Random Forest model and runs predictions on raw IMU data.

Designed to be used by:
  - Katie's Flutter/WebSocket backend (real-time feedback during exercise)
  - Post-session analysis scripts (per-rep accuracy report)
  - DR5 demo pipeline (live BLE stream → classification)

Two prediction modes:
  1. predict_rep(buffer)        - call when a full rep is complete
  2. predict_window(buffer)     - call every ~500ms on a sliding window
                                  for live in-rep feedback

Usage example:
    from form_classifier import FormClassifier

    clf = FormClassifier("results/random_forest_mode.pkl")

    # Per-rep mode (accurate, ~3-5 sec latency from rep start)
    label, confidence = clf.predict_rep(rep_buffer)

    # Live sliding-window mode (responsive, ~500ms latency)
    label, confidence = clf.predict_window(recent_buffer)

Buffer format:
    A list of dicts (or pandas DataFrame) with these keys per sample:
        t (sec), ax_u, ay_u, az_u, gx_u, gy_u, gz_u,
                 ax_f, ay_f, az_f, gx_f, gy_f, gz_f
    Sampled at any rate (will be resampled internally to 50 Hz).
    Accel in mg, gyro in dps. Match your firmware's units.
"""

import pickle
from collections import deque
from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------------------------------
# Config — keep in sync with extract_features.py
# ---------------------------------------------------------------------------
TARGET_FS = 50.0
COMP_FILTER_ALPHA = 0.98
LOWPASS_HZ = 4.0
MIN_REP_DIST_SEC = 1.0
MIN_PEAK_PROMINENCE = 8.0
XCORR_MAX_LAG_SEC = 0.2

FEATURE_NAMES = [
    "elbow_angle_max",
    "elbow_angle_min",
    "elbow_angle_range",
    "upper_gyro_mag_mean",
    "upper_gyro_mag_max",
    "upper_pitch_std",
    "xcorr_upper_forearm",
    "forearm_angvel_range",
    "rep_duration_sec",
]

# Sliding-window params for live mode
WINDOW_SEC = 3.0      # how much recent data to look at
WINDOW_HOP_SEC = 0.5  # how often to call predict_window


# ---------------------------------------------------------------------------
# Signal processing primitives (same as extract_features.py)
# ---------------------------------------------------------------------------
def _pitch_from_accel(ax, ay, az):
    return np.degrees(np.arctan2(ax, np.sqrt(ay * ay + az * az)))


def _complementary_filter(pitch_accel, gyro_rate, dt, alpha=COMP_FILTER_ALPHA):
    n = len(pitch_accel)
    pitch = np.zeros(n)
    pitch[0] = pitch_accel[0] if not np.isnan(pitch_accel[0]) else 0.0
    for k in range(1, n):
        gyro_est = pitch[k - 1] + gyro_rate[k] * dt
        pitch[k] = alpha * gyro_est + (1 - alpha) * pitch_accel[k]
    return pitch


def _lowpass(x, fs, cutoff_hz=LOWPASS_HZ, order=4):
    nyq = fs / 2
    # Need at least order*3 samples for filtfilt
    if len(x) < order * 3 + 1:
        return x.copy()
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, x)


def _gyro_magnitude(gx, gy, gz):
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def _normalized_xcorr(a, b, max_lag_samples):
    a = np.asarray(a) - np.mean(a)
    b = np.asarray(b) - np.mean(b)
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return 0.0
    best = -1.0
    n = len(a)
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            x, y = a[-lag:], b[: n + lag]
        elif lag > 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 3:
            continue
        corr = np.sum(x * y) / (len(x) * sa * sb)
        if corr > best:
            best = corr
    return float(best)


# ---------------------------------------------------------------------------
# Buffer normalization
# ---------------------------------------------------------------------------
def _normalize_buffer(buffer):
    """
    Accept buffer as DataFrame, list-of-dicts, or dict-of-arrays.
    Returns a DataFrame indexed 0..N-1 with the standard 13 columns,
    resampled to TARGET_FS.
    """
    if isinstance(buffer, pd.DataFrame):
        df = buffer.copy()
    elif isinstance(buffer, dict):
        df = pd.DataFrame(buffer)
    elif isinstance(buffer, (list, tuple)) and len(buffer) > 0 and isinstance(buffer[0], dict):
        df = pd.DataFrame(buffer)
    else:
        raise ValueError("buffer must be DataFrame, dict-of-arrays, or list-of-dicts")

    required = {"t", "ax_u", "ay_u", "az_u", "gx_u", "gy_u", "gz_u",
                "ax_f", "ay_f", "az_f", "gx_f", "gy_f", "gz_f"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"buffer missing required columns: {missing}")

    df = df.sort_values("t").reset_index(drop=True)
    t = df["t"].values
    if t[-1] - t[0] < 0.5:
        return None  # too short to do anything useful
    t_grid = np.arange(t[0], t[-1], 1.0 / TARGET_FS)
    out = {"t": t_grid - t_grid[0]}
    for col in ["ax_u", "ay_u", "az_u", "gx_u", "gy_u", "gz_u",
                "ax_f", "ay_f", "az_f", "gx_f", "gy_f", "gz_f"]:
        out[col] = np.interp(t_grid, t, df[col].values)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Feature computation (per buffer segment)
# ---------------------------------------------------------------------------
def _compute_elbow_angle(df):
    dt = 1.0 / TARGET_FS
    upper_pitch_acc = _pitch_from_accel(df["ax_u"].values, df["ay_u"].values, df["az_u"].values)
    forearm_pitch_acc = _pitch_from_accel(df["ax_f"].values, df["ay_f"].values, df["az_f"].values)
    upper_pitch = _complementary_filter(upper_pitch_acc, df["gx_u"].values, dt)
    forearm_pitch = _complementary_filter(forearm_pitch_acc, df["gx_f"].values, dt)
    return forearm_pitch - upper_pitch, upper_pitch


def _features_for_segment(df, elbow, upper_pitch, start, end):
    """Compute the 9 features over df[start:end+1]."""
    e = elbow[start:end + 1]
    up = upper_pitch[start:end + 1]
    um = _gyro_magnitude(df["gx_u"].values[start:end + 1],
                         df["gy_u"].values[start:end + 1],
                         df["gz_u"].values[start:end + 1])
    fm = _gyro_magnitude(df["gx_f"].values[start:end + 1],
                         df["gy_f"].values[start:end + 1],
                         df["gz_f"].values[start:end + 1])
    max_lag = int(XCORR_MAX_LAG_SEC * TARGET_FS)

    return {
        "elbow_angle_max": float(np.max(e)),
        "elbow_angle_min": float(np.min(e)),
        "elbow_angle_range": float(np.max(e) - np.min(e)),
        "upper_gyro_mag_mean": float(np.mean(um)),
        "upper_gyro_mag_max": float(np.max(um)),
        "upper_pitch_std": float(np.std(up)),
        "xcorr_upper_forearm": _normalized_xcorr(um, fm, max_lag),
        "forearm_angvel_range": float(np.max(fm) - np.min(fm)),
        "rep_duration_sec": float((end - start) / TARGET_FS),
    }


# ---------------------------------------------------------------------------
# Main classifier wrapper
# ---------------------------------------------------------------------------
class FormClassifier:
    """
    Wraps a trained Random Forest with feature extraction so callers can
    pass raw IMU buffers and get predictions back.
    """

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.classes = bundle["classes"]
        # Sanity check: make sure feature contract matches
        if list(self.feature_names) != FEATURE_NAMES:
            raise ValueError(
                f"Model feature names don't match this inference module.\n"
                f"  model expects: {self.feature_names}\n"
                f"  module provides: {FEATURE_NAMES}"
            )

    # ---------------- Per-rep mode ----------------
    def predict_rep(self, buffer):
        """
        Classify a buffer assumed to contain one complete rep.
        Returns (predicted_label, confidence, features_dict).
        """
        df = _normalize_buffer(buffer)
        if df is None or len(df) < int(TARGET_FS * 1.0):
            return None, 0.0, {}

        elbow, upper_pitch = _compute_elbow_angle(df)
        feats = _features_for_segment(df, elbow, upper_pitch, 0, len(df) - 1)

        return self._predict_from_features(feats)

    # ---------------- Auto-segment-and-predict ----------------
    def predict_session(self, buffer):
        """
        Run full rep segmentation on a long buffer (e.g., a full exercise set).
        Returns a list of (rep_num, label, confidence, features) tuples.
        """
        df = _normalize_buffer(buffer)
        if df is None or len(df) < int(TARGET_FS * 2):
            return []

        elbow, upper_pitch = _compute_elbow_angle(df)
        elbow_filt = _lowpass(elbow, TARGET_FS)

        peaks, _ = find_peaks(
            elbow_filt,
            distance=int(MIN_REP_DIST_SEC * TARGET_FS),
            prominence=MIN_PEAK_PROMINENCE,
        )
        if len(peaks) == 0:
            return []

        # Build valley-bounded reps
        valleys = []
        for i in range(len(peaks) - 1):
            seg = elbow_filt[peaks[i]:peaks[i + 1]]
            valleys.append(peaks[i] + int(np.argmin(seg)))
        pre = int(np.argmin(elbow_filt[:peaks[0]])) if peaks[0] > 0 else 0
        post = peaks[-1] + int(np.argmin(elbow_filt[peaks[-1]:])) if peaks[-1] < len(elbow_filt) - 1 else len(elbow_filt) - 1
        all_valleys = [pre] + valleys + [post]

        results = []
        for i in range(len(all_valleys) - 1):
            start, end = all_valleys[i], all_valleys[i + 1]
            if end - start < int(TARGET_FS * 0.5):
                continue
            feats = _features_for_segment(df, elbow, upper_pitch, start, end)
            label, conf, _ = self._predict_from_features(feats)
            results.append({
                "rep_num": i + 1,
                "label": label,
                "confidence": conf,
                "features": feats,
            })
        return results

    # ---------------- Sliding-window mode ----------------
    def predict_window(self, buffer):
        """
        Classify the most recent WINDOW_SEC of data.
        Designed for live in-rep feedback at ~500ms cadence.
        Returns (label, confidence, features).
        """
        df = _normalize_buffer(buffer)
        if df is None or len(df) < int(TARGET_FS * 1.0):
            return None, 0.0, {}

        # Take the last WINDOW_SEC of samples
        n = min(len(df), int(WINDOW_SEC * TARGET_FS))
        df = df.iloc[-n:].reset_index(drop=True)

        elbow, upper_pitch = _compute_elbow_angle(df)
        feats = _features_for_segment(df, elbow, upper_pitch, 0, len(df) - 1)
        return self._predict_from_features(feats)

    # ---------------- Internal ----------------
    def _predict_from_features(self, feats):
        x = np.array([[feats[name] for name in FEATURE_NAMES]])
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = self.classes[idx]
        confidence = float(proba[idx])
        return label, confidence, feats


# ---------------------------------------------------------------------------
# Real-time streaming helper for BLE callback pattern
# ---------------------------------------------------------------------------
class LiveBuffer:
    """
    Rolling buffer for BLE callback pattern. Push individual IMU samples
    as they arrive; periodically call predict_window to get live form feedback.

    Usage with bleak BLE callback:

        clf = FormClassifier("model.pkl")
        live = LiveBuffer(clf, max_seconds=10)

        def on_ble_packet(data):
            # parse your packet to get timestamp + 6-axis samples for both IMUs
            live.push(t, ax_u, ay_u, az_u, gx_u, gy_u, gz_u,
                         ax_f, ay_f, az_f, gx_f, gy_f, gz_f)

        # In a separate timer (e.g., every 500 ms):
        result = live.predict_latest()
        if result:
            label, confidence, _ = result
            send_to_app(label, confidence)
    """

    def __init__(self, classifier: FormClassifier, max_seconds: float = 10.0):
        self.clf = classifier
        # Buffer up to max_seconds of samples — assumes ~50 Hz, but accepts more
        self._buffer = deque(maxlen=int(max_seconds * 200))  # generous cap
        self.max_seconds = max_seconds

    def push(self, t, ax_u, ay_u, az_u, gx_u, gy_u, gz_u,
             ax_f, ay_f, az_f, gx_f, gy_f, gz_f):
        self._buffer.append({
            "t": t,
            "ax_u": ax_u, "ay_u": ay_u, "az_u": az_u,
            "gx_u": gx_u, "gy_u": gy_u, "gz_u": gz_u,
            "ax_f": ax_f, "ay_f": ay_f, "az_f": az_f,
            "gx_f": gx_f, "gy_f": gy_f, "gz_f": gz_f,
        })

    def predict_latest(self):
        """Run sliding-window prediction on the most recent window."""
        if len(self._buffer) < int(TARGET_FS * 1.0):
            return None
        return self.clf.predict_window(list(self._buffer))

    def clear(self):
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/random_forest_mode.pkl")
    ap.add_argument("--csv", help="raw long-format CSV to test on")
    args = ap.parse_args()

    clf = FormClassifier(args.model)
    print(f"Loaded model. Classes: {clf.classes}")

    if args.csv:
        # Convert long-format CSV to the buffer format
        raw = pd.read_csv(args.csv)
        t = pd.to_datetime(raw["timestamp"])
        raw["t"] = (t - t.iloc[0]).dt.total_seconds()

        upper = raw[raw["imu_ch"] == 2].reset_index(drop=True)
        forearm = raw[raw["imu_ch"] == 5].reset_index(drop=True)

        t_grid = np.arange(
            max(upper["t"].iloc[0], forearm["t"].iloc[0]),
            min(upper["t"].iloc[-1], forearm["t"].iloc[-1]),
            1.0 / TARGET_FS
        )
        buffer = pd.DataFrame({
            "t": t_grid,
            "ax_u": np.interp(t_grid, upper["t"], upper["ax_mg"]),
            "ay_u": np.interp(t_grid, upper["t"], upper["ay_mg"]),
            "az_u": np.interp(t_grid, upper["t"], upper["az_mg"]),
            "gx_u": np.interp(t_grid, upper["t"], upper["gx_dps"]),
            "gy_u": np.interp(t_grid, upper["t"], upper["gy_dps"]),
            "gz_u": np.interp(t_grid, upper["t"], upper["gz_dps"]),
            "ax_f": np.interp(t_grid, forearm["t"], forearm["ax_mg"]),
            "ay_f": np.interp(t_grid, forearm["t"], forearm["ay_mg"]),
            "az_f": np.interp(t_grid, forearm["t"], forearm["az_mg"]),
            "gx_f": np.interp(t_grid, forearm["t"], forearm["gx_dps"]),
            "gy_f": np.interp(t_grid, forearm["t"], forearm["gy_dps"]),
            "gz_f": np.interp(t_grid, forearm["t"], forearm["gz_dps"]),
        })

        print(f"\nLoaded {len(buffer)} samples ({buffer['t'].iloc[-1]:.1f} sec)")

        # Run session-level prediction (auto rep segmentation)
        results = clf.predict_session(buffer)
        print(f"\n=== Per-rep predictions ({len(results)} reps detected) ===")
        for r in results:
            print(f"  Rep {r['rep_num']}: {r['label']:20s} (confidence={r['confidence']:.2f})")

        # Run sliding-window on the last 3 sec
        label, conf, _ = clf.predict_window(buffer)
        print(f"\n=== Live window prediction (last {WINDOW_SEC}s) ===")
        print(f"  Label: {label} (confidence={conf:.2f})")
