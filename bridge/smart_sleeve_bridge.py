"""
Smart Sleeve Bridge: HYBRID8 (EMG activation % display)
==================================================================================
Base: HYBRID6 working bridge (multi-exercise dispatch + Supabase logging)

What's new vs HYBRID5:
  - Loads one classifier per exercise from a models/ directory
  - Tracks current_exercise (default: bicep_curl, matching Flutter's
    ExerciseManager default) and routes live predictions to the right model
  - New command: {"command": "set_exercise", "exercise": "..."}
  - Live buffer cleared on exercise switch
  - form_label and form_confidence broadcast on every packet

What's new vs HYBRID6 (original):
  - Session-long sample buffer in addition to LiveBuffer
  - At stop_session: predict_session() is run on the full buffer,
    one row per rep is inserted into rep_features, and a summary row
    is upserted into sessions
  - Fail-soft: Supabase outages never crash the bridge

What's new vs HYBRID7:
  - EMG is now broadcast as an *activation percentage* (emg_activation_pct)
    in addition to the raw envelope value (emg / emgValue, unchanged).
  - Normalization is SESSION-PEAK relative: pct = raw / session_peak * 100,
    clamped to [0, 100]. The peak is tracked per session and reset on
    start_session, so the number means "% of your hardest effort this
    session" — a RELATIVE effort indicator, not a calibrated %MVC.
  - A noise floor (EMG_PEAK_FLOOR) prevents the "100% the instant you
    start" artifact: until the running peak rises above the floor, the
    bridge reports 0% rather than dividing by near-noise.
  - HONEST-FRAMING NOTE for DHF/slides: this is NOT %MVC and NOT a clinical
    muscle-activation measurement. The DFRobot EMG remains envelope-only at
    ~19.6 Hz effective rate. Document it as a relative display metric.
  - Raw emg/emgValue fields are kept in the packet, so this change is purely
    additive on the Flutter side (no existing reads break).

What's new vs HYBRID8:
  - SENSOR GLITCH GUARD for the intermittent IMU transients seen even when the
    device is stationary (forearm gyro spiking to ~40 dps on a table). Adds
    frame_is_implausible() (gyro-magnitude check vs GYRO_SANITY_LIMIT_DPS) in
    the parser. Two modes via GLITCH_GUARD_DROP:
      * False (default) = detect-only: logs "[glitch:detect-only]" but still
        processes the frame, for correlating with the firmware's "# BAD_READ".
      * True = drop: a bad frame skips the ML push, session_buffer append, and
        EMG-peak update (so one corrupt frame can't permanently inflate
        emg_session_peak), while still broadcasting a packet so the live view
        does not stall.
  - Pairs with the firmware "# BAD_READ" diagnostic: BAD_READ + glitch = I2C
    bus/mux fault; glitch with no BAD_READ = power or sensor. See DHF.

Flutter contract (from lib/models/exercise.dart):
  exercise.id ∈ {"bicep_curl", "hammer_curl", "tricep_extension"}

Expected files in same folder:
  form_classifier.py
  models/bicep_curl.pkl
  models/hammer_curl.pkl
  models/tricep_extension.pkl
  latency_profiler.py
  .env  (copy from .env.example and fill in)

Required packages:
  pip3 install --break-system-packages pyserial websockets numpy scipy pandas scikit-learn supabase python-dotenv

Required Supabase schema additions (run once):
  alter table rep_features
    add column if not exists form_label text,
    add column if not exists form_confidence double precision,
    add column if not exists exercise_id text;
"""

import asyncio
import json
import math
import os
import time
import uuid
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports
import websockets

from dotenv import load_dotenv
load_dotenv()  # Loads SUPABASE_URL and SUPABASE_ANON_KEY from .env (if present)

from latency_profiler import LatencyProfiler

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("!! supabase package not installed. Run:")
    print("   pip3 install --break-system-packages supabase")
    print("   Bridge will run without Supabase logging.")
    SUPABASE_AVAILABLE = False


# ============================================================
# SETTINGS
# ============================================================

BAUD_RATE = 115200
SERIAL_PORT = None

WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765

DEFAULT_TARGET_REPS = 10

MODELS_DIR = "models"

# Map exercise ID (matches Flutter's Exercise.id) to model filename.
# Adding a new exercise = one line here + drop the .pkl in models/.
EXERCISE_MODEL_FILES = {
    "bicep_curl": "bicep_curl.pkl",
    "hammer_curl": "hammer_curl.pkl",
    "tricep_extension": "tricep_extension.pkl",
}

# Default must match Flutter's ExerciseManager._selected
DEFAULT_EXERCISE = "bicep_curl"

PREDICT_INTERVAL_SEC = 0.5

# ----- EMG activation % normalization -----
# The displayed EMG is normalized against the running peak envelope seen
# during the current session: pct = raw / peak * 100, clamped [0, 100].
# This is a RELATIVE effort indicator, NOT %MVC and NOT clinical.
#
# EMG_PEAK_FLOOR is a minimum the running peak must exceed before we report
# a non-zero percentage. Without it, the first sample is the peak and the
# display would read 100% the instant a session starts. Set this just above
# your resting-baseline envelope value from bench characterization. Units are
# the same raw counts the Arduino sends in DATA field 14. Tune as needed.
EMG_PEAK_FLOOR = 50.0

# ----- Sensor glitch guard -----
# Independent safety net for the intermittent IMU transients observed even
# when the device is stationary (e.g. a forearm gyro reading 40 dps on a
# table). A corrupt frame that reaches the ML buffers can fabricate a rep or
# corrupt a feature window, and a corrupt EMG envelope can permanently inflate
# emg_session_peak for the rest of a session. This guard catches such frames
# regardless of whether the firmware flagged them as BAD_READ.
#
# GYRO_SANITY_LIMIT_DPS: any IMU gyro axis exceeding this magnitude marks the
#   frame as implausible. Real bicep/hammer/tricep motion produces gyro values
#   well under this; observed clean stationary noise is ~±5 dps and the glitch
#   spiked to ~40 dps, so there is a wide safe gap. TUNE from your own bench:
#   set it comfortably above the fastest legitimate rep's peak gyro and well
#   below the glitch magnitude. Setting it very high effectively disables it.
GYRO_SANITY_LIMIT_DPS = 250.0

# GLITCH_GUARD_DROP: behavior when a frame is flagged implausible.
#   False (default) = DETECT-ONLY: log "[glitch]" to console but still process
#     the frame normally. Use this during diagnosis so you can correlate glitch
#     frames with the firmware's "# BAD_READ" lines without losing any data.
#   True = DROP: skip ML push, session_buffer append, and EMG-peak update for
#     the bad frame (still broadcasts a packet to Flutter so the live view does
#     not stall — it just reuses last-known form/EMG state). Flip to True only
#     AFTER you have characterized the glitch and confirmed the threshold.
GLITCH_GUARD_DROP = False

LATENCY_LOG_DIR = "latency_logs"

# Supabase config — loaded from .env, see .env.example
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


# ============================================================
# GLOBAL STATE
# ============================================================

clients = set()
start_time = time.time()
mode = "IDLE"

latest_packet = {
    "angle": 0.0,
    "emg": 0.0,
    "heartRate": 0,
    "ppgConfidence": 0,
    "timestamp": 0.0,
    "imu2": {},
    "imu5": {},
}

calibration = {
    "extended_angle": None,
    "flexed_angle": None,
    "lower_angle": None,
    "upper_angle": None,
}

session = {
    "active": False,
    "session_id": None,
    "start_time": None,
    "target_reps": DEFAULT_TARGET_REPS,
    "rep_count": 0,
}

# Per-exercise ML state.
# classifiers[ex_id] = FormClassifier
# buffers[ex_id]     = LiveBuffer
classifiers = {}
buffers = {}

# Active exercise — updated by Flutter via set_exercise command
current_exercise = DEFAULT_EXERCISE

last_predict_t = 0.0
last_form_label = "normal"
last_form_confidence = 0.0

# Running peak of the raw EMG envelope for the current session.
# Used to normalize raw counts -> activation %. Reset on start_session.
emg_session_peak = 0.0

# Last emg activation % from a clean frame; reused when a glitch frame is
# suppressed (DROP mode) so the live display holds steady instead of dropping.
last_emg_pct = 0.0

profiler = LatencyProfiler(output_dir=LATENCY_LOG_DIR)

# Session-long buffer for end-of-session per-rep classification.
# Separate from LiveBuffer (which is for the 2 Hz live predictions
# broadcast to Flutter). This one holds every sample during an active
# session for one final pass through predict_session().
session_buffer = []

# Supabase client (set by init_supabase; None if unavailable)
supabase = None


# ============================================================
# ML LOADING (per-exercise, fail-soft)
# ============================================================

def try_load_classifier(ex_id, filename):
    """
    Load a single per-exercise classifier. If loading fails, return (None, None)
    so the bridge can still run with the other models. This is intentional:
    a missing pkl during demo setup shouldn't take the whole bridge down.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"!! [{ex_id}] Model file '{path}' not found — exercise disabled.")
        return None, None

    try:
        from form_classifier import FormClassifier, LiveBuffer
        clf = FormClassifier(path)
        buf = LiveBuffer(clf, max_seconds=10)
        print(f"   [{ex_id}] Classifier loaded from {path}")
        print(f"   [{ex_id}] Classes: {clf.classes}")
        return clf, buf
    except ImportError as e:
        print(f"!! [{ex_id}] form_classifier.py not importable: {e}")
        return None, None
    except Exception as e:
        print(f"!! [{ex_id}] Classifier load failed: {e}")
        return None, None


def try_load_all_classifiers():
    """Load every classifier registered in EXERCISE_MODEL_FILES."""
    loaded_clfs = {}
    loaded_bufs = {}

    print(f"Loading classifiers from '{MODELS_DIR}/' ...")
    for ex_id, filename in EXERCISE_MODEL_FILES.items():
        clf, buf = try_load_classifier(ex_id, filename)
        if clf is not None and buf is not None:
            loaded_clfs[ex_id] = clf
            loaded_bufs[ex_id] = buf

    if not loaded_clfs:
        print("!! WARNING: No classifiers loaded. Bridge will stream sensor "
              "data only, with no form classification.")
    else:
        print(f"Loaded {len(loaded_clfs)}/{len(EXERCISE_MODEL_FILES)} classifiers: "
              f"{list(loaded_clfs.keys())}")

    return loaded_clfs, loaded_bufs


def get_active_buffer():
    """Return the LiveBuffer for the currently active exercise, or None."""
    return buffers.get(current_exercise)


def switch_exercise(new_exercise):
    """
    Update current_exercise and clear that exercise's buffer so the first
    prediction after switching is based on fresh samples only.

    Returns (success: bool, message: str).
    """
    global current_exercise, last_form_label, last_form_confidence

    if new_exercise not in EXERCISE_MODEL_FILES:
        return False, (
            f"Unknown exercise '{new_exercise}'. "
            f"Valid: {list(EXERCISE_MODEL_FILES.keys())}"
        )

    if new_exercise not in classifiers:
        return False, (
            f"Exercise '{new_exercise}' is registered but its model "
            f"failed to load. Check models/ directory."
        )

    if new_exercise == current_exercise:
        buf = buffers.get(new_exercise)
        if buf is not None:
            buf.clear()
        return True, f"Exercise already set to '{new_exercise}'; buffer cleared."

    current_exercise = new_exercise

    buf = buffers.get(new_exercise)
    if buf is not None:
        buf.clear()

    last_form_label = "normal"
    last_form_confidence = 0.0

    return True, f"Exercise switched to '{new_exercise}'."


# ============================================================
# SUPABASE
# ============================================================

def init_supabase():
    """
    Build the Supabase client. Sets the module-level `supabase` global.
    Fail-soft: any error leaves supabase=None and the bridge keeps running
    (it just won't write to the database).
    """
    global supabase
    if not SUPABASE_AVAILABLE:
        supabase = None
        return
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("!! Supabase URL or ANON_KEY not set in .env — logging disabled.")
        print("   Copy .env.example to .env and fill in your credentials.")
        supabase = None
        return
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print(f"Supabase client initialized: {SUPABASE_URL}")
    except Exception as e:
        print(f"!! Supabase init failed: {e}")
        print("   Bridge will run without Supabase logging.")
        supabase = None


def write_session_to_supabase():
    """
    Called at stop_session. Runs predict_session on the full session buffer,
    inserts one summary row into `sessions` and one row per rep into
    `rep_features`. Fail-soft: errors are logged, never raised.

    Returns a summary dict so stop_session can broadcast results to Flutter
    even if Supabase write fails.
    """
    summary = {
        "rep_count": 0,
        "valid_reps": 0,
        "bad_reps": 0,
        "avg_rom_deg": None,
        "max_rom_deg": None,
        "per_rep": [],
    }

    if not session_buffer:
        print("[supabase] session buffer empty; nothing to write")
        return summary

    clf = classifiers.get(current_exercise)
    if clf is None:
        print(f"[supabase] no classifier for '{current_exercise}'; "
              f"skipping rep classification")
        return summary

    # 1. Per-rep classification on the full session buffer.
    try:
        reps = clf.predict_session(session_buffer)
    except Exception as e:
        print(f"[supabase] predict_session failed: {e}")
        return summary

    if not reps:
        print("[supabase] predict_session found 0 reps; nothing to insert")
        return summary

    # 2. Build rep_features rows + accumulate summary stats.
    rep_rows = []
    roms = []
    valid_count = 0
    for r in reps:
        feats = r["features"]
        label = r["label"]
        conf = r["confidence"]
        rom = feats["elbow_angle_range"]
        is_valid = (label == "normal")
        if is_valid:
            valid_count += 1
        roms.append(rom)

        rep_rows.append({
            "session_id": session["session_id"],
            "rep_index": r["rep_num"] - 1,   # 0-indexed
            "rep_start_s": None,
            "rep_end_s": None,
            "rom_deg": rom,
            "normalized_emg": None,          # EMG/HR per-rep aggregation: skipped for DR5
            "heart_rate_bpm": None,
            "is_valid": is_valid,
            "form_label": label,
            "form_confidence": conf,
            "exercise_id": current_exercise,
        })

    summary["rep_count"] = len(reps)
    summary["valid_reps"] = valid_count
    summary["bad_reps"] = len(reps) - valid_count
    summary["avg_rom_deg"] = float(np.mean(roms)) if roms else None
    summary["max_rom_deg"] = float(np.max(roms)) if roms else None
    summary["per_rep"] = [
        {"rep_index": row["rep_index"], "label": row["form_label"],
         "confidence": row["form_confidence"], "rom_deg": row["rom_deg"]}
        for row in rep_rows
    ]

    # 3. Insert rows. Fail-soft: if Supabase is unreachable, still return
    #    the summary so Flutter gets the rep breakdown via the WebSocket.
    if supabase is None:
        print(f"[supabase] client unavailable; computed {len(rep_rows)} reps "
              f"but not inserting")
        return summary

    # 3a. sessions summary row
    start_iso = session.get("start_time")
    end_iso = datetime.now().isoformat()
    duration_s = None
    try:
        if start_iso:
            duration_s = int(
                (datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso))
                .total_seconds()
            )
    except Exception:
        pass

    session_row = {
        "session_id": session["session_id"],
        "session_day": datetime.now().strftime("%Y-%m-%d"),
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_s": duration_s,
        "total_reps": summary["rep_count"],
        "valid_reps": summary["valid_reps"],
        "bad_reps": summary["bad_reps"],
        "target_reps": session.get("target_reps"),
        "calibrated_lower_angle_deg": calibration.get("lower_angle"),
        "calibrated_upper_angle_deg": calibration.get("upper_angle"),
        "avg_rom_deg": summary["avg_rom_deg"],
        "max_rom_deg": summary["max_rom_deg"],
        "rom_consistency": None,        # skip for DR5
        "avg_normalized_emg": None,
        "max_emg": None,
        "avg_heart_rate_bpm": None,
    }

    try:
        supabase.table("sessions").upsert(session_row).execute()
        print(f"[supabase] sessions row upserted "
              f"(session_id={session['session_id']})")
    except Exception as e:
        print(f"[supabase] sessions upsert failed: {e}")
        # don't return — try the rep rows anyway

    # 3b. rep_features rows (batch insert)
    try:
        supabase.table("rep_features").insert(rep_rows).execute()
        print(f"[supabase] inserted {len(rep_rows)} rep_features rows")
    except Exception as e:
        print(f"[supabase] rep_features insert failed: {e}")

    return summary


# ============================================================
# SERIAL PORT
# ============================================================

def find_serial_port():
    ports = list(serial.tools.list_ports.comports())

    print("\nAvailable serial ports:")
    for p in ports:
        print(f"  {p.device} - {p.description}")

    # Skip macOS internal ports that look like serial but aren't useful
    SKIP_KEYWORDS = ["debug-console", "bluetooth"]
    ports = [p for p in ports
             if not any(s in p.device.lower() for s in SKIP_KEYWORDS)]

    for p in ports:
        desc = p.description.lower()
        if (
            "cp210" in desc or "silicon labs" in desc or "usb serial" in desc
            or "adafruit" in desc or "feather" in desc or "nrf" in desc
        ):
            print(f"Auto-selected port: {p.device}")
            return p.device

    if ports:
        print(f"Auto-selected first available port: {ports[0].device}")
        return ports[0].device

    return None


# ============================================================
# ANGLE CALCULATION (display-only; classifier uses its own)
# ============================================================

def pitch_deg(ax, ay, az):
    return math.degrees(math.atan2(ax, math.sqrt((ay * ay) + (az * az))))


def compute_elbow_angle(imu2, imu5):
    p2 = pitch_deg(imu2["ax"], imu2["ay"], imu2["az"])
    p5 = pitch_deg(imu5["ax"], imu5["ay"], imu5["az"])
    angle = abs(p2 - p5)
    return max(0.0, min(180.0, round(angle, 2)))


# ============================================================
# EMG ACTIVATION % (session-peak relative; display-only)
# ============================================================

def emg_activation_pct(raw_emg):
    """
    Update the running session peak with this raw envelope value and return
    the current activation percentage, clamped to [0, 100].

    Session-peak relative: pct = raw / peak * 100. Returns 0.0 until the peak
    rises above EMG_PEAK_FLOOR, which avoids reporting 100% on the first sample
    of a session. This is a RELATIVE effort indicator, not %MVC.

    Mutates the module-level emg_session_peak.
    """
    global emg_session_peak

    if raw_emg > emg_session_peak:
        emg_session_peak = raw_emg

    # Don't divide by a near-noise peak — report 0% until we've seen real effort
    if emg_session_peak < EMG_PEAK_FLOOR:
        return 0.0

    pct = (raw_emg / emg_session_peak) * 100.0
    # Clamp: raw can momentarily equal the peak (-> 100) or, with float noise,
    # nudge just past it before the peak updates; keep it in range.
    return round(max(0.0, min(100.0, pct)), 1)


# ============================================================
# SENSOR GLITCH GUARD
# ============================================================

def frame_is_implausible(imu2, imu5):
    """
    Return (bad: bool, reason: str) for an obviously-corrupt IMU frame.

    Currently checks gyro magnitude on every axis of both IMUs against
    GYRO_SANITY_LIMIT_DPS. This is intentionally simple and conservative:
    it only fires on values that no legitimate rep produces. Extend here if
    bench data shows other reliable signatures (e.g. accel-magnitude far from
    1 g while stationary), but keep the bar high enough to never reject a
    real rep.
    """
    for tag, imu in (("imu2", imu2), ("imu5", imu5)):
        for axis in ("gx", "gy", "gz"):
            if abs(imu[axis]) > GYRO_SANITY_LIMIT_DPS:
                return True, f"{tag}.{axis}={imu[axis]:.2f} dps > {GYRO_SANITY_LIMIT_DPS}"
    return False, ""


# ============================================================
# PARSER (routes to active exercise's classifier)
# ============================================================

def parse_serial_line(line, ctx):
    """
    Parse one DATA line, push samples into the ACTIVE exercise's LiveBuffer
    (and the session_buffer if a session is active), optionally run prediction,
    and return a packet dict (or None).
    """
    global latest_packet, last_predict_t, last_form_label, last_form_confidence
    global last_emg_pct

    parts = line.split(",")

    if len(parts) != 17:
        print(f"[bad DATA packet] expected 17 fields, got {len(parts)}")
        print(line)
        return None

    try:
        arduino_time_ms = float(parts[1])

        imu2 = {
            "ax": float(parts[2]), "ay": float(parts[3]), "az": float(parts[4]),
            "gx": float(parts[5]), "gy": float(parts[6]), "gz": float(parts[7]),
        }
        imu5 = {
            "ax": float(parts[8]), "ay": float(parts[9]), "az": float(parts[10]),
            "gx": float(parts[11]), "gy": float(parts[12]), "gz": float(parts[13]),
        }
        emg = float(parts[14])
        heart_rate = int(float(parts[15]))
        confidence = int(float(parts[16]))

        # ----- Sensor glitch guard -----
        # Evaluate plausibility BEFORE the EMG peak update, because a corrupt
        # frame must not be allowed to raise emg_session_peak (which is
        # monotonic and would skew the rest of the session) when dropping.
        bad_frame, bad_reason = frame_is_implausible(imu2, imu5)
        if bad_frame:
            tag = "DROP" if GLITCH_GUARD_DROP else "detect-only"
            print(f"[glitch:{tag}] arduino_ms={arduino_time_ms:.0f} {bad_reason}")

        # In DROP mode a bad frame skips EMG-peak update, ML push, and the
        # session_buffer append. We still build and broadcast a packet so the
        # Flutter live view does not stall — it reuses last-known form/EMG.
        suppress = bad_frame and GLITCH_GUARD_DROP

        # Convert raw EMG envelope -> session-peak-relative activation %.
        # Raw value is kept in the packet too; this is purely additive.
        # Skipped on a suppressed frame so the peak is never raised by garbage.
        if suppress:
            emg_pct = last_emg_pct
        else:
            emg_pct = emg_activation_pct(emg)
            last_emg_pct = emg_pct

        angle = compute_elbow_angle(imu2, imu5)
        timestamp_sec = round(time.time() - start_time, 3)

        # ----- ML feed (per-exercise) -----
        active_buffer = get_active_buffer()
        if active_buffer is not None and not suppress:
            try:
                active_buffer.push(
                    timestamp_sec,
                    imu2["ax"], imu2["ay"], imu2["az"],
                    imu2["gx"], imu2["gy"], imu2["gz"],
                    imu5["ax"], imu5["ay"], imu5["az"],
                    imu5["gx"], imu5["gy"], imu5["gz"],
                )
            except Exception as e:
                print(f"[live_buffer.push error] {e}")

            # Also accumulate every sample during an active session, so
            # stop_session can run predict_session() on the full thing.
            if session["active"]:
                session_buffer.append({
                    "t": timestamp_sec,
                    "ax_u": imu2["ax"], "ay_u": imu2["ay"], "az_u": imu2["az"],
                    "gx_u": imu2["gx"], "gy_u": imu2["gy"], "gz_u": imu2["gz"],
                    "ax_f": imu5["ax"], "ay_f": imu5["ay"], "az_f": imu5["az"],
                    "gx_f": imu5["gx"], "gy_f": imu5["gy"], "gz_f": imu5["gz"],
                })

            now = time.time()
            if (now - last_predict_t) >= PREDICT_INTERVAL_SEC:
                last_predict_t = now
                try:
                    result = active_buffer.predict_latest()
                    if result:
                        label, conf, _ = result
                        last_form_label = label
                        last_form_confidence = float(conf)
                        print(f"[ml] exercise={current_exercise} "
                              f"form={label} conf={conf:.2f}")
                except Exception as e:
                    print(f"[predict error] {e}")
                profiler.mark_inferred(ctx)

        packet = {
            "event": "live",
            "mode": mode,
            "exercise": current_exercise,
            "angle": angle, "angleDeg": angle, "elbow_angle": angle,
            "emg": emg, "emgValue": emg,
            "emg_activation_pct": emg_pct, "emgActivationPct": emg_pct,
            "heartRate": heart_rate, "heartRateBpm": heart_rate,
            "ppgConfidence": confidence,
            "timestamp": timestamp_sec, "timestampSec": timestamp_sec,
            "arduinoTimeMs": arduino_time_ms,
            "imu2": imu2, "imu5": imu5,
            "session_id": session["session_id"],
            "rep_count": session["rep_count"],
            "calibration": calibration,
            "form_label": last_form_label,
            "form_confidence": last_form_confidence,
        }

        latest_packet = packet
        return packet

    except Exception as e:
        print("[parse error]", e)
        print(line)
        return None


# ============================================================
# WEBSOCKET BROADCAST
# ============================================================

async def broadcast(packet):
    if not clients:
        return
    payload = json.dumps(packet)
    dead = set()
    for ws in clients:
        try:
            await ws.send(payload)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


async def send_status(message, extra=None):
    payload = {"event": "status", "message": message}
    if extra:
        payload.update(extra)
    await broadcast(payload)


# ============================================================
# COMMAND HANDLING
# ============================================================

async def handle_command(message):
    global mode, emg_session_peak, last_emg_pct

    try:
        data = json.loads(message)
    except Exception:
        print("[bad flutter message]", message)
        return

    command = data.get("command")
    print("[command]", command, data)

    if command == "ping":
        await send_status("pong")
        return

    # ----- EXERCISE SWITCHING -----
    if command == "set_exercise":
        new_exercise = data.get("exercise")
        if not isinstance(new_exercise, str):
            await send_status(
                "set_exercise failed: 'exercise' field missing or not a string"
            )
            return
        ok, msg = switch_exercise(new_exercise)
        await send_status(msg, {
            "event_kind": "exercise_changed" if ok else "exercise_change_failed",
            "exercise": current_exercise,
            "ok": ok,
        })
        print(f"[exercise] {msg}")
        return

    # ----- LATENCY PROFILER COMMANDS -----
    if command == "start_trial":
        trial_index = int(data.get("trial_index", 0))
        profiler.start_trial(trial_index)
        await send_status(f"Trial {trial_index} started",
                          {"trial_index": trial_index})
        return

    if command == "end_trial":
        path = profiler.end_trial()
        await send_status("Trial ended", {"csv_path": path})
        return

    if command == "report_latency":
        profiler.report()
        await send_status("Latency report printed to bridge console")
        return

    if command == "write_summary":
        path = profiler.write_summary()
        await send_status("Session summary written",
                          {"summary_path": path})
        return

    # ----- CALIBRATION / SESSION COMMANDS -----
    if command == "calibrate_extended":
        angle = float(latest_packet.get("angle", 0.0))
        calibration["extended_angle"] = angle
        calibration["lower_angle"] = angle
        mode = "CALIBRATING"
        print(f"[calibration] extended/lower = {angle:.2f}")
        await send_status(
            f"Extended angle saved: {angle:.2f}",
            {"calibrated_angle": angle},
        )
        return

    if command == "calibrate_flexed":
        angle = float(latest_packet.get("angle", 0.0))
        calibration["flexed_angle"] = angle
        calibration["upper_angle"] = angle
        mode = "CALIBRATING"
        print(f"[calibration] flexed/upper = {angle:.2f}")
        await send_status(
            f"Flexed angle saved: {angle:.2f}",
            {"calibrated_angle": angle},
        )
        return

    if command == "start_session":
        mode = "SESSION"
        target_reps = int(data.get("target_reps", DEFAULT_TARGET_REPS))
        session["active"] = True
        session["session_id"] = str(uuid.uuid4())
        session["start_time"] = datetime.now().isoformat()
        session["target_reps"] = target_reps
        session["rep_count"] = 0
        session_buffer.clear()   # fresh slate for this session
        emg_session_peak = 0.0   # activation % is relative to THIS session's peak
        last_emg_pct = 0.0       # clear glitch-fallback so it starts fresh too
        print(f"\n========== SESSION STARTED: {session['session_id']} ==========")
        print(f"           exercise: {current_exercise}")
        await broadcast({
            "event": "session_start",
            "mode": mode,
            "exercise": current_exercise,
            "session_id": session["session_id"],
            "start_time": session["start_time"],
            "target_reps": target_reps,
            "calibration": calibration,
        })
        return

    if command == "stop_session":
        mode = "IDLE"
        print("\n========== SESSION STOPPED ==========")

        # Run end-of-session classification and write to Supabase.
        # This blocks briefly (one predict_session call + two HTTP requests);
        # for a 30-sec session it's well under a second on a laptop.
        try:
            summary = write_session_to_supabase()
        except Exception as e:
            print(f"[stop_session] summary write crashed unexpectedly: {e}")
            summary = {
                "rep_count": session["rep_count"],
                "valid_reps": 0, "bad_reps": 0,
                "avg_rom_deg": None, "max_rom_deg": None, "per_rep": [],
            }

        await broadcast({
            "event": "session_complete",
            "mode": mode,
            "exercise": current_exercise,
            "session_id": session["session_id"],
            "total_reps": summary["rep_count"],
            "valid_reps": summary["valid_reps"],
            "bad_reps": summary["bad_reps"],
            "avg_rom_deg": summary["avg_rom_deg"],
            "max_rom_deg": summary["max_rom_deg"],
            "per_rep": summary["per_rep"],
            "end_time": datetime.now().isoformat(),
        })
        session["active"] = False
        session_buffer.clear()   # free memory
        return

    print("[unknown command]", command)


# ============================================================
# SERIAL LOOP
# ============================================================

async def serial_loop():
    global SERIAL_PORT

    while True:
        try:
            if SERIAL_PORT is None:
                SERIAL_PORT = find_serial_port()

            if SERIAL_PORT is None:
                print("No serial port found. Plug in Feather and retrying...")
                await asyncio.sleep(3)
                continue

            print(f"\nOpening {SERIAL_PORT} at {BAUD_RATE} baud...")

            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
                print("Serial connected. Reading Arduino DATA packets...")
                await send_status("Serial connected")

                while True:
                    raw = ser.readline()

                    if not raw:
                        await asyncio.sleep(0.001)
                        continue

                    line = raw.decode("utf-8", errors="ignore") \
                              .strip().replace("\x00", "")

                    if not line:
                        await asyncio.sleep(0)
                        continue

                    # LOOP report from Arduino
                    if line.startswith("LOOP,"):
                        try:
                            parts = line.split(",")
                            avg_us = float(parts[1])
                            profiler.record_arduino_loop(avg_us)
                        except (ValueError, IndexError):
                            pass
                        await asyncio.sleep(0)
                        continue

                    # Comments from Arduino
                    if line.startswith("#"):
                        print("[arduino]", line)
                        await asyncio.sleep(0)
                        continue

                    if not line.startswith("DATA,"):
                        print("[unparsed non-DATA]", line)
                        await asyncio.sleep(0)
                        continue

                    parts_for_ts = line.split(",", 2)
                    arduino_ms = None
                    if len(parts_for_ts) >= 2:
                        try:
                            arduino_ms = float(parts_for_ts[1])
                        except ValueError:
                            arduino_ms = None

                    ctx = profiler.mark_recv(arduino_ms)

                    packet = parse_serial_line(line, ctx)
                    profiler.mark_parsed(ctx)

                    if packet is not None:
                        await broadcast(packet)
                        profiler.mark_broadcast(ctx)

                    await asyncio.sleep(0)

        except serial.SerialException as e:
            print("Serial error:", e)
            SERIAL_PORT = None
            await send_status("Serial disconnected")

        except Exception as e:
            print("Unexpected serial error:", e)
            SERIAL_PORT = None
            await send_status("Serial error")

        print("Reconnecting in 3 seconds...")
        await asyncio.sleep(3)


# ============================================================
# WEBSOCKET HANDLER
# ============================================================

async def flutter_handler(websocket):
    print(f"Flutter connected: {websocket.remote_address}")
    clients.add(websocket)

    # Tell the new client which exercise is currently active and which are
    # available. This lets Flutter sync its UI with the bridge on connect.
    await send_status(
        "Flutter connected",
        {
            "exercise": current_exercise,
            "available_exercises": list(classifiers.keys()),
        },
    )

    try:
        async for message in websocket:
            await handle_command(message)

    except websockets.exceptions.ConnectionClosed:
        pass

    finally:
        clients.discard(websocket)
        print("Flutter disconnected")


# ============================================================
# MAIN
# ============================================================

async def main():
    global classifiers, buffers

    print("=" * 60)
    print("Smart Sleeve Bridge: HYBRID8 (multi-exercise + Supabase)")
    print("=" * 60)
    print(f"Baud rate: {BAUD_RATE}")
    print(f"Models dir: {MODELS_DIR}/")
    print(f"Default exercise: {DEFAULT_EXERCISE}")
    print(f"Latency log dir: {LATENCY_LOG_DIR}/")
    print()

    classifiers, buffers = try_load_all_classifiers()
    init_supabase()

    # If the default exercise's model didn't load, fall back to any loaded one.
    global current_exercise
    if current_exercise not in classifiers and classifiers:
        fallback = next(iter(classifiers.keys()))
        print(f"!! Default exercise '{current_exercise}' not loaded; "
              f"falling back to '{fallback}'.")
        current_exercise = fallback

    print()
    print("=" * 60)

    async with websockets.serve(
        flutter_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
    ):
        print(f"WebSocket server started on "
              f"{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        print(f"  Local:   ws://localhost:{WEBSOCKET_PORT}")
        print("=" * 60)
        print("Commands accepted from Flutter / CLI:")
        print('  {"command": "set_exercise", "exercise": "bicep_curl"}')
        print('  {"command": "set_exercise", "exercise": "hammer_curl"}')
        print('  {"command": "set_exercise", "exercise": "tricep_extension"}')
        print('  {"command": "start_session"}')
        print('  {"command": "stop_session"}')
        print('  {"command": "calibrate_extended"}')
        print('  {"command": "calibrate_flexed"}')
        print('  {"command": "start_trial", "trial_index": 0}')
        print('  {"command": "end_trial"}')
        print('  {"command": "report_latency"}')
        print('  {"command": "write_summary"}')
        print("=" * 60)

        await serial_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWriting final session summary before exit...")
        profiler.write_summary()
