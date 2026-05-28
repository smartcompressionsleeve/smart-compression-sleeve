"""
Smart Sleeve Latency Profiler
------------------------------
Per-segment latency instrumentation for the Arduino -> Bridge -> WebSocket pipeline.

Segments tracked:
    A. Arduino loop time          (reported by Arduino LOOP,<us>,<count> messages)
    B. Arduino -> Bridge transit  (inter-arrival jitter; Arduino millis vs bridge perf_counter)
    C. Bridge parse + ML feed     (raw line received -> parsed dict)
    D. Bridge inference           (live_buffer.predict_latest, only when triggered)
    E. Bridge -> WebSocket        (parsed -> broadcast complete)

Trial-aware: each trial gets its own CSV file. Use start_trial() / end_trial()
to bracket recording. Outside a trial, samples still go to the cumulative
SegmentSamples buffers but no per-sample CSV row is logged.
"""

import csv
import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SegmentSamples:
    name: str
    samples: deque = field(default_factory=lambda: deque(maxlen=100000))

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def stats(self) -> dict:
        if not self.samples:
            return {"name": self.name, "n": 0}
        s = sorted(self.samples)
        n = len(s)
        return {
            "name": self.name,
            "n": n,
            "min_ms": round(s[0], 3),
            "p50_ms": round(s[n // 2], 3),
            "p95_ms": round(s[min(n - 1, int(n * 0.95))], 3),
            "p99_ms": round(s[min(n - 1, int(n * 0.99))], 3),
            "max_ms": round(s[-1], 3),
            "mean_ms": round(statistics.mean(s), 3),
            "stdev_ms": round(statistics.stdev(s), 3) if n > 1 else 0.0,
        }


class LatencyProfiler:
    def __init__(self, output_dir: str = "latency_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = time.strftime("%Y%m%d_%H%M%S")

        self.seg_arduino_loop = SegmentSamples("A_arduino_loop_us")
        self.seg_arrival_jitter = SegmentSamples("B_arrival_jitter_ms")
        self.seg_parse = SegmentSamples("C_bridge_parse_ms")
        self.seg_inference = SegmentSamples("D_inference_ms")
        self.seg_broadcast = SegmentSamples("E_broadcast_ms")

        self._current_trial: Optional[int] = None
        self._current_trial_log: list = []

        self._prev_arduino_ms: Optional[float] = None
        self._prev_bridge_t: Optional[float] = None

        self._last_report_t = time.perf_counter()
        self.report_interval_s = 5.0

    # ---------- TRIAL CONTROL ----------

    def start_trial(self, trial_index: int) -> None:
        self._current_trial = trial_index
        self._current_trial_log = []
        self._prev_arduino_ms = None
        self._prev_bridge_t = None
        print(f"\n[profiler] === TRIAL {trial_index} STARTED ===")

    def end_trial(self) -> Optional[str]:
        if self._current_trial is None:
            return None
        trial_idx = self._current_trial
        path = self.output_dir / f"trial_{trial_idx:02d}_{self.session_id}.csv"
        if self._current_trial_log:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._current_trial_log[0].keys()))
                writer.writeheader()
                writer.writerows(self._current_trial_log)
            print(f"[profiler] === TRIAL {trial_idx} ENDED — {len(self._current_trial_log)} samples → {path.name}")
        else:
            print(f"[profiler] === TRIAL {trial_idx} ENDED — no samples logged")
        result = str(path) if self._current_trial_log else None
        self._current_trial = None
        return result

    # ---------- ARDUINO STAGE ----------

    def record_arduino_loop(self, micros_per_loop: float) -> None:
        self.seg_arduino_loop.add(micros_per_loop)

    # ---------- BRIDGE STAGES ----------

    def mark_recv(self, arduino_millis: Optional[float] = None) -> dict:
        t_recv = time.perf_counter()

        if (
            arduino_millis is not None
            and self._prev_arduino_ms is not None
            and self._prev_bridge_t is not None
        ):
            arduino_delta = arduino_millis - self._prev_arduino_ms
            bridge_delta = (t_recv - self._prev_bridge_t) * 1000.0
            jitter = bridge_delta - arduino_delta
            self.seg_arrival_jitter.add(jitter)

        if arduino_millis is not None:
            self._prev_arduino_ms = arduino_millis
            self._prev_bridge_t = t_recv

        return {
            "t_recv": t_recv,
            "arduino_ms": arduino_millis,
            "trial": self._current_trial,
        }

    def mark_parsed(self, ctx: dict) -> None:
        t_parsed = time.perf_counter()
        ctx["t_parsed"] = t_parsed
        self.seg_parse.add((t_parsed - ctx["t_recv"]) * 1000.0)

    def mark_inferred(self, ctx: dict) -> None:
        t_inf = time.perf_counter()
        ctx["t_inferred"] = t_inf
        anchor = ctx.get("t_parsed", ctx["t_recv"])
        self.seg_inference.add((t_inf - anchor) * 1000.0)

    def mark_broadcast(self, ctx: dict) -> None:
        t_bc = time.perf_counter()
        ctx["t_broadcast"] = t_bc
        anchor = ctx.get("t_inferred") or ctx.get("t_parsed") or ctx["t_recv"]
        self.seg_broadcast.add((t_bc - anchor) * 1000.0)

        if self._current_trial is not None:
            self._current_trial_log.append({
                "trial": self._current_trial,
                "arduino_ms": ctx.get("arduino_ms"),
                "t_recv_perf": ctx["t_recv"],
                "parse_ms": (ctx.get("t_parsed", ctx["t_recv"]) - ctx["t_recv"]) * 1000.0,
                "inference_ms": ((ctx.get("t_inferred") or ctx.get("t_parsed") or ctx["t_recv"])
                                 - (ctx.get("t_parsed") or ctx["t_recv"])) * 1000.0,
                "broadcast_ms": (ctx["t_broadcast"]
                                 - (ctx.get("t_inferred") or ctx.get("t_parsed") or ctx["t_recv"])) * 1000.0,
                "total_bridge_ms": (ctx["t_broadcast"] - ctx["t_recv"]) * 1000.0,
            })

        if t_bc - self._last_report_t >= self.report_interval_s:
            self.report()
            self._last_report_t = t_bc

    # ---------- REPORTING ----------

    def report(self) -> None:
        trial_str = f"trial {self._current_trial}" if self._current_trial is not None else "no active trial"
        print("\n" + "=" * 78)
        print(f"LATENCY REPORT (session {self.session_id}, {trial_str})")
        print("=" * 78)
        for seg in [self.seg_arduino_loop, self.seg_arrival_jitter,
                    self.seg_parse, self.seg_inference, self.seg_broadcast]:
            stats = seg.stats()
            if stats["n"] == 0:
                print(f"  {stats['name']:<28} (no samples)")
                continue
            print(f"  {stats['name']:<28} "
                  f"n={stats['n']:>5}  "
                  f"p50={stats['p50_ms']:>7.2f}  "
                  f"p95={stats['p95_ms']:>7.2f}  "
                  f"p99={stats['p99_ms']:>7.2f}  "
                  f"max={stats['max_ms']:>7.2f}")
        print("=" * 78 + "\n")

    def write_summary(self) -> str:
        path = self.output_dir / f"session_summary_{self.session_id}.json"
        summary = {
            "session_id": self.session_id,
            "segments": [
                self.seg_arduino_loop.stats(),
                self.seg_arrival_jitter.stats(),
                self.seg_parse.stats(),
                self.seg_inference.stats(),
                self.seg_broadcast.stats(),
            ],
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[profiler] wrote session summary → {path.name}")
        return str(path)
