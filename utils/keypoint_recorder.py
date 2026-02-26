from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from utils.config_reader import ConfigReader
from utils.logger_2 import setup_logger


@dataclass
class _Obs:
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def add(self, x: float) -> None:
        self.count += 1
        self.sum += float(x)
        if x < self.min:
            self.min = float(x)
        if x > self.max:
            self.max = float(x)

    def to_dict(self) -> dict:
        if self.count <= 0:
            return {"count": 0, "sum": 0.0, "min": None, "max": None, "avg": None}
        return {
            "count": int(self.count),
            "sum": float(self.sum),
            "min": (None if self.min == float("inf") else float(self.min)),
            "max": (None if self.max == float("-inf") else float(self.max)),
            "avg": float(self.sum) / float(self.count),
        }


class KeypointRecorder:
    """Thread-safe lightweight metrics recorder.

    Design goals:
    - No external deps
    - Low overhead
    - Coarse-grained counters + gauges + simple observations
    - Safe to call from hot paths (best effort; never raises)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._obs: Dict[str, _Obs] = defaultdict(_Obs)

    def inc(self, k: str, n: int = 1) -> None:
        try:
            if not k:
                return
            with self._lock:
                self._counters[str(k)] += int(n)
        except Exception:
            return

    def set_gauge(self, k: str, v: float) -> None:
        try:
            if not k:
                return
            with self._lock:
                self._gauges[str(k)] = float(v)
        except Exception:
            return

    def observe(self, k: str, v: float) -> None:
        try:
            if not k:
                return
            with self._lock:
                self._obs[str(k)].add(float(v))
        except Exception:
            return

    @contextmanager
    def timeit(self, k: str):
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            self.observe(k, dt)

    def snapshot(self) -> Tuple[dict, dict, dict]:
        """Returns (counters, gauges, observations) as plain dict copies."""
        with self._lock:
            c = dict(self._counters)
            g = dict(self._gauges)
            o = {k: v.to_dict() for k, v in self._obs.items()}
        return c, g, o


# Global singleton (safe for import)
KP = KeypointRecorder()


def execute_keypoint_reporter() -> None:
    """Periodic logger for KeypointRecorder.

    Emits deltas + gauges and a compact observations summary.
    Can be disabled/configured via config.json.

    Config keys:
      - kp_enabled (bool, default True)
      - kp_log_every_sec (int, default 30)
      - kp_log_reset_on_emit (bool, default False)  (if True, deltas become totals)
      - kp_log_include_queue_depth (bool, default True)
      - kp_log_include_trigger_stats (bool, default True)
    """

    cfg = ConfigReader()

    enabled = bool(getattr(cfg, "kp_enabled", True))
    every = int(getattr(cfg, "kp_log_every_sec", 30) or 30)
    reset = bool(getattr(cfg, "kp_log_reset_on_emit", False))
    include_q = bool(getattr(cfg, "kp_log_include_queue_depth", True))
    include_trig = bool(getattr(cfg, "kp_log_include_trigger_stats", True))

    log = setup_logger("keypoints", "logs/keypoints.log")

    if not enabled or every <= 0:
        log.info("[keypoints] disabled")
        while True:
            time.sleep(60.0)

    last_ts = time.time()
    last_counters = {}

    log.info(f"[keypoints] started every={every}s reset={reset} include_q={include_q} include_trig={include_trig}")

    while True:
        time.sleep(float(every))
        now = time.time()

        c, g, o = KP.snapshot()

        # Optional external gauges/stats (best effort)
        extra = {}
        if include_q:
            try:
                from runtime.event_bus import qsize
                q2, q3 = qsize()
                extra["queue.phase2_depth"] = int(q2)
                extra["queue.phase3_depth"] = int(q3)
            except Exception:
                pass

        if include_trig:
            try:
                from thread.phase_trigger_bus import stats
                extra["trigger_bus"] = stats(now)
            except Exception:
                pass

        # delta counters
        delta = {}
        for k, v in c.items():
            pv = int(last_counters.get(k, 0))
            delta[k] = int(v) - pv

        dt = max(1e-6, now - last_ts)

        payload = {
            "ts_epoch": round(now, 3),
            "dt_sec": round(dt, 3),
            "counters_delta": delta,
            "counters_total": (c if not reset else {}),
            "gauges": {**g, **({k: v for k, v in extra.items() if not isinstance(v, dict)})},
            "observations": o,
        }
        # attach dict extras separately (avoid mixing gauge types)
        for k, v in extra.items():
            if isinstance(v, dict):
                payload[k] = v

        log.info("[keypoints] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))

        last_ts = now
        last_counters = c

        if reset:
            # Soft-reset by updating last_counters only; KP keeps totals (intentional)
            # Full reset is risky in multi-threaded setup; omit by design.
            pass
