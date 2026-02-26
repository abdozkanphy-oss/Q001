from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class FrameFinalizeResult:
    """Result emitted when a time bucket is finalized."""

    bucket_start_epoch_sec: int
    bucket_last_event_epoch_ms: int  # last event time observed within the bucket (ms)
    values: Dict[str, float]  # sensor_id -> value (last seen in bucket)
    sensor_order: List[str]  # stable order (bounded, append-only)
    meta: Dict[str, object]  # last-seen message meta for the bucket (best-effort)
    gap_reset: bool  # whether a reset should be applied at the bucket boundary
    reset_kind: str = ""  # "gap" | "epoch" | ""


class EventTimeBucketizer:
    """Event-time bucketizer with watermark-based lateness handling.

    Key properties:
    - Resamples by event-time into fixed buckets (resample_sec).
    - Maintains a watermark: accepts out-of-order events within allowed lateness,
      and finalizes buckets that fall behind the watermark.
    - Does NOT fabricate frames for empty buckets; only buckets with at least one
      event are emitted.
    - Emits gap_reset=True when there is a large forward gap (gap > max_gap_sec)
      between emitted buckets, or when an epoch reset occurs.

    Why watermark:
    - Mixed-cadence / mixed-order streams (and replays) often include mild disorder.
      A strict 'drop all past bucket updates' policy causes excessive late_drops.
    """

    def __init__(
        self,
        *,
        resample_sec: int,
        max_gap_sec: int,
        allowed_lateness_sec: int,
        max_sensors: int = 64,
        epoch_reset_sec: int = 6 * 3600,
        max_open_buckets: Optional[int] = None,
    ) -> None:
        self.resample_sec = max(1, int(resample_sec))
        self.max_gap_sec = max(1, int(max_gap_sec))
        self.allowed_lateness_sec = max(0, int(allowed_lateness_sec))
        self.max_sensors = max(1, int(max_sensors))

        self.epoch_reset_sec = max(0, int(epoch_reset_sec))

        # Open buckets are bounded by lateness window (+ slack).
        if max_open_buckets is None:
            # ceil(allowed_lateness / resample) + a small slack for boundary conditions
            lateness_buckets = (self.allowed_lateness_sec + self.resample_sec - 1) // self.resample_sec
            max_open_buckets = int(max(4, lateness_buckets + 4))
        self.max_open_buckets = max(1, int(max_open_buckets))

        self._sensor_order: List[str] = []
        self._sensor_index: Dict[str, int] = {}

        # bucket_start_epoch_sec -> last seen values
        self._open_vals: Dict[int, Dict[str, float]] = {}
        self._open_last_ms: Dict[int, int] = {}
        self._open_meta: Dict[int, Dict[str, object]] = {}

        # Maxima for watermark computation
        self._max_bucket: Optional[int] = None
        self._max_event_ts_epoch: Optional[float] = None

        # Ready-to-emit finalized frames (ordered ascending by bucket start)
        self._ready: Deque[FrameFinalizeResult] = deque()

        # For gap_reset detection across emitted buckets
        self._last_emitted_bucket: Optional[int] = None
        self._force_gap_reset_next: bool = False

    @property
    def sensor_order(self) -> List[str]:
        return list(self._sensor_order)

    def reset(self) -> None:
        """Hard reset of bucketizer state (keeps sensor order)."""
        self._open_vals.clear()
        self._open_last_ms.clear()
        self._open_meta.clear()
        self._max_bucket = None
        self._max_event_ts_epoch = None
        self._ready.clear()
        self._last_emitted_bucket = None
        self._force_gap_reset_next = True

    def _bucket_start(self, event_ts_epoch: float) -> int:
        s = int(event_ts_epoch)
        return (s // self.resample_sec) * self.resample_sec

    def _track_sensor(self, sid: str) -> None:
        if sid in self._sensor_index:
            return
        if len(self._sensor_order) >= self.max_sensors:
            return
        self._sensor_index[sid] = len(self._sensor_order)
        self._sensor_order.append(sid)

    def _lateness_buckets(self) -> int:
        if self.allowed_lateness_sec <= 0:
            return 0
        return int((self.allowed_lateness_sec + self.resample_sec - 1) // self.resample_sec)

    def _watermark_bucket(self) -> Optional[int]:
        if self._max_bucket is None:
            return None
        lb = self._lateness_buckets()
        return int(self._max_bucket - lb * self.resample_sec)

    def _emit_finalize(self, bucket_start: int, vals: Dict[str, float], last_ms: int, meta: Optional[Dict[str, object]] = None, *, reset_kind: str = "") -> None:
        if not vals:
            return

        gap_reset = False
        if self._force_gap_reset_next:
            gap_reset = True
            if not reset_kind:
                reset_kind = "epoch"
            self._force_gap_reset_next = False
        elif self._last_emitted_bucket is not None:
            gap_sec = int(bucket_start - int(self._last_emitted_bucket))
            if gap_sec > int(self.max_gap_sec):
                gap_reset = True
                if not reset_kind:
                    reset_kind = "gap"

        if str(reset_kind or "") == "epoch":
            gap_reset = True
        self._ready.append(
            FrameFinalizeResult(
                bucket_start_epoch_sec=int(bucket_start),
                bucket_last_event_epoch_ms=int(last_ms),
                values=dict(vals),
                sensor_order=list(self._sensor_order),
                meta=dict(meta or {}),
                gap_reset=bool(gap_reset),
                reset_kind=str(reset_kind or ""),
            )
        )
        self._last_emitted_bucket = int(bucket_start)

    def ingest(
        self,
        *,
        event_ts_epoch: float,
        values: Dict[str, float],
        meta: Optional[Dict[str, object]] = None,
    ) -> Tuple[Optional[FrameFinalizeResult], bool]:
        """Ingest a message. Returns (finalized_frame, late_dropped).

        late_dropped=True means the message was ignored because it fell behind the watermark
        (i.e., beyond allowed lateness).
        """
        if not values:
            return (None, False)

        b = self._bucket_start(float(event_ts_epoch))

        # Epoch reset: large backwards jump in event time for this key.
        if (
            self.epoch_reset_sec > 0
            and self._max_event_ts_epoch is not None
            and float(event_ts_epoch) < float(self._max_event_ts_epoch) - float(self.epoch_reset_sec)
        ):
            # Flush remaining open buckets (ascending). Mark the last flushed bucket to reset state.
            open_keys = sorted(self._open_vals.keys())
            if open_keys:
                for i, k in enumerate(open_keys):
                    vals = self._open_vals.get(k) or {}
                    last_ms = int(self._open_last_ms.get(k) or (k * 1000))
                    is_last = (i == (len(open_keys) - 1))
                    self._emit_finalize(int(k), vals, last_ms, self._open_meta.get(int(k)) or {}, reset_kind=("epoch" if is_last else ""))
            # Reset internal bucket state for new epoch.
            self._open_vals.clear()
            self._open_last_ms.clear()
            self._open_meta.clear()
            self._max_bucket = None
            self._max_event_ts_epoch = None
            # Ensure next emitted bucket forces a reset even if there was nothing to flush.
            self._force_gap_reset_next = True

        # Update maxima
        if self._max_event_ts_epoch is None or float(event_ts_epoch) > float(self._max_event_ts_epoch):
            self._max_event_ts_epoch = float(event_ts_epoch)

        if self._max_bucket is None or int(b) > int(self._max_bucket):
            self._max_bucket = int(b)

        wm = self._watermark_bucket()
        if wm is not None and int(b) < int(wm):
            # Too late (beyond allowed lateness) -> drop.
            return (None, True)

        # Update the target bucket state
        if b not in self._open_vals:
            self._open_vals[int(b)] = {}
            self._open_meta[int(b)] = dict(meta or {})
        # last event time in ms for this bucket
        try:
            cur_ms = int(float(event_ts_epoch) * 1000.0)
        except Exception:
            cur_ms = int(time.time() * 1000.0)
        prev_ms = int(self._open_last_ms.get(int(b)) or 0)
        if meta is not None:
            # last-wins meta within bucket
            self._open_meta[int(b)] = dict(meta)
        if cur_ms > prev_ms:
            self._open_last_ms[int(b)] = int(cur_ms)

        for sid, v in values.items():
            if sid is None:
                continue
            sid = str(sid).strip()
            if not sid:
                continue
            self._track_sensor(sid)
            if sid in self._sensor_index:
                self._open_vals[int(b)][sid] = float(v)

        # Finalize buckets that are behind watermark
        wm2 = self._watermark_bucket()
        if wm2 is not None:
            finalize_keys = [k for k in self._open_vals.keys() if int(k) < int(wm2)]
            for k in sorted(finalize_keys):
                vals = self._open_vals.pop(int(k), None) or {}
                last_ms = int(self._open_last_ms.pop(int(k), None) or (int(k) * 1000))
                meta_k = self._open_meta.pop(int(k), None) or {}
                self._emit_finalize(int(k), vals, last_ms, meta_k, reset_kind="")

        # Bound open buckets: if exceeded, finalize the oldest observed bucket to prevent unbounded growth.
        if len(self._open_vals) > int(self.max_open_buckets):
            k = int(sorted(self._open_vals.keys())[0])
            vals = self._open_vals.pop(k, None) or {}
            last_ms = int(self._open_last_ms.pop(k, None) or (k * 1000))
            meta_k = self._open_meta.pop(k, None) or {}
            self._emit_finalize(k, vals, last_ms, meta_k, reset_kind="")

        if self._ready:
            return (self._ready.popleft(), False)

        return (None, False)