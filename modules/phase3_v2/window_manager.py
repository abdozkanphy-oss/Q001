"""modules/phase3_v2/window_manager.py

Window manager for Phase3V2 (redesign).

Patch3 introduced finalize events (task/batch/ws partition). Patch4 extends the
window manager to retain in-memory frame buffers and attach them to finalize
events so downstream compute (correlation/FI) can run on window end.

Design notes:
- Frames are appended after transition checks, so a finalize event never
  includes the *current* frame that triggered the transition.
- Buffers are capped to avoid unbounded memory. Capping keeps the most recent
  frames; for very long windows this approximates end-of-window statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from modules.phase3_v2.types import Phase3Frame


@dataclass(frozen=True)
class WindowEvent:
    kind: str  # FINALIZE_TASK | FINALIZE_BATCH | FINALIZE_WS
    reason: str
    ws_uid: str
    stock_key: str

    batch_id: str
    batch_root: str
    phase_id: str

    start_ts_ms: int
    end_ts_ms: int
    n_frames: int

    # Window frames (ordered by arrival/finalization). Tuple to keep immutability.
    frames: Tuple[Phase3Frame, ...] = ()


class _CtxWinState:
    __slots__ = (
        "ws_uid",
        "stock_key",
        "cur_batch_id",
        "cur_batch_root",
        "cur_phase_id",
        "batch_start_ts_ms",
        "phase_start_ts_ms",
        "last_frame_ts_ms",
        "batch_frames",
        "phase_frames",
        "last_seen_wall_ms",
        "batch_buf",
        "phase_buf",
        "ws_buf",
        "ws_part_id",
        "ws_part_start_ts_ms",
    )

    def __init__(self, ws_uid: str, stock_key: str):
        self.ws_uid = ws_uid
        self.stock_key = stock_key

        self.cur_batch_id = ""
        self.cur_batch_root = ""
        self.cur_phase_id = ""

        self.batch_start_ts_ms = 0
        self.phase_start_ts_ms = 0
        self.last_frame_ts_ms = 0

        self.batch_frames = 0
        self.phase_frames = 0

        self.last_seen_wall_ms = int(time.time() * 1000.0)

        self.batch_buf: List[Phase3Frame] = []
        self.phase_buf: List[Phase3Frame] = []
        self.ws_buf: List[Phase3Frame] = []

        self.ws_part_id = 0
        self.ws_part_start_ts_ms = 0


class WindowManager:
    def __init__(
        self,
        *,
        idle_finalize_sec: int = 900,
        ws_partition_sec: int = 86400,
        max_inmem_frames: int = 20000,
    ):
        self.idle_finalize_sec = int(idle_finalize_sec)
        self.ws_partition_sec = int(ws_partition_sec)
        self.max_inmem_frames = int(max(1000, max_inmem_frames))

        self._states: Dict[str, _CtxWinState] = {}

    def _ctx_key(self, ws_uid: str, stock_key: str) -> str:
        return f"{ws_uid}::{stock_key}"

    def _ws_part_id(self, ts_ms: int) -> int:
        step_ms = max(1, int(self.ws_partition_sec)) * 1000
        return int(int(ts_ms) // step_ms)

    def _cap(self, buf: List[Phase3Frame]) -> None:
        if len(buf) <= self.max_inmem_frames:
            return
        # keep the most recent frames
        drop = len(buf) - self.max_inmem_frames
        if drop > 0:
            del buf[:drop]

    def _mk_event(
        self,
        *,
        kind: str,
        reason: str,
        st: _CtxWinState,
        frames: List[Phase3Frame],
        start_ts_ms: int,
        end_ts_ms: int,
        batch_id: Optional[str] = None,
        batch_root: Optional[str] = None,
        phase_id: Optional[str] = None,
    ) -> Optional[WindowEvent]:
        if not frames:
            return None
        return WindowEvent(
            kind=kind,
            reason=reason,
            ws_uid=st.ws_uid,
            stock_key=st.stock_key,
            batch_id=str(batch_id if batch_id is not None else (st.cur_batch_id or "0")),
            batch_root=str(batch_root if batch_root is not None else (st.cur_batch_root or "0")),
            phase_id=str(phase_id if phase_id is not None else (st.cur_phase_id or "0")),
            start_ts_ms=int(start_ts_ms),
            end_ts_ms=int(end_ts_ms),
            n_frames=int(len(frames)),
            frames=tuple(frames),
        )

    def _get_state(self, fr: Phase3Frame) -> _CtxWinState:
        k = self._ctx_key(fr.ws_uid, fr.stock_key)
        st = self._states.get(k)
        if st is None:
            st = _CtxWinState(fr.ws_uid, fr.stock_key)
            # initialize context from first frame
            st.cur_batch_id = fr.batch_id or ""
            st.cur_batch_root = fr.batch_root or ""
            st.cur_phase_id = fr.phase_id or ""
            st.batch_start_ts_ms = int(fr.ts_ms)
            st.phase_start_ts_ms = int(fr.ts_ms)
            st.last_frame_ts_ms = int(fr.ts_ms)

            st.ws_part_id = self._ws_part_id(fr.ts_ms)
            st.ws_part_start_ts_ms = int(fr.ts_ms)

            self._states[k] = st
        return st

    def on_frame(self, fr: Phase3Frame) -> List[WindowEvent]:
        """Process a new frame. Returns 0..N finalize events."""
        events: List[WindowEvent] = []

        st = self._get_state(fr)
        st.last_seen_wall_ms = int(time.time() * 1000.0)

        # 1) workstation partition rollover (time-based)
        part_id = self._ws_part_id(fr.ts_ms)
        if part_id != int(st.ws_part_id):
            if st.ws_buf:
                ev = self._mk_event(
                    kind="FINALIZE_WS",
                    reason="partition_rollover",
                    st=st,
                    frames=st.ws_buf,
                    start_ts_ms=st.ws_part_start_ts_ms or st.ws_buf[0].ts_ms,
                    end_ts_ms=st.ws_buf[-1].ts_ms,
                    batch_id=st.cur_batch_id,
                    batch_root=st.cur_batch_root,
                    phase_id=st.cur_phase_id,
                )
                if ev:
                    events.append(ev)
            st.ws_buf = []
            st.ws_part_id = int(part_id)
            st.ws_part_start_ts_ms = int(fr.ts_ms)

        # 2) context changes (phase, batch)
        batch_changed = bool(st.cur_batch_id and fr.batch_id and st.cur_batch_id != fr.batch_id)
        phase_changed = bool(st.cur_phase_id and fr.phase_id and st.cur_phase_id != fr.phase_id)

        if batch_changed:
            # finalize task under the old batch
            if st.phase_buf:
                ev = self._mk_event(
                    kind="FINALIZE_TASK",
                    reason="batch_change",
                    st=st,
                    frames=st.phase_buf,
                    start_ts_ms=st.phase_start_ts_ms or st.phase_buf[0].ts_ms,
                    end_ts_ms=st.phase_buf[-1].ts_ms,
                    batch_id=st.cur_batch_id,
                    batch_root=st.cur_batch_root,
                    phase_id=st.cur_phase_id,
                )
                if ev:
                    events.append(ev)
            # finalize batch window
            if st.batch_buf:
                ev = self._mk_event(
                    kind="FINALIZE_BATCH",
                    reason="batch_change",
                    st=st,
                    frames=st.batch_buf,
                    start_ts_ms=st.batch_start_ts_ms or st.batch_buf[0].ts_ms,
                    end_ts_ms=st.batch_buf[-1].ts_ms,
                    batch_id=st.cur_batch_id,
                    batch_root=st.cur_batch_root,
                    phase_id=st.cur_phase_id,
                )
                if ev:
                    events.append(ev)

            # reset for the new batch/phase
            st.cur_batch_id = fr.batch_id or ""
            st.cur_batch_root = fr.batch_root or ""
            st.cur_phase_id = fr.phase_id or ""
            st.batch_start_ts_ms = int(fr.ts_ms)
            st.phase_start_ts_ms = int(fr.ts_ms)
            st.batch_frames = 0
            st.phase_frames = 0
            st.batch_buf = []
            st.phase_buf = []

        elif phase_changed:
            # finalize task window only
            if st.phase_buf:
                ev = self._mk_event(
                    kind="FINALIZE_TASK",
                    reason="phase_change",
                    st=st,
                    frames=st.phase_buf,
                    start_ts_ms=st.phase_start_ts_ms or st.phase_buf[0].ts_ms,
                    end_ts_ms=st.phase_buf[-1].ts_ms,
                    batch_id=st.cur_batch_id,
                    batch_root=st.cur_batch_root,
                    phase_id=st.cur_phase_id,
                )
                if ev:
                    events.append(ev)

            st.cur_phase_id = fr.phase_id or ""
            st.phase_start_ts_ms = int(fr.ts_ms)
            st.phase_frames = 0
            st.phase_buf = []

        # 3) append frame to current buffers
        st.last_frame_ts_ms = int(fr.ts_ms)

        st.batch_frames += 1
        st.phase_frames += 1

        st.ws_buf.append(fr)
        st.batch_buf.append(fr)
        st.phase_buf.append(fr)

        self._cap(st.ws_buf)
        self._cap(st.batch_buf)
        self._cap(st.phase_buf)

        return events

    def tick(self) -> List[WindowEvent]:
        """Periodic maintenance.

        Currently only emits finalize events on idle timeouts.
        """
        out: List[WindowEvent] = []

        now_ms = int(time.time() * 1000.0)
        idle_ms = max(1, int(self.idle_finalize_sec)) * 1000

        to_delete: List[str] = []
        for k, st in self._states.items():
            if now_ms - int(st.last_seen_wall_ms) < idle_ms:
                continue

            # finalize everything we have
            if st.phase_buf:
                ev = self._mk_event(
                    kind="FINALIZE_TASK",
                    reason="idle_timeout",
                    st=st,
                    frames=st.phase_buf,
                    start_ts_ms=st.phase_start_ts_ms or st.phase_buf[0].ts_ms,
                    end_ts_ms=st.phase_buf[-1].ts_ms,
                )
                if ev:
                    out.append(ev)

            if st.batch_buf:
                ev = self._mk_event(
                    kind="FINALIZE_BATCH",
                    reason="idle_timeout",
                    st=st,
                    frames=st.batch_buf,
                    start_ts_ms=st.batch_start_ts_ms or st.batch_buf[0].ts_ms,
                    end_ts_ms=st.batch_buf[-1].ts_ms,
                )
                if ev:
                    out.append(ev)

            if st.ws_buf:
                ev = self._mk_event(
                    kind="FINALIZE_WS",
                    reason="idle_timeout",
                    st=st,
                    frames=st.ws_buf,
                    start_ts_ms=st.ws_part_start_ts_ms or st.ws_buf[0].ts_ms,
                    end_ts_ms=st.ws_buf[-1].ts_ms,
                )
                if ev:
                    out.append(ev)

            to_delete.append(k)

        for k in to_delete:
            try:
                del self._states[k]
            except Exception:
                pass

        return out
