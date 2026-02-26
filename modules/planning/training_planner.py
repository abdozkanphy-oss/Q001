from __future__ import annotations
from utils.identity import NULL_TOKENS

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time
import statistics


@dataclass
class SegmentSummary:
    n_segments: int = 0
    total_points: int = 0
    durations_ms: list = None          # bounded
    points_per_segment: list = None    # bounded

    def __post_init__(self):
        if self.durations_ms is None:
            self.durations_ms = []
        if self.points_per_segment is None:
            self.points_per_segment = []

    def add_segment(self, points: int, dur_ms: int, max_keep: int = 200):
        self.n_segments += 1
        self.total_points += int(points)
        self.durations_ms.append(int(dur_ms))
        self.points_per_segment.append(int(points))
        # Bound memory
        if len(self.durations_ms) > max_keep:
            self.durations_ms = self.durations_ms[-max_keep:]
        if len(self.points_per_segment) > max_keep:
            self.points_per_segment = self.points_per_segment[-max_keep:]

    def median_points(self) -> Optional[float]:
        return statistics.median(self.points_per_segment) if self.points_per_segment else None

    def median_duration_ms(self) -> Optional[float]:
        return statistics.median(self.durations_ms) if self.durations_ms else None


class TrainingPlanner:
    """
    Shadow-only planner:
      - Observes message stream (event-time) per workstation
      - Tracks segment stats for candidate strategies REFNO/JOREF/SESSION
      - Periodically logs a recommendation (does not enforce anything)
    """

    def __init__(
        self,
        session_gap_sec: int = 900,
        min_segments: int = 5,
        min_points_per_segment: int = 200,
        log_every_sec: int = 60,
        keep_last_n: int = 200,
    ):
        self.session_gap_ms = int(session_gap_sec) * 1000
        self.min_segments = int(min_segments)
        self.min_points_per_segment = int(min_points_per_segment)
        self.log_every_sec = int(log_every_sec)
        self.keep_last_n = int(keep_last_n)
        self._cov: Dict[str, Dict[str, int]] = {}


        # per workstation + strategy stats
        self._stats: Dict[Tuple[str, str], SegmentSummary] = {}

        # per workstation current open segment for each strategy
        self._open: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # session tracking per workstation
        self._last_seen_ts: Dict[str, int] = {}
        self._session_id: Dict[str, int] = {}
        self._session_start_ts: Dict[str, int] = {}

        # log rate limiting
        self._last_log_ts: Dict[str, float] = {}

    @staticmethod
    def _norm_id(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        sl = s.lower().strip()
        if sl in NULL_TOKENS:
            return None
        if "|" in sl:
            parts = [p.strip() for p in re.split(r"\|+", sl) if p.strip() != ""]
            if parts and all(p in NULL_TOKENS for p in parts):
                return None
        return s

    def _cov_inc(self, ws: str, key: str, n: int = 1):
        if ws not in self._cov:
            self._cov[ws] = {"total": 0, "refNo": 0, "joRef": 0, "joOpId": 0}
        self._cov[ws][key] = int(self._cov[ws].get(key, 0)) + int(n)

    def _cov_ratio(self, ws: str, key: str) -> float:
        d = self._cov.get(ws) or {}
        tot = int(d.get("total", 0) or 0)
        if tot <= 0:
            return 0.0
        return float(d.get(key, 0) or 0) / float(tot)


    def _get_stats(self, workstation_uid: str, strategy: str) -> SegmentSummary:
        k = (workstation_uid, strategy)
        if k not in self._stats:
            self._stats[k] = SegmentSummary()
        return self._stats[k]

    def _finalize_open(self, workstation_uid: str, strategy: str, end_ts_ms: int):
        k = (workstation_uid, strategy)
        cur = self._open.get(k)
        if not cur:
            return
        start = int(cur.get("start_ts_ms") or 0)
        points = int(cur.get("points") or 0)
        dur = max(0, int(end_ts_ms) - start) if start and end_ts_ms else 0
        if points > 0:
            self._get_stats(workstation_uid, strategy).add_segment(points, dur, max_keep=self.keep_last_n)
        self._open.pop(k, None)

    def _observe_strategy(self, workstation_uid: str, strategy: str, seg_id: Optional[str], event_ts_ms: int):
        """
        Observe a candidate segmentation strategy:
          - seg_id defines current segment id (None => cannot segment using this strategy for this event)
        """
        k = (workstation_uid, strategy)
        cur = self._open.get(k)

        if seg_id is None:
            # Can't use this strategy for this event; do nothing (shadow).
            return

        if cur is None:
            self._open[k] = {"seg_id": seg_id, "start_ts_ms": event_ts_ms, "points": 1}
            return

        if cur["seg_id"] != seg_id:
            # segment boundary due to id change
            self._finalize_open(workstation_uid, strategy, end_ts_ms=event_ts_ms)
            self._open[k] = {"seg_id": seg_id, "start_ts_ms": event_ts_ms, "points": 1}
            return

        # same segment
        cur["points"] += 1

    def _observe_session(self, workstation_uid: str, event_ts_ms: int):
        last = self._last_seen_ts.get(workstation_uid, 0)
        if workstation_uid not in self._session_id:
            self._session_id[workstation_uid] = 0
            self._session_start_ts[workstation_uid] = event_ts_ms

        if last and event_ts_ms and (event_ts_ms - last) > self.session_gap_ms:
            # session boundary
            self._finalize_open(workstation_uid, "SESSION", end_ts_ms=event_ts_ms)
            self._session_id[workstation_uid] += 1
            self._session_start_ts[workstation_uid] = event_ts_ms

        self._last_seen_ts[workstation_uid] = max(last, event_ts_ms or last)

        sid = self._session_id[workstation_uid]
        self._observe_strategy(workstation_uid, "SESSION", seg_id=f"S{sid}", event_ts_ms=event_ts_ms)

    def recommend(self, workstation_uid: str) -> Tuple[str, str]:
        """
        Score strategies and recommend best viable one (shadow-only).
        Viable = enough segments AND enough median points per segment.
        Otherwise fallback SESSION.
        """
        strategies = ["REFNO", "JOREF", "JOOP", "SESSION"]

        def coverage(strategy: str) -> float:
            if strategy == "REFNO":
                return self._cov_ratio(workstation_uid, "refNo")
            if strategy == "JOREF":
                return self._cov_ratio(workstation_uid, "joRef")
            if strategy == "JOOP":
                return self._cov_ratio(workstation_uid, "joOpId")
            return 1.0  # SESSION always available

        def viability(strategy: str) -> Tuple[bool, str]:
            st = self._get_stats(workstation_uid, strategy)
            med_pts = st.median_points()
            if st.n_segments < self.min_segments:
                return False, f"{strategy}: n_segments={st.n_segments} < {self.min_segments}"
            if med_pts is None or med_pts < self.min_points_per_segment:
                return False, f"{strategy}: median_points={med_pts} < {self.min_points_per_segment}"
            return True, f"{strategy}: n_segments={st.n_segments}, median_points={med_pts}"

        def score(strategy: str) -> float:
            st = self._get_stats(workstation_uid, strategy)
            cov = coverage(strategy)
            seg_norm = min(1.0, float(st.n_segments) / float(self.min_segments or 1))
            med_pts = float(st.median_points() or 0.0)
            med_norm = min(1.0, med_pts / float(self.min_points_per_segment or 1))
            return 0.45 * cov + 0.35 * med_norm + 0.20 * seg_norm

        viable = []
        reasons = {}
        scores = {}
        for s in strategies:
            ok, why = viability(s)
            reasons[s] = why
            scores[s] = score(s)
            if ok:
                viable.append(s)

        if viable:
            best = max(viable, key=lambda s: scores[s])
            return best, f"score={scores[best]:.2f}; {reasons[best]}"
        return "SESSION", f"fallback; score={scores['SESSION']:.2f}; {reasons['SESSION']}"


    def observe(self, message: Dict[str, Any]):
        """
        message is expected to already contain:
          - _workstation_uid
          - _event_ts_ms (or 0)
        """
        workstation_uid = self._norm_id(message.get("_workstation_uid")) or "UNKNOWN_WS"
        event_ts_ms = int(message.get("_event_ts_ms") or 0)

        self._cov_inc(workstation_uid, "total", 1)

        # Candidate ids from message
        ref_no = self._norm_id(message.get("refNo"))
        jo_ref = self._norm_id(message.get("joRef"))
        jo_op  = self._norm_id(message.get("joOpId"))

        if ref_no: self._cov_inc(workstation_uid, "refNo", 1)
        if jo_ref: self._cov_inc(workstation_uid, "joRef", 1)
        if jo_op:  self._cov_inc(workstation_uid, "joOpId", 1)

        # Always track session strategy
        self._observe_session(workstation_uid, event_ts_ms)

        # Track explicit-id strategies when available
        self._observe_strategy(workstation_uid, "REFNO", seg_id=ref_no, event_ts_ms=event_ts_ms)
        self._observe_strategy(workstation_uid, "JOREF", seg_id=jo_ref, event_ts_ms=event_ts_ms)
        self._observe_strategy(workstation_uid, "JOOP", seg_id=jo_op, event_ts_ms=event_ts_ms)


    def _open_points(self, workstation_uid: str, strategy: str) -> int:
        cur = self._open.get((workstation_uid, strategy))
        return int(cur.get("points") or 0) if cur else 0

    def _strategy_snapshot(self, workstation_uid: str, strategy: str) -> Dict[str, Any]:
        st = self._get_stats(workstation_uid, strategy)
        return {
            "n_segments": int(st.n_segments),
            "median_points": st.median_points(),
            "median_duration_ms": st.median_duration_ms(),
            "open_points": self._open_points(workstation_uid, strategy),
            "total_points": int(st.total_points),
        }


    def maybe_log(self, workstation_uid: str, logger):
        now = time.time()
        last = self._last_log_ts.get(workstation_uid, 0.0)
        if (now - last) < self.log_every_sec:
            return
        self._last_log_ts[workstation_uid] = now

        reco, reason = self.recommend(workstation_uid)

        ref = self._strategy_snapshot(workstation_uid, "REFNO")
        jr  = self._strategy_snapshot(workstation_uid, "JOREF")
        ses = self._strategy_snapshot(workstation_uid, "SESSION")
        joop = self._strategy_snapshot(workstation_uid, "JOOP")


        # Compact single-line stats string (keeps logs readable)
        stats_str = (
            f"REFNO(n={ref['n_segments']},med_pts={ref['median_points']},open={ref['open_points']}) "
            f"JOREF(n={jr['n_segments']},med_pts={jr['median_points']},open={jr['open_points']}) "
            f"JOOP(n={joop['n_segments']},med_pts={joop['median_points']},open={joop['open_points']}) "
            f"SESSION(n={ses['n_segments']},med_pts={ses['median_points']},open={ses['open_points']})"
        )
        
        cov_str = (
            f"cov(refNo={self._cov_ratio(workstation_uid,'refNo'):.2f},"
            f"joRef={self._cov_ratio(workstation_uid,'joRef'):.2f},"
            f"joOpId={self._cov_ratio(workstation_uid,'joOpId'):.2f})"
        )


        logger.info(
            f"[M1] planner_reco ws={workstation_uid} reco={reco} reason={reason} {cov_str} stats={stats_str}"
        )


