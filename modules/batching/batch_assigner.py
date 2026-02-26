from __future__ import annotations
from utils.identity import NULL_TOKENS, get_workstation_uid

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


def _normalize_id(v: Any) -> Optional[str]:
    """
    Normalizes identifier fields.
    Treats placeholders like 'null||null' / 'null|null' / '0||0' as missing.
    """
    if v is None:
        return None

    s = str(v).strip()
    if not s:
        return None

    sl = s.lower().strip()
    if sl in NULL_TOKENS:
        return None

    # Composite placeholders, e.g. "null||null", "null|null", "0||0"
    if "|" in sl:
        parts = [p.strip() for p in re.split(r"\|+", sl) if p.strip() != ""]
        if parts and all(p in NULL_TOKENS for p in parts):
            return None

    return s


def _parse_epoch_ms(v: Any) -> Optional[int]:
    """
    Accepts:
      - epoch ms / sec (int/float or numeric string)
      - ISO datetime string (with/without Z)
      - datetime
    Returns ms since epoch or None.
    """
    if v is None:
        return None

    if isinstance(v, datetime):
        dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    # numeric epoch
    try:
        iv = int(float(str(v).strip()))
        if iv > 10**12:  # ms
            return iv
        if iv > 10**9:   # sec
            return iv * 1000
    except Exception:
        pass

    # ISO string
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def extract_event_ts_ms(message: Dict[str, Any]) -> int:
    """Event-time anchor (ms since epoch).

    Project policy (v2.25+): prefer **crDt** as the canonical event time (user decision).
    Fallbacks are kept for robustness when crDt is missing.
    """
    # 1) Prefer message-level crDt (can be epoch ms/sec, ISO, or datetime-like)
    t = _parse_epoch_ms(message.get("crDt"))
    if t:
        return t

    # 2) Next: outVals measDt (mixed-cadence messages may carry multiple measDt values)
    out_vals = message.get("outVals")
    if isinstance(out_vals, list) and out_vals:
        ts = []
        for ov in out_vals:
            if isinstance(ov, dict):
                tt = _parse_epoch_ms(ov.get("measDt"))
                if tt:
                    ts.append(tt)
        if ts:
            return max(ts)

    # 3) Fallback: other message-level timestamps
    for k in ("measDt", "ts", "timestamp"):
        tt = _parse_epoch_ms(message.get(k))
        if tt:
            return tt

    return 0

def extract_customer_uid(message: Dict[str, Any]) -> str:
    """
    Derive a stable customer/site namespace.

    IMPORTANT: do NOT rely on outVals[*].cust here; it is not a stable customer identity in your data.
    """
    for k in ("plId", "plNm", "orgId", "orgNo", "orgNm", "customer", "cust"):
        v = _normalize_id(message.get(k))
        if v:
            return v
    return "UNKNOWN_CUSTOMER"


def extract_workstation_id(message: Dict[str, Any]) -> str:
    """
    Workstation identity must be stable and NOT collapse multiple WS under the same WC.

    Prefer (wcId, wsId) together; fall back gracefully.
    """
    wc = _normalize_id(message.get("work_center_id") or message.get("wcId") or message.get("wcNo") or message.get("wcNm"))
    ws = _normalize_id(message.get("work_station_id") or message.get("wsId") or message.get("wsNo") or message.get("wsNm"))

    if wc and ws:
        return f"WC{wc}:WS{ws}"
    if ws:
        return f"WS{ws}"
    if wc:
        return f"WC{wc}"

    return "UNKNOWN_WS"



@dataclass(frozen=True)
class BatchContext:
    workstation_uid: str
    batch_id: str
    strategy: str
    confidence: float
    reason: str
    phase_id: Optional[str]
    event_ts_ms: int


class BatchAssigner:
    """
    Workstation-first batching with safe fallbacks.

    Strategies (initial):
      - REFNO: refNo present
      - JOREF: joRef present
      - SESSION: event-time inactivity session fallback
    """

    def __init__(
        self,
        session_gap_sec: int = 900,
        explicit_gap_split_sec: int = 3600,
        enable_hybrid_split: bool = False,
    ):
        self.session_gap_ms = int(session_gap_sec) * 1000
        self.explicit_gap_split_ms = int(explicit_gap_split_sec) * 1000
        self.enable_hybrid_split = bool(enable_hybrid_split)

        # state per workstation_uid
        self._last_max_event_ts_ms: Dict[str, int] = {}
        self._session_seq: Dict[str, int] = {}
        self._session_start_ms: Dict[str, int] = {}

        # optional: split explicit ids on long gaps
        self._explicit_subseq: Dict[Tuple[str, str, str], int] = {}

    def assign(self, message: Dict[str, Any]) -> BatchContext:
        # Canonical identity for batching state. MUST match Stage0/Phase2/Phase3.
        workstation_uid = get_workstation_uid(message)


        event_ts_ms = extract_event_ts_ms(message)

        # Protect against out-of-order: track max observed event_ts per workstation
        last_max = self._last_max_event_ts_ms.get(workstation_uid, 0)
        observed_ts = max(event_ts_ms, last_max)
        self._last_max_event_ts_ms[workstation_uid] = observed_ts

        # Sessioning fallback (and hybrid split trigger)
        gap_ms = observed_ts - last_max if last_max else 0
        new_session = (last_max > 0 and gap_ms > self.session_gap_ms)

        if workstation_uid not in self._session_seq:
            self._session_seq[workstation_uid] = 0
            self._session_start_ms[workstation_uid] = observed_ts or 0
        elif new_session:
            self._session_seq[workstation_uid] += 1
            self._session_start_ms[workstation_uid] = observed_ts or self._session_start_ms[workstation_uid]

        ref_no = _normalize_id(message.get("refNo"))
        jo_ref = _normalize_id(message.get("joRef"))
        phase_id = _normalize_id(message.get("joOpId"))

        if ref_no:
            strategy = "REFNO"
            root = ref_no
            confidence = 0.95
            reason = "refNo present"
        elif jo_ref:
            strategy = "JOREF"
            root = jo_ref
            confidence = 0.90
            reason = "joRef present"
        else:
            strategy = "SESSION"
            root = str(self._session_start_ms.get(workstation_uid, 0))
            confidence = 0.70
            reason = "fallback sessioning (missing refNo/joRef)"

        # Optionally split explicit ids on long inactivity gaps (hybrid)
        suffix = ""
        if self.enable_hybrid_split and strategy in ("REFNO", "JOREF") and last_max and gap_ms > self.explicit_gap_split_ms:
            key = (workstation_uid, strategy, root)
            self._explicit_subseq[key] = self._explicit_subseq.get(key, 0) + 1
            suffix = f":S{self._explicit_subseq[key]}"
            reason += f"; split_on_gap={gap_ms}ms"

        if strategy == "SESSION":
            batch_id = f"SESSION:{root}"
        else:
            batch_id = f"{strategy}:{root}{suffix}"

        return BatchContext(
            workstation_uid=workstation_uid,
            batch_id=batch_id,
            strategy=strategy,
            confidence=confidence,
            reason=reason,
            phase_id=phase_id,
            event_ts_ms=event_ts_ms,
        )
