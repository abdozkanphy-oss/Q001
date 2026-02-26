import threading
import time
from typing import Optional, Tuple

from utils.keypoint_recorder import KP

_LOCK = threading.Lock()

# key -> (expires_epoch, reason)
_ANOM = {}

# key -> last_baseline_epoch
_LAST_BASELINE = {}

# Back-compat aliases (some helpers/worker code may refer to these names)
_TRIGGERS = _ANOM
_BASELINES = _LAST_BASELINE


def _k(ws_uid: str, stock_key: str, scope: str) -> str:
    stock_key = stock_key or "ALL"
    scope = scope or "ws"
    return f"{ws_uid}|{stock_key}|{scope}"

def request_anomaly(
    ws_uid: str,
    stock_key: Optional[str] = None,
    *,
    scope: str = "ws",
    ttl_sec: int = 300,
    reason: str = "anomaly",
) -> None:
    if not ws_uid:
        return
    now = time.time()
    exp = now + max(1, int(ttl_sec))
    key = _k(ws_uid, str(stock_key or "ALL"), scope)
    with _LOCK:
        _ANOM[key] = (exp, reason)
    KP.inc("trigger.anomaly.request", 1)

def peek_anomaly(ws_uid: str, stock_key: Optional[str] = None, *, scope: str = "ws") -> bool:
    if not ws_uid:
        return False
    now = time.time()
    key = _k(ws_uid, str(stock_key or "ALL"), scope)
    with _LOCK:
        ent = _ANOM.get(key)
        if not ent:
            return False
        exp, _ = ent
        if now > exp:
            _ANOM.pop(key, None)
            return False
        return True

def consume_anomaly(ws_uid: str, stock_key: Optional[str] = None, *, scope: str = "ws") -> Optional[str]:
    """
    Consume (clear) an active anomaly trigger and return its reason. If none, returns None.
    """
    if not ws_uid:
        return None
    now = time.time()
    key = _k(ws_uid, str(stock_key or "ALL"), scope)
    with _LOCK:
        ent = _ANOM.get(key)
        if not ent:
            return None
        exp, reason = ent
        if now > exp:
            _ANOM.pop(key, None)
            return None
        _ANOM.pop(key, None)
        KP.inc("trigger.anomaly.consume", 1)
        return reason

def baseline_due(
    ws_uid: str,
    stock_key: Optional[str] = None,
    *,
    scope: str = "ws",
    every_sec: int = 0,
    now_epoch: Optional[float] = None,
) -> bool:
    if not ws_uid:
        return False
    if int(every_sec or 0) <= 0:
        return False
    now = float(now_epoch) if now_epoch is not None else time.time()
    key = _k(ws_uid, str(stock_key or "ALL"), scope)
    with _LOCK:
        last = float(_LAST_BASELINE.get(key, 0.0))
        return (now - last) >= float(every_sec)

def mark_baseline_run(
    ws_uid: str,
    stock_key: Optional[str] = None,
    *,
    scope: str = "ws",
    now_epoch: Optional[float] = None,
) -> None:
    if not ws_uid:
        return
    now = float(now_epoch) if now_epoch is not None else time.time()
    key = _k(ws_uid, str(stock_key or "ALL"), scope)
    with _LOCK:
        _LAST_BASELINE[key] = now
    KP.inc("trigger.baseline.mark", 1)



def gc_expired(now_epoch: float | None = None, *, baseline_ttl_sec: int = 7 * 24 * 3600) -> int:
    """
    Remove expired anomaly triggers and (optionally) very old baseline keys.
    Returns number of removed keys.
    """
    now = float(now_epoch) if now_epoch is not None else time.time()
    removed = 0

    with _LOCK:
        # Expire triggers
        for k in list(_TRIGGERS.keys()):
            ent = _TRIGGERS.get(k)
            if not ent:
                _TRIGGERS.pop(k, None)
                removed += 1
                continue

            # ent is (exp, reason)
            try:
                exp = float(ent[0])
            except Exception:
                exp = None

            if (exp is None) or (exp <= now):
                _TRIGGERS.pop(k, None)
                removed += 1

        # Optional: prune baseline markers if they are extremely old (memory hygiene)
        if int(baseline_ttl_sec or 0) > 0:
            ttl = float(baseline_ttl_sec)
            for k in list(_BASELINES.keys()):
                try:
                    last = float(_BASELINES.get(k, 0.0))
                except Exception:
                    last = 0.0
                if (now - last) >= ttl:
                    _BASELINES.pop(k, None)
                    removed += 1

    try:
        KP.observe("trigger.gc.removed", float(removed))
    except Exception:
        pass
    return removed


def stats(now_epoch: float | None = None) -> dict:
    """
    Lightweight stats for logging/observability.
    """
    now = float(now_epoch) if now_epoch is not None else time.time()
    with _LOCK:
        trig_total = len(_TRIGGERS)
        base_total = len(_BASELINES)

        active = 0
        next_exp = None
        for ent in _TRIGGERS.values():
            if not ent:
                continue
            try:
                exp = float(ent[0])
            except Exception:
                continue
            if exp > now:
                active += 1
                next_exp = exp if next_exp is None else min(next_exp, exp)

    return {
        "triggers_total": trig_total,
        "triggers_active": active,
        "baselines_total": base_total,
        "next_trigger_in_sec": (round(next_exp - now, 3) if next_exp is not None else None),
    }
