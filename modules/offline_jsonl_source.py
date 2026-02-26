# modules/offline_jsonl_source.py
"""
Offline JSONL source utilities.

Purpose
-------
Allow offline trainers to consume Kafka dump recordings (JSONL) when Cassandra RAW/DW
recordings are unavailable.

Design constraints
------------------
- Streaming-friendly (line-by-line).
- Works with "flattened message" dumps where each line is a dict that contains:
    - plId, wcId, wsId, opTc, crDt (epoch ms), outVals (list of sensors), prodList (list of stocks)
  (This matches the existing kafka_dump_filtered_*.jsonl artifacts used in this project.)
- Produces "row-like" dicts compatible with existing offline trainer conversion utilities:
    measurement_date, equipment_name, counter_reading, produced_stock_no, operationtaskcode, ...
- Event-time policy: prefer crDt (epoch ms) at the message level. (If missing, fall back to
  _event_ts_ms or max(outVals.measDt).)

NOTE: This module intentionally does NOT depend on Cassandra models.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


UTC = timezone.utc
TR_TZ = timezone(timedelta(hours=3))  # for naive timestamp parsing if needed


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, float):
            if math.isnan(x):
                return default
            return int(x)
        return int(str(x))
    except Exception:
        return default


def extract_event_ts_ms(msg: Dict[str, Any]) -> int:
    """
    Return event timestamp in epoch ms.

    Priority:
      1) crDt (message-level)
      2) _event_ts_ms
      3) max(outVals[*].measDt)
      4) measDt / timestamp (if present)
    """
    ts = _safe_int(msg.get("crDt") or 0, 0)
    if ts > 0:
        return ts
    ts = _safe_int(msg.get("_event_ts_ms") or 0, 0)
    if ts > 0:
        return ts
    ov = msg.get("outVals") or []
    if isinstance(ov, list) and ov:
        mx = 0
        for o in ov:
            if isinstance(o, dict):
                mx = max(mx, _safe_int(o.get("measDt") or 0, 0))
        if mx > 0:
            return mx
    ts = _safe_int(msg.get("measDt") or msg.get("timestamp") or 0, 0)
    return ts


def ts_ms_to_dt_utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=UTC)


def _norm_s(s: Any) -> str:
    return str(s or "").strip()


def _stock_items(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    pl = msg.get("prodList") or []
    if isinstance(pl, list):
        return [x for x in pl if isinstance(x, dict)]
    return []


def primary_stock(msg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (stNo, stNm) for the "primary" stock in a message.

    Convention: use prodList[0] when present.
    """
    items = _stock_items(msg)
    if not items:
        return ("", "")
    it = items[0]
    return (_norm_s(it.get("stNo")), _norm_s(it.get("stNm") or it.get("stNo")))


def stock_matches(msg: Dict[str, Any], selector: str) -> bool:
    """
    Match selector against any prodList stNo/stNm (case-insensitive exact match).
    Selector may be "ALL" to match everything.
    """
    sel = _norm_s(selector)
    if not sel or sel.upper() == "ALL":
        return True
    sel_u = sel.upper()
    for it in _stock_items(msg):
        a = _norm_s(it.get("stNo")).upper()
        b = _norm_s(it.get("stNm") or it.get("stNo")).upper()
        if a == sel_u or b == sel_u:
            return True
    # also match primary_stock derived
    st_no, st_nm = primary_stock(msg)
    if st_no.upper() == sel_u or st_nm.upper() == sel_u:
        return True
    return False


def parse_dt_like_to_utc(dt_like: str) -> datetime:
    """
    Parse a datetime string into UTC.

    - If dt_like includes timezone, parse and convert to UTC.
    - If dt_like is naive (no tz), interpret as TR local wall-clock (+03:00) and convert to UTC.

    This is used ONLY for JSONL filtering (where event timestamps are epoch ms, UTC-based).
    """
    s = _norm_s(dt_like)
    if not s:
        raise ValueError("empty datetime")
    # very small parser using datetime.fromisoformat when possible
    # normalize 'Z'
    if s.endswith("Z"):
        s2 = s[:-1] + "+00:00"
    else:
        s2 = s
    try:
        dt = datetime.fromisoformat(s2)
    except Exception:
        # fallback: try common "YYYY-mm-dd HH:MM:SS(.ffffff)" without tz
        dt = datetime.fromisoformat(s2.replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TR_TZ)
    return dt.astimezone(UTC)


def iter_jsonl_messages(paths: Union[str, Sequence[str]]) -> Iterator[Dict[str, Any]]:
    if isinstance(paths, str):
        ps = [p.strip() for p in paths.split(",") if p.strip()]
    else:
        ps = list(paths)
    for p in ps:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj


def jsonl_probe_latest_dt(
    paths: Union[str, Sequence[str]],
    *,
    pl_id: int,
    wc_id: int,
    ws_id: int,
    st_no: str = "ALL",
    op_tc: str = "ALL",
) -> Optional[datetime]:
    mx = 0
    for msg in iter_jsonl_messages(paths):
        if int(msg.get("plId") or 0) != int(pl_id):
            continue
        if int(msg.get("wcId") or 0) != int(wc_id):
            continue
        if int(msg.get("wsId") or 0) != int(ws_id):
            continue
        if op_tc and str(op_tc).upper() != "ALL":
            if _norm_s(msg.get("opTc")).upper() != str(op_tc).upper():
                continue
        if not stock_matches(msg, st_no):
            continue
        ts_ms = extract_event_ts_ms(msg)
        if ts_ms > mx:
            mx = ts_ms
    if mx <= 0:
        return None
    return ts_ms_to_dt_utc(mx)


def jsonl_to_rows(
    paths: Union[str, Sequence[str]],
    *,
    pl_id: int,
    wc_id: int,
    ws_id: int,
    start_dt: datetime,
    end_dt: datetime,
    limit: int = 0,
    st_no: str = "ALL",
    op_tc: str = "ALL",
) -> List[Dict[str, Any]]:
    """
    Convert JSONL messages to row-like dicts (one per outVals item).

    Returned dict keys are intentionally aligned to dw_tbl_raw_data_by_ws field names used by trainers:
      - measurement_date (datetime, tz-aware UTC)
      - equipment_name (str)
      - counter_reading (float)
      - produced_stock_no / produced_stock_name
      - operationtaskcode (opTc)
      - plus some segment candidates: refNo, joRef, joOpId
    """
    start_ms = int(start_dt.astimezone(UTC).timestamp() * 1000.0)
    end_ms = int(end_dt.astimezone(UTC).timestamp() * 1000.0)
    rows: List[Dict[str, Any]] = []
    for msg in iter_jsonl_messages(paths):
        if int(msg.get("plId") or 0) != int(pl_id):
            continue
        if int(msg.get("wcId") or 0) != int(wc_id):
            continue
        if int(msg.get("wsId") or 0) != int(ws_id):
            continue
        if op_tc and str(op_tc).upper() != "ALL":
            if _norm_s(msg.get("opTc")).upper() != str(op_tc).upper():
                continue
        if not stock_matches(msg, st_no):
            continue

        ts_ms = extract_event_ts_ms(msg)
        if ts_ms <= 0:
            continue
        if ts_ms < start_ms or ts_ms >= end_ms:
            continue

        st_primary_no, st_primary_nm = primary_stock(msg)
        ov = msg.get("outVals") or []
        if not isinstance(ov, list) or not ov:
            continue

        # Build one row per outVal.
        ts_dt = ts_ms_to_dt_utc(ts_ms)
        for o in ov:
            if not isinstance(o, dict):
                continue
            v = o.get("cntRead")
            try:
                v_f = float(v)
                if math.isnan(v_f):
                    continue
            except Exception:
                continue
            sensor = _norm_s(o.get("eqNm") or o.get("eqNo") or o.get("eqId"))
            if not sensor:
                continue
            row = {
                "measurement_date": ts_dt,
                "equipment_name": sensor,
                "counter_reading": v_f,
                "produced_stock_no": st_primary_no,
                "produced_stock_name": st_primary_nm,
                "operationtaskcode": _norm_s(msg.get("opTc")),
                "refNo": msg.get("refNo"),
                "joRef": msg.get("joRef"),
                "joOpId": msg.get("joOpId"),
            }
            rows.append(row)
            if limit and len(rows) >= int(limit):
                return rows
    return rows
