from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

_NULL_TOKENS = {"", "0", "none", "null", "nan", "undefined", "n/a"}


def norm_id(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    if s.lower().strip() in _NULL_TOKENS:
        return ""
    return s


def parse_batch_root(batch_id: Any) -> Tuple[str, str]:
    """Returns (batch_id_norm, batch_root_for_pk).

    Examples:
      "REFNO:009927" -> (same, "009927")
      "SESSION:1700000000" -> (same, "0")
      "009927" -> ("REFNO:009927"? no; keep, "009927")
    """
    b = norm_id(batch_id)
    if not b:
        return "", "0"

    # already formatted
    if ":" in b:
        strat, root = b.split(":", 1)
        root = root.strip()
        if root.isdigit():
            return b, root
        return b, "0"

    # raw numeric
    if b.isdigit():
        return b, b

    return b, "0"


def parse_phase_process_no(phase_id: Any) -> Tuple[str, str]:
    """Returns (phase_id_norm, process_no_for_saveData).

    saveData() prefixes PID_. We should pass the numeric part if possible.
    """
    p = norm_id(phase_id)
    if not p:
        return "", "0"

    # common format "PID_123" or "PID-123"
    m = re.search(r"(\d{3,})", p)
    if m:
        return p, m.group(1)

    # fallback: strip non-alnum
    safe = re.sub(r"[^A-Za-z0-9]+", "_", p).strip("_")
    return p, safe or "0"


def event_ts_ms_from_msg(msg: Dict[str, Any]) -> int:
    v = msg.get("_event_ts_ms")
    try:
        if v is not None:
            return int(float(v))
    except Exception:
        pass

    # fallback: measDt / crDt
    for k in ("crDt", "measDt", "ts", "timestamp"):
        try:
            vv = msg.get(k)
            if vv is None:
                continue
            iv = int(float(str(vv).strip()))
            if iv > 10**12:
                return iv
            if iv > 10**9:
                return iv * 1000
        except Exception:
            continue
    return 0


def ms_to_dt_utc(ms: int) -> datetime:
    if not ms:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def json_dumps_stable(obj: Any) -> str:
    """Stable JSON for logs/debug; not used for Cassandra Map storage."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def pick_first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def extract_stock_key(msg: Dict[str, Any]) -> str:
    """Return a stable stock key for routing Phase3V2 artifacts.

    Priority:
      1) top-level produced/output stock id/name fields (if Stage0 lifts them)
      2) prodList[0].stNo / stNm (canonical in Kafka payloads)
      3) outVals[0].stNo / stNm (if present)
      4) fallback "ALL"
    """
    # prefer produced/output stock_no for stability; fallback to name
    v = pick_first(msg, ("produced_stock_no", "output_stock_no", "stNo", "stock_no"))
    if v is None:
        v = pick_first(msg, ("produced_stock_name", "output_stock_name", "stNm", "stock_name"))

    if v is None:
        prod_list = msg.get("prodList") or []
        if isinstance(prod_list, list):
            it0 = next((it for it in prod_list if isinstance(it, dict)), None)
            if it0:
                v = it0.get("stNo") or it0.get("stNm")

    if v is None:
        out_vals = msg.get("outVals") or []
        if isinstance(out_vals, list):
            it0 = next((it for it in out_vals if isinstance(it, dict)), None)
            if it0:
                v = it0.get("stNo") or it0.get("stNm")

    return norm_id(v) or "ALL"

def extract_op_tc(msg: Dict[str, Any]) -> str:
    return norm_id(pick_first(msg, ("operationtaskcode", "opTc", "op_tc", "operation_task_code")))


def extract_sensor_values(msg: Dict[str, Any]) -> Dict[str, float]:
    """Extract wide sensor readings from an incoming message.

    Uses outVals[*].eqNm preferred; fallback to eqNo.
    Values use cntRead (counter_reading) preferred.
    """
    out: Dict[str, float] = {}
    out_vals = msg.get("outVals") or msg.get("out_vals") or []
    if isinstance(out_vals, list):
        for ov in out_vals:
            if not isinstance(ov, dict):
                continue
            name = pick_first(ov, ("eqNm", "equipment_name", "eqNo", "equipment_no"))
            name_s = norm_id(name)
            if not name_s:
                continue
            val = pick_first(ov, ("cntRead", "counter_reading", "genReadVal", "gen_read_val"))
            out[name_s] = safe_float(val, default=0.0)

    # support some payloads where value is already flattened
    if not out and isinstance(msg.get("sensor_values"), dict):
        for k, v in (msg.get("sensor_values") or {}).items():
            ks = norm_id(k)
            if ks:
                out[ks] = safe_float(v, default=0.0)

    return out


def extract_meta_for_predictions(msg: Dict[str, Any], *, batch_root: str, process_no: str, ts_utc: datetime) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    out_vals = msg.get("outVals") or []
    if not isinstance(out_vals, list):
        out_vals = []

    prod_list = msg.get("prodList") or []
    if not isinstance(prod_list, list):
        prod_list = []

    # customer can live at top-level (Stage0 lifted) or inside outVals[*].cust in raw payloads
    cust = msg.get("customer") or msg.get("cust")
    if not cust and out_vals:
        cust = next((it.get("cust") for it in out_vals if isinstance(it, dict) and it.get("cust")), None)
    meta["customer"] = norm_id(cust)

    meta["plant_id"] = norm_id(msg.get("plId") or msg.get("plant_id"))

    meta["workcenter_name"] = norm_id(msg.get("wcNm") or msg.get("work_center_name"))
    meta["workcenter_no"] = norm_id(msg.get("wcNo") or msg.get("work_center_no") or msg.get("wcId") or msg.get("work_center_id"))

    meta["workstation_name"] = norm_id(msg.get("wsNm") or msg.get("work_station_name"))
    meta["workstation_no"] = norm_id(msg.get("wsNo") or msg.get("work_station_no") or msg.get("wsId") or msg.get("work_station_id"))

    meta["operator_name"] = norm_id(msg.get("operator_name") or msg.get("opNm"))
    meta["operator_no"] = norm_id(msg.get("operator_no") or msg.get("opNo"))

    # stock metadata: prefer lifted fields, otherwise prodList[0] (canonical)
    out_stock_name = msg.get("produced_stock_name") or msg.get("output_stock_name")
    out_stock_no = msg.get("produced_stock_no") or msg.get("output_stock_no")

    if (not out_stock_no and not out_stock_name) and prod_list:
        it0 = next((it for it in prod_list if isinstance(it, dict)), None)
        if it0:
            out_stock_no = out_stock_no or it0.get("stNo")
            out_stock_name = out_stock_name or it0.get("stNm") or it0.get("stNo")

    meta["output_stock_name"] = norm_id(out_stock_name)
    meta["output_stock_no"] = norm_id(out_stock_no)

    meta["job_order_reference_no"] = norm_id(msg.get("joRef") or msg.get("job_order_reference_no"))
    meta["process_no"] = norm_id(process_no)

    # prod_order_reference_no is the partition key for legacy prediction table
    meta["prod_order_reference_no"] = norm_id(batch_root) or "0"
    meta["start_date"] = ts_utc

    # passthrough (legacy fields that some writers expect)
    meta["process_no"] = norm_id(process_no) or "0"
    meta["proces_no"] = norm_id(msg.get("proces_no") or msg.get("process_no"))

    return meta

