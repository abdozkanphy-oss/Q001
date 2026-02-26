"""Shared identity + context helpers.

Goal: keep workstation/stock identity consistent across Stage0, Phase2, Phase3, and offline trainers.

Design:
- Prefer M1-derived fields added during Stage0 normalization (_workstation_uid, etc.)
- Fall back to computing from plant/workcenter/workstation IDs when needed.
- Normalize 'missing' tokens consistently ("0", "null", etc.)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

NULL_TOKENS = {"", "0", "none", "null", "nan", "n/a", "unknown", "unk"}


def _clean_token(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in NULL_TOKENS:
        return None
    return s


def get_workstation_uid(message: Dict[str, Any]) -> str:
    """Return canonical workstation UID.

    Canonical format used across this project (logs/artifacts + model registry):
        "<plId>_WC<wcId>_WS<wsId>"

    Notes:
    - If Stage0 already created "_workstation_uid", we trust it (source of truth).
    - Prefer numeric/ID fields (plId/wcId/wsId). Do NOT use outVals[*].cust as identity.
    """
    if not isinstance(message, dict):
        return "WSUID_UNKNOWN"

    ws = _clean_token(message.get("_workstation_uid"))
    if ws:
        return ws

    # Preferred keys (Kafka-style)
    pl = _clean_token(message.get("plId") or message.get("plant_id") or message.get("plantid"))
    wc = _clean_token(
        message.get("wcId")
        or message.get("work_center_id")
        or message.get("workcenterid")
        or message.get("wcNo")
        or message.get("workcenter_no")
    )
    wsid = _clean_token(
        message.get("wsId")
        or message.get("work_station_id")
        or message.get("workstationid")
        or message.get("wsNo")
        or message.get("workstation_no")
    )

    if pl and wc and wsid:
        return f"{pl}_WC{wc}_WS{wsid}"

    # Extremely defensive: if plant is missing, still avoid collapsing across WS
    if wc and wsid:
        return f"UNKNOWNPL_WC{wc}_WS{wsid}"

    return "WSUID_UNKNOWN"


def get_stock_key(message: Dict[str, Any], default: str = "ALL") -> str:
    """Return canonical stock key for WS+STOCK context.

    Preference order:
      1) output_stock_no / produced_stock_no (Phase3 / DW often uses this)
      2) prodList[0].stNo / stNm / stId  (Kafka messages)
      3) default ("ALL")
    """
    if not isinstance(message, dict):
        return default

    st = _clean_token(message.get("output_stock_no") or message.get("produced_stock_no"))
    if st:
        return st

    prod = message.get("prodList") or []
    if isinstance(prod, dict):
        prod = [prod]
    if isinstance(prod, list) and prod:
        p0 = prod[0] if isinstance(prod[0], dict) else {}
        st2 = _clean_token(p0.get("stNo") or p0.get("stNm") or p0.get("stId"))
        if st2:
            return st2

    return default
