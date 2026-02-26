# modules/workstation_profile.py
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from modules.resample_policy import recommend_resample_policy, policy_to_dict

# IMPORTANT:
# Use the training-friendly projection table keyed by (plId,wcId,wsId)+measurement_date.
# This avoids ALLOW FILTERING and replica-heavy scans on the legacy dw_tbl_raw_data table.
from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws


def _probe_latest_measurement_date(pl_id: int, wc_id: int, ws_id: int):
    """
    Best-effort fetch of the latest measurement_date for (pl,wc,ws).
    Assumes clustering order is DESC on measurement_date (common for time-series tables).
    """
    try:
        q = dw_tbl_raw_data_by_ws.objects.filter(
            plant_id=int(pl_id),
            work_center_id=int(wc_id),
            work_station_id=int(ws_id),
        ).limit(1)
        rows = list(q)
        if not rows:
            return None
        return getattr(rows[0], "measurement_date", None)
    except Exception:
        return None


def _rows_to_df(rows) -> pd.DataFrame:
    """
    Convert Cassandra rows into a thin dataframe containing only what the profiler needs.

    Expected row fields (dw_tbl_raw_data_by_ws):
      - measurement_date
      - produced_stock_no
      - operationtaskcode
      - equipment_name
      - counter_reading
    """
    if not rows:
        return pd.DataFrame()

    recs = []
    for r in rows:
        try:
            recs.append(
                {
                    "ts": getattr(r, "measurement_date", None),
                    "st_no": getattr(r, "produced_stock_no", None),
                    "op_tc": getattr(r, "operationtaskcode", None),
                    "sensor": getattr(r, "equipment_name", None),
                    "val": getattr(r, "counter_reading", None),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df["st_no"] = df["st_no"].fillna("0").astype(str)
    df["op_tc"] = df["op_tc"].fillna("0").astype(str)
    df["sensor"] = df["sensor"].fillna("").astype(str)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])
    return df


def _fetch_rows(
    pl_id: int,
    wc_id: int,
    ws_id: int,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    *,
    st_no: Optional[str] = None,
    op_tc: Optional[str] = None,
):
    """
    Fetch raw rows from dw_tbl_raw_data_by_ws for a specific workstation partition and time window.
    Efficient range query: (plId,wcId,wsId) partition + measurement_date clustering.
    """
    q = dw_tbl_raw_data_by_ws.objects.filter(
        plant_id=int(pl_id),
        work_center_id=int(wc_id),
        work_station_id=int(ws_id),
        measurement_date__gte=start_dt,
        measurement_date__lt=end_dt,
    )

    if st_no is not None:
        q = q.filter(produced_stock_no=str(st_no))
    if op_tc is not None:
        q = q.filter(operationtaskcode=str(op_tc))

    return list(q.limit(int(limit)))


def build_workstation_profile(
    pl_id: int,
    wc_id: int,
    ws_id: int,
    *,
    days: int = 7,
    limit: int = 50000,
    st_no: Optional[str] = None,
    op_tc: Optional[str] = None,
    resample_sec: int = 60,
) -> Dict:
    """
    Lightweight workstation profiler for planning offline training.

    Design goals:
      - anchor the time window in **event-time** (latest measurement_date), not wall-clock,
        to handle retrospective dumps (WS441165-like customers).
      - keep reads bounded and deterministic

    Returns:
      - stocks_top
      - sensors_top
      - out_targets_top (coverage-based candidates)
      - debug stats
    """
    latest = _probe_latest_measurement_date(pl_id, wc_id, ws_id)
    end_dt = pd.to_datetime(latest, utc=True).to_pydatetime() if latest is not None else datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(days))

    # Chunk reads to bound queries for large windows/partitions.
    chunk_hours = 12
    per_chunk_limit = max(5000, int(limit // max(1, (days * 24) // chunk_hours)))

    rows_all = []
    cur = start_dt
    while cur < end_dt and len(rows_all) < int(limit):
        nxt = min(end_dt, cur + timedelta(hours=chunk_hours))
        take = min(per_chunk_limit, int(limit) - len(rows_all))
        rows_all.extend(_fetch_rows(pl_id, wc_id, ws_id, cur, nxt, take, st_no=st_no, op_tc=op_tc))
        cur = nxt

    df = _rows_to_df(rows_all)
    if df.empty:
        return {
            "ok": False,
            "reason": "no_rows",
            "pl_id": pl_id,
            "wc_id": wc_id,
            "ws_id": ws_id,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "anchor_latest_measurement_date": None if latest is None else str(pd.to_datetime(latest)),
        }

    # Resample policy (v1): derive a recommended resample grid from event-time observations.
    # This is used by Phase2 and Phase3 for routing and to avoid hardcoding 60s everywhere.
    rsp = recommend_resample_policy(df, ts_col="ts", sensor_col="sensor", val_col="val")
    rsp_dict = policy_to_dict(rsp)

    resample_sec_effective = int(resample_sec) if resample_sec and int(resample_sec) > 0 else int(rsp.recommended_resample_sec)

    stocks = Counter(df["st_no"].tolist())
    sensors = Counter(df["sensor"].tolist())

    # Coverage-based output candidate selection (cheap; trainer does trainability scoring later)
    out_cov = []
    freq = f"{int(resample_sec_effective)}s"
    for s, c in sensors.most_common(200):
        sdf = df[df["sensor"] == s][["ts", "val"]].copy()
        if sdf.empty:
            continue
        sdf = sdf.set_index("ts").sort_index()
        r = sdf["val"].resample(freq).last()
        nn = int(r.notna().sum())
        cov = float(nn / len(r)) if len(r) else 0.0
        # (sensor, coverage over full window, non_nan_buckets, raw_points, total_buckets)
        out_cov.append((s, cov, nn, int(c), int(len(r))))

    out_cov.sort(key=lambda x: (x[2], x[3]), reverse=True)
    out_targets_top = [s for (s, cov, nn, c, n) in out_cov if nn >= 20][:10]

    return {
        "ok": True,
        "pl_id": pl_id,
        "wc_id": wc_id,
        "ws_id": ws_id,
        "resample_sec_input": int(resample_sec) if resample_sec is not None else None,
        "resample_sec_effective": int(resample_sec_effective),
        "resample_policy": rsp_dict,
        "start": df["ts"].min().isoformat(),
        "end": df["ts"].max().isoformat(),
        "rows": int(len(df)),
        "anchor_latest_measurement_date": None if latest is None else str(pd.to_datetime(latest)),
        "stocks_top": [{"st_no": k, "count": int(v)} for k, v in stocks.most_common(10)],
        "sensors_top": [{"sensor": k, "count": int(v)} for k, v in sensors.most_common(20)],
        "out_targets_top": out_targets_top,
        "out_targets_debug": [
            {"sensor": s, "coverage": cov, "non_nan_buckets": nn, "raw_n": c, "total_buckets": n}
            for (s, cov, nn, c, n) in out_cov[:30]
        ],
    }