# modules/offline_mimo_rf_trainer.py
"""M4.x: Offline supervised Multi-Input / Multi-Output RandomForest trainer.

Goal
----
Train *one* supervised model per (workstation, selected stock) that predicts
multiple target sensors jointly (multi-output regression). This addresses the
"--stNo trains a single model for that product" requirement.

Design constraints (project standards)
-------------------------------------
- Read training data from dw_tbl_raw_data_by_ws using only partition keys + time
  range (no ALLOW FILTERING assumptions). Stock/op filters are applied *after*
  fetch.
- Event-time anchored, resample/bin to a fixed grid, gap-guard, minimal forward
  fill.
- Avoid cross-batch leakage by segmenting (AUTO policy supported).
- Produce explicit artifacts (model + meta.json) with semantics:
    wsuid, stNo, opTc, resample_sec, horizon_sec, n_lags, targets, feature_names.

CLI
---
python -m modules.offline_mimo_rf_trainer \
  --plId 149 --wcId 951 --wsId 441165 --stNo "Antares PV" \
  --time_min "2025-11-06 08:00:00" --time_max "2025-12-06 08:00:00" \
  --targets "S1,S2,S3" --n_lags 6 --resample_sec 60 --horizon_sec 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modules.context_profiler import profile_context_from_rows
from modules.context_policy import select_context_policy
from modules.model_registry import safe_token


def _get_dw_model():
    """Lazy import to avoid a hard Cassandra dependency during unit tests.

    The Cassandra Python driver is not required to run the project's unit tests.
    This trainer only needs Cassandra when fetching rows (runtime usage).
    """

    from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws  # local import

    return dw_tbl_raw_data_by_ws


_WSUID_RE = re.compile(r"^(?P<pl>\d+)_WC(?P<wc>\d+)_WS(?P<ws>\d+)$")


def _parse_wsuid(wsuid: str) -> Optional[Tuple[int, int, int]]:
    if not wsuid:
        return None
    m = _WSUID_RE.match(str(wsuid).strip())
    if not m:
        return None
    return int(m.group("pl")), int(m.group("wc")), int(m.group("ws"))


def _norm_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"0", "none", "null", "nan"}:
        return None
    return s


def _hash_list(items: List[str], *, n: int = 10) -> str:
    h = hashlib.sha1("|".join(items).encode("utf-8")).hexdigest()
    return h[: max(6, int(n))]


def probe_latest_measurement_date(pl_id: int, wc_id: int, ws_id: int):
    """Best-effort fetch of latest measurement_date for (pl,wc,ws)."""
    try:
        dw_tbl_raw_data_by_ws = _get_dw_model()
        q = (
            dw_tbl_raw_data_by_ws.objects.filter(
                plant_id=int(pl_id),
                work_center_id=int(wc_id),
                work_station_id=int(ws_id),
            )
            .limit(1)
        )
        rows = list(q)
        if not rows:
            return None
        return getattr(rows[0], "measurement_date", None)
    except Exception:
        return None


def fetch_rows_by_ws_time_range(
    pl_id: int,
    wc_id: int,
    ws_id: int,
    *,
    time_min: Optional[datetime],
    time_max: Optional[datetime],
    days: int,
    limit: int = 50000,
) -> List[Any]:
    """Fetch rows from dw_tbl_raw_data_by_ws for WS partition + time range.

    IMPORTANT: We intentionally do NOT filter by stock/opTc at the Cassandra
    query level, to avoid relying on ALLOW FILTERING or secondary indexes.
    """
    dw_tbl_raw_data_by_ws = _get_dw_model()
    latest = probe_latest_measurement_date(pl_id, wc_id, ws_id)
    end_dt = pd.to_datetime(latest, utc=True).to_pydatetime() if latest is not None else datetime.utcnow()

    if time_max is not None:
        end_dt = min(end_dt, time_max)

    if time_min is not None:
        start_dt = time_min
    else:
        start_dt = end_dt - timedelta(days=int(days))

    if start_dt >= end_dt:
        return []

    chunk_hours = 12
    span_hours = max(1.0, (end_dt - start_dt).total_seconds() / 3600.0)
    n_chunks = max(1, int(np.ceil(span_hours / float(chunk_hours))))
    per_chunk_limit = max(2000, int(int(limit) // n_chunks))

    rows_all: List[Any] = []
    cur = start_dt
    while cur < end_dt and len(rows_all) < int(limit):
        nxt = min(end_dt, cur + timedelta(hours=chunk_hours))
        take = min(per_chunk_limit, int(limit) - len(rows_all))

        q = dw_tbl_raw_data_by_ws.objects.filter(
            plant_id=int(pl_id),
            work_center_id=int(wc_id),
            work_station_id=int(ws_id),
            measurement_date__gte=cur,
            measurement_date__lt=nxt,
        )
        rows_all.extend(list(q.limit(int(take))))
        cur = nxt

    return rows_all


def filter_rows_by_stock_op(
    rows: List[Any], *, st_no: str, op_tc: str
) -> List[Any]:
    """Filter fetched rows in Python (safe w.r.t. Cassandra query constraints)."""
    if not rows:
        return []

    st_no = str(st_no or "ALL").strip()
    op_tc = str(op_tc or "ALL").strip()
    st_all = st_no.upper() == "ALL" or st_no == ""
    op_all = op_tc.upper() == "ALL" or op_tc == ""

    out: List[Any] = []
    for r in rows:
        try:
            if not st_all:
                st1 = getattr(r, "produced_stock_no", None)
                st2 = getattr(r, "produced_stock_name", None)
                if str(st1) != st_no and str(st2) != st_no:
                    continue
            if not op_all:
                op1 = getattr(r, "operationtaskcode", None)
                if str(op1) != op_tc:
                    continue
            out.append(r)
        except Exception:
            continue
    return out


def rows_to_df(rows: List[Any], *, segment_field: str) -> pd.DataFrame:
    """Normalize Cassandra rows into long-form df: ts,sensor,val,seg."""
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        try:
            ts = getattr(r, "measurement_date", None)
            sensor = getattr(r, "equipment_name", None)
            val = getattr(r, "counter_reading", None)
            seg_val = None
            if segment_field and segment_field not in {"", "AUTO"}:
                seg_val = getattr(r, segment_field, None)
            recs.append({"ts": ts, "sensor": sensor, "val": val, "seg": seg_val})
        except Exception:
            continue

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df["sensor"] = df["sensor"].fillna("").astype(str)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])
    if "seg" in df.columns:
        df["seg"] = df["seg"].map(_norm_id)
    return df


def _build_session_ids(ts_index: pd.DatetimeIndex, *, session_gap_sec: int) -> pd.Series:
    if len(ts_index) == 0:
        return pd.Series(dtype="object")
    s = pd.Series(ts_index, index=ts_index).sort_index()
    gaps = s.diff().dt.total_seconds().fillna(0.0)
    new_sess = (gaps > float(session_gap_sec)).astype(int)
    sess_id = new_sess.cumsum()
    return sess_id.astype(str)


def _auto_ffill_limit(obs_ts: pd.DatetimeIndex, resample_sec: int, n_lags: int, cap: int = 600) -> int:
    if len(obs_ts) < 2:
        return min(cap, n_lags + 1)
    gaps = obs_ts.to_series().diff().dt.total_seconds().dropna()
    if len(gaps) == 0:
        return min(cap, n_lags + 1)
    med = float(gaps.median())
    buckets = max(1, int(round(med / float(resample_sec))))
    return min(cap, max(n_lags + 1, buckets))


def _auto_max_gap_sec(obs_ts: pd.DatetimeIndex) -> int:
    if len(obs_ts) < 2:
        return 6 * 3600
    gaps = obs_ts.to_series().diff().dt.total_seconds().dropna()
    if len(gaps) == 0:
        return 6 * 3600
    med = float(gaps.median())
    return int(min(24 * 3600, max(3600, 6 * med)))


def build_multitarget_dataset(
    df: pd.DataFrame,
    *,
    targets: List[str],
    resample_sec: int,
    horizon_sec: int,
    n_lags: int,
    segment_field: str,
    session_gap_sec: int,
    max_sensors: int,
    min_sensor_non_nan: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str], List[str], Dict[str, Any]]:
    """Build multi-target supervised dataset.

    Features: lag_1..lag_n for each selected sensor.
    Targets: values of each target sensor at +horizon.
    Baseline: lag0 persistence for each target.
    """
    if df is None or df.empty:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "empty_df"}

    targets = [str(t) for t in targets if str(t).strip()]
    targets = list(dict.fromkeys(targets))
    if not targets:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_targets"}

    freq = f"{int(resample_sec)}s"
    horizon_steps = max(1, int(round(float(horizon_sec) / float(resample_sec))))

    # segment ids
    seg_series = df["seg"].copy() if "seg" in df.columns else pd.Series([None] * len(df))
    if str(segment_field).upper() == "SESSION":
        tdf = df[df["sensor"].isin(targets)][["ts"]].copy()
        if tdf.empty:
            return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_target_rows"}
        tdf = tdf.sort_values("ts")
        sess_ids = _build_session_ids(pd.DatetimeIndex(tdf["ts"].tolist()), session_gap_sec=int(session_gap_sec))
        ts_sorted = pd.DatetimeIndex(tdf["ts"].tolist())
        sess_by_ts = pd.Series(sess_ids.values, index=ts_sorted)
        sess_by_ts = sess_by_ts[~sess_by_ts.index.duplicated(keep="last")].sort_index()
        all_ts = pd.DatetimeIndex(df["ts"].tolist())
        pos = sess_by_ts.index.searchsorted(all_ts, side="right") - 1
        out = []
        for p in pos:
            out.append("MISSING" if int(p) < 0 else str(sess_by_ts.iloc[int(p)]))
        seg_series = pd.Series(out, index=df.index)
    else:
        seg_series = seg_series.fillna("MISSING").astype(str)

    df2 = df.copy()
    df2["seg_id"] = seg_series
    segments = df2["seg_id"].unique().tolist()

    # global sensor selection
    sensor_counts = df2["sensor"].value_counts().to_dict()
    sensors_sorted = sorted(sensor_counts.keys(), key=lambda s: int(sensor_counts.get(s, 0)), reverse=True)
    sensors_used: List[str] = []
    for s in sensors_sorted:
        if int(sensor_counts.get(s, 0)) < int(min_sensor_non_nan):
            continue
        sensors_used.append(str(s))
        if len(sensors_used) >= int(max_sensors):
            break

    # ensure targets are present in sensors_used
    for t in targets:
        if t not in sensors_used:
            sensors_used = [t] + [x for x in sensors_used if x != t]
    sensors_used = sensors_used[: max(1, int(max_sensors))]

    if not sensors_used:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_sensors_selected"}

    # build feature names deterministically
    feature_names: List[str] = []
    for s in sensors_used:
        for k in range(1, int(n_lags) + 1):
            feature_names.append(f"{s}__lag_{k}")
    expected_dim = int(len(feature_names))

    X_parts: List[np.ndarray] = []
    Y_parts: List[np.ndarray] = []
    B0_parts: List[np.ndarray] = []
    SEG_parts: List[np.ndarray] = []

    seg_stats: Dict[str, Any] = {"segments_total": int(len(segments)), "segments_used": 0}

    for seg_id in segments:
        sdf = df2[df2["seg_id"] == seg_id]
        if sdf.empty:
            continue

        piv = sdf.pivot_table(index="ts", columns="sensor", values="val", aggfunc="last")
        if piv.empty:
            continue

        piv = piv.sort_index()
        rr = piv.resample(freq).last()

        # ensure all sensors exist
        for s in sensors_used:
            if s not in rr.columns:
                rr[s] = np.nan
        rr = rr[sensors_used]

        # ensure all targets exist (if missing entirely, skip this segment)
        if any(t not in rr.columns for t in targets):
            continue

        # target support check
        tgt_non_nan = [int(rr[t].notna().sum()) for t in targets]
        if int(min(tgt_non_nan)) < int(min_sensor_non_nan):
            continue

        # obs timestamps from union of target sensors within segment
        tgt_obs = sdf[sdf["sensor"].isin(targets)]["ts"].dropna()
        obs_ts = pd.DatetimeIndex(pd.to_datetime(tgt_obs, utc=True))
        ffill_limit = _auto_ffill_limit(obs_ts, resample_sec=resample_sec, n_lags=n_lags)
        max_gap_sec = _auto_max_gap_sec(obs_ts)

        idx = pd.Series(rr.index, index=rr.index)
        for s in rr.columns:
            last_obs = idx.where(rr[s].notna()).ffill()
            age_sec = (idx - last_obs).dt.total_seconds()
            rr[s] = rr[s].where(age_sec <= float(max_gap_sec))
            rr[s] = rr[s].ffill(limit=int(ffill_limit))

        # build lag features
        X_df = pd.DataFrame(index=rr.index)
        for s in sensors_used:
            for k in range(1, int(n_lags) + 1):
                X_df[f"{s}__lag_{k}"] = rr[s].shift(k)

        Y_df = rr[targets].shift(-horizon_steps)
        B0_df = rr[targets]

        mask = Y_df.notna().all(axis=1)
        if int(mask.sum()) < int(min_sensor_non_nan):
            continue

        X = X_df.loc[mask].to_numpy(dtype="float32")
        Y = Y_df.loc[mask].to_numpy(dtype="float32")
        B0 = B0_df.loc[mask].to_numpy(dtype="float32")

        if X.shape[1] != expected_dim:
            # should not happen, but keep deterministic failure mode
            continue

        X_parts.append(X)
        Y_parts.append(Y)
        B0_parts.append(B0)
        SEG_parts.append(np.asarray([str(seg_id)] * X.shape[0], dtype=object))

        seg_stats["segments_used"] = int(seg_stats["segments_used"]) + 1

    if not X_parts:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, feature_names, sensors_used, {
            "reason": "no_segments_trainable",
            **seg_stats,
        }

    X_all = np.vstack(X_parts)
    Y_all = np.vstack(Y_parts)
    B0_all = np.vstack(B0_parts)
    seg_all = np.concatenate(SEG_parts)
    info = {
        "segments_total": int(seg_stats["segments_total"]),
        "segments_used": int(seg_stats["segments_used"]),
        "n_supervised": int(Y_all.shape[0]),
        "n_targets": int(len(targets)),
        "horizon_steps": int(horizon_steps),
    }
    return X_all, Y_all, B0_all, seg_all, feature_names, sensors_used, info


def _mae_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        return []
    out: List[float] = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        m = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(m):
            out.append(float("nan"))
        else:
            out.append(float(mean_absolute_error(yt[m], yp[m])))
    return out


def _eval_time_split(
    X: np.ndarray,
    Y: np.ndarray,
    B0: np.ndarray,
    *,
    min_test: int,
    baseline_perfect_eps: float,
) -> Dict[str, Any]:
    n = int(Y.shape[0])
    if n < int(min_test) + 10:
        return {"ok": False, "reason": "insufficient_rows", "n": n}
    n_test = max(int(min_test), int(round(0.2 * n)))
    n_test = min(n_test, n - 5)
    n_train = n - n_test

    Xtr, Xte = X[:n_train], X[n_train:]
    Ytr, Yte = Y[:n_train], Y[n_train:]
    B0te = B0[n_train:]

    base_mae = _mae_per_target(Yte, B0te)
    if base_mae and all(np.isfinite(x) and float(x) <= float(baseline_perfect_eps) for x in base_mae):
        return {
            "ok": False,
            "reason": "baseline_perfect",
            "baseline_mae_lag0_per_target": base_mae,
            "n_train": n_train,
            "n_test": n_test,
        }

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=None,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, Ytr)
    pred = pipe.predict(Xte)

    model_mae = _mae_per_target(Yte, pred)
    lift = []
    for a, b in zip(base_mae, model_mae):
        try:
            lift.append(float(a) - float(b))
        except Exception:
            lift.append(float("nan"))

    return {
        "ok": True,
        "eval_mode": "time_split",
        "n_train": int(n_train),
        "n_test": int(n_test),
        "baseline_mae_lag0_per_target": base_mae,
        "model_mae_per_target": model_mae,
        "lift_vs_lag0_per_target": lift,
        "pipeline": pipe,
    }


def _eval_batch_holdout(
    X: np.ndarray,
    Y: np.ndarray,
    B0: np.ndarray,
    seg: np.ndarray,
    *,
    holdout_k: int,
    min_points_per_batch: int,
    min_test: int,
    baseline_perfect_eps: float,
) -> Dict[str, Any]:
    if seg is None or len(seg) != int(Y.shape[0]):
        return {"ok": False, "reason": "no_seg"}

    # segment order by appearance (time-resampled index order already encoded in stacking)
    seg = np.asarray(seg, dtype=object)
    uniq = []
    seen = set()
    for s in seg.tolist():
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    if len(uniq) < int(holdout_k) + 1:
        return {"ok": False, "reason": "too_few_segments", "n_segments": int(len(uniq))}

    test_segs = set(uniq[-int(holdout_k) :])
    m_test = np.array([s in test_segs for s in seg.tolist()], dtype=bool)
    m_train = ~m_test

    # enforce per-batch point minimum for test segments
    ok_test = True
    for s in test_segs:
        if int(np.sum(seg[m_test] == s)) < int(min_points_per_batch):
            ok_test = False
            break
    if not ok_test or int(np.sum(m_test)) < int(min_test):
        return {"ok": False, "reason": "holdout_too_small"}

    Xtr, Xte = X[m_train], X[m_test]
    Ytr, Yte = Y[m_train], Y[m_test]
    B0te = B0[m_test]

    base_mae = _mae_per_target(Yte, B0te)
    if base_mae and all(np.isfinite(x) and float(x) <= float(baseline_perfect_eps) for x in base_mae):
        return {
            "ok": False,
            "reason": "baseline_perfect",
            "baseline_mae_lag0_per_target": base_mae,
            "n_train": int(Ytr.shape[0]),
            "n_test": int(Yte.shape[0]),
        }

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=None,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, Ytr)
    pred = pipe.predict(Xte)

    model_mae = _mae_per_target(Yte, pred)
    lift = []
    for a, b in zip(base_mae, model_mae):
        try:
            lift.append(float(a) - float(b))
        except Exception:
            lift.append(float("nan"))

    return {
        "ok": True,
        "eval_mode": "batch_holdout",
        "n_train": int(Ytr.shape[0]),
        "n_test": int(Yte.shape[0]),
        "holdout_k": int(holdout_k),
        "test_segments": sorted([str(s) for s in test_segs]),
        "baseline_mae_lag0_per_target": base_mae,
        "model_mae_per_target": model_mae,
        "lift_vs_lag0_per_target": lift,
        "pipeline": pipe,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline MIMO RandomForest trainer (per WS + selected stock).")
    ap.add_argument("--wsUid", default="", help="Optional: '<pl>_WC<wc>_WS<ws>'")
    ap.add_argument("--plId", type=int, default=0)
    ap.add_argument("--wcId", type=int, default=0)
    ap.add_argument("--wsId", type=int, default=0)

    ap.add_argument("--stNo", required=True, help="Stock selector (matches produced_stock_no or produced_stock_name)")
    ap.add_argument("--opTc", default="ALL")

    # time range controls
    ap.add_argument("--days", type=int, default=14, help="Used if time_min is not provided")
    ap.add_argument("--time_min", default="", help="Optional ISO-ish datetime (UTC assumed if no tz)")
    ap.add_argument("--time_max", default="", help="Optional ISO-ish datetime (UTC assumed if no tz)")
    ap.add_argument("--limit", type=int, default=50000)

    # model/data controls
    ap.add_argument("--targets", default="", help="Comma-separated target sensors; empty => auto")
    ap.add_argument("--max_targets", type=int, default=15)
    ap.add_argument("--n_lags", type=int, default=6)
    ap.add_argument("--resample_sec", type=int, default=60)
    ap.add_argument("--horizon_sec", type=int, default=60)
    ap.add_argument("--segment_field", default="AUTO", help="AUTO | SESSION | specific field name")
    ap.add_argument("--session_gap_sec", type=int, default=6 * 3600)

    ap.add_argument("--max_sensors", type=int, default=50)
    ap.add_argument("--min_sensor_non_nan", type=int, default=10)
    ap.add_argument("--min_rows", type=int, default=200, help="Minimum supervised rows required")

    ap.add_argument("--eval_mode", default="auto", choices=["auto", "time_split", "batch_holdout"])
    ap.add_argument("--holdout_k", type=int, default=2)
    ap.add_argument("--min_points_per_batch", type=int, default=50)
    ap.add_argument("--min_test", type=int, default=50)
    ap.add_argument("--baseline_perfect_eps", type=float, default=1e-12)
    ap.add_argument("--accept_min_lift_mean", type=float, default=0.0)

    ap.add_argument("--out_dir", default="./models/offline_mimo_rf")
    ap.add_argument("--run_id", default="")

    args = ap.parse_args()

    # Resolve ids
    pl_id = int(args.plId or 0)
    wc_id = int(args.wcId or 0)
    ws_id = int(args.wsId or 0)
    if args.wsUid:
        p = _parse_wsuid(str(args.wsUid))
        if p is not None:
            pl_id, wc_id, ws_id = p
    if pl_id <= 0 or wc_id <= 0 or ws_id <= 0:
        raise SystemExit("Must provide either --wsUid or --plId/--wcId/--wsId")

    out_dir = str(args.out_dir or "./models/offline_mimo_rf")
    os.makedirs(out_dir, exist_ok=True)

    run_id = str(args.run_id or "").strip() or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    time_min = pd.to_datetime(args.time_min, utc=True, errors="coerce") if str(args.time_min).strip() else None
    time_max = pd.to_datetime(args.time_max, utc=True, errors="coerce") if str(args.time_max).strip() else None
    if isinstance(time_min, pd.Timestamp) and pd.isna(time_min):
        time_min = None
    if isinstance(time_max, pd.Timestamp) and pd.isna(time_max):
        time_max = None
    time_min_dt = time_min.to_pydatetime() if time_min is not None else None
    time_max_dt = time_max.to_pydatetime() if time_max is not None else None

    # Fetch
    rows = fetch_rows_by_ws_time_range(
        pl_id,
        wc_id,
        ws_id,
        time_min=time_min_dt,
        time_max=time_max_dt,
        days=int(args.days),
        limit=int(args.limit),
    )
    if not rows:
        print("No rows fetched.")
        return 2

    # Post-filter by stock/op
    rows = filter_rows_by_stock_op(rows, st_no=str(args.stNo), op_tc=str(args.opTc))
    if not rows:
        print("No rows after stock/op filter.")
        return 2

    # Segment field selection
    segment_field = str(args.segment_field or "AUTO")
    ctx_policy = None
    if segment_field.upper() == "AUTO":
        ctx = profile_context_from_rows(rows)
        ctx_policy = select_context_policy(ctx, prefer_stock=True)
        segment_field = str(ctx_policy.get("segment_field") or "SESSION")

    df = rows_to_df(rows, segment_field=segment_field)
    if df.empty:
        print("No usable rows after normalization.")
        return 2

    # Targets
    if str(args.targets).strip():
        targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    else:
        vc = df["sensor"].value_counts()
        targets = vc.head(int(args.max_targets)).index.tolist()

    # Ensure we keep only targets present
    targets = [t for t in targets if t in set(df["sensor"].unique().tolist())]
    targets = list(dict.fromkeys([str(t) for t in targets]))
    if len(targets) < 1:
        print("No usable targets selected.")
        return 3

    resample_sec = int(args.resample_sec)
    horizon_sec = int(args.horizon_sec)
    n_lags = int(args.n_lags)

    X, Y, B0, seg, feature_names, sensors_used, info = build_multitarget_dataset(
        df,
        targets=targets,
        resample_sec=resample_sec,
        horizon_sec=horizon_sec,
        n_lags=n_lags,
        segment_field=str(segment_field),
        session_gap_sec=int(args.session_gap_sec),
        max_sensors=int(args.max_sensors),
        min_sensor_non_nan=int(args.min_sensor_non_nan),
    )

    if int(Y.shape[0]) < int(args.min_rows):
        print(f"Insufficient supervised rows: {int(Y.shape[0])} < {int(args.min_rows)}")
        return 3

    # Eval selection
    eval_mode = str(args.eval_mode)
    eval_result: Dict[str, Any]
    if eval_mode == "auto":
        if seg is not None and str(segment_field).upper() != "SESSION":
            eval_result = _eval_batch_holdout(
                X,
                Y,
                B0,
                seg,
                holdout_k=int(args.holdout_k),
                min_points_per_batch=int(args.min_points_per_batch),
                min_test=int(args.min_test),
                baseline_perfect_eps=float(args.baseline_perfect_eps),
            )
            if not eval_result.get("ok"):
                eval_result = _eval_time_split(
                    X,
                    Y,
                    B0,
                    min_test=int(args.min_test),
                    baseline_perfect_eps=float(args.baseline_perfect_eps),
                )
        else:
            eval_result = _eval_time_split(
                X,
                Y,
                B0,
                min_test=int(args.min_test),
                baseline_perfect_eps=float(args.baseline_perfect_eps),
            )
    elif eval_mode == "batch_holdout":
        eval_result = _eval_batch_holdout(
            X,
            Y,
            B0,
            seg,
            holdout_k=int(args.holdout_k),
            min_points_per_batch=int(args.min_points_per_batch),
            min_test=int(args.min_test),
            baseline_perfect_eps=float(args.baseline_perfect_eps),
        )
        if not eval_result.get("ok"):
            print(f"Batch holdout failed: {eval_result.get('reason')}; try time_split")
            return 4
    else:
        eval_result = _eval_time_split(
            X,
            Y,
            B0,
            min_test=int(args.min_test),
            baseline_perfect_eps=float(args.baseline_perfect_eps),
        )

    if not eval_result.get("ok"):
        print(f"Training skipped: {eval_result.get('reason')}")
        return 4

    pipe = eval_result.pop("pipeline")

    # Aggregate metrics
    base_mae = eval_result.get("baseline_mae_lag0_per_target") or []
    model_mae = eval_result.get("model_mae_per_target") or []
    lift = eval_result.get("lift_vs_lag0_per_target") or []
    lift_mean = float(np.nanmean(np.asarray(lift, dtype="float64"))) if lift else float("nan")
    accepted = bool(np.isfinite(lift_mean) and lift_mean >= float(args.accept_min_lift_mean))

    wsuid = f"{pl_id}_WC{wc_id}_WS{ws_id}"
    wsuid_token = safe_token(wsuid, default="WS")
    st_tag = safe_token(str(args.stNo), default="ALL")
    op_tag = safe_token(str(args.opTc), default="ALL") if str(args.opTc).upper() != "ALL" else "ALL"
    t_hash = _hash_list([safe_token(t) for t in targets], n=10)

    prefix = (
        f"WSUID_{wsuid_token}_ST_{st_tag}_OPTC_{op_tag}"
        f"__MIMO_RF__HSEC_{int(horizon_sec)}__RSEC_{int(resample_sec)}__NLAG_{int(n_lags)}__TSET_{t_hash}"
    )
    model_path = os.path.join(out_dir, prefix + "__ALG_RF.pkl")
    meta_path = os.path.join(out_dir, prefix + "__meta.json")

    joblib.dump(pipe, model_path)

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "ws": {"pl_id": pl_id, "wc_id": wc_id, "ws_id": ws_id, "wsuid": wsuid},
        "stNo": str(args.stNo),
        "opTc": str(args.opTc),
        "segment_field": str(segment_field),
        "segment_policy": ctx_policy,
        "resample_sec": int(resample_sec),
        "horizon_sec": int(horizon_sec),
        "n_lags": int(n_lags),
        "targets": [str(t) for t in targets],
        "sensors_used": [str(s) for s in sensors_used],
        "feature_names": [str(x) for x in feature_names],
        "data_info": info,
        "model_path": model_path,
        "metrics": {
            **{k: v for k, v in eval_result.items() if k != "pipeline"},
            "baseline_mae_lag0_mean": float(np.nanmean(np.asarray(base_mae, dtype="float64"))) if base_mae else float("nan"),
            "model_mae_mean": float(np.nanmean(np.asarray(model_mae, dtype="float64"))) if model_mae else float("nan"),
            "lift_vs_lag0_mean": float(lift_mean),
            "accepted": bool(accepted),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("OK")
    print(f"Model: {model_path}")
    print(f"Meta : {meta_path}")
    print(f"Targets: {len(targets)}  SensorsUsed: {len(sensors_used)}")
    print(f"Lift(mean): {lift_mean}  Accepted: {accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
