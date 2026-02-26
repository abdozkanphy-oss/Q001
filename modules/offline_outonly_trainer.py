# modules/offline_outonly_trainer.py
"""
M2.2/M2.3: Offline OUT_ONLY trainer

Key points:
- Supports irregular sampling + retrospective dumps by anchoring reads to event-time (latest measurement_date)
- Evaluates against a strong baseline: lag0 persistence (predict y(t+h)=y(t))
- Option-2 readiness: can train UNI or MV feature modes (sweep decides fallback logic)
- Batch-holdout evaluation when a reliable segment_field exists; otherwise time split
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from modules.offline_jsonl_source import jsonl_probe_latest_dt, jsonl_to_rows, parse_dt_like_to_utc
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws
from modules.context_profiler import profile_context_from_rows
from modules.context_policy import select_context_policy
from modules.model_registry import safe_token



def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    """Return mask where all arrays are finite."""
    m = None
    for a in arrs:
        aa = np.asarray(a, dtype="float64")
        mm = np.isfinite(aa)
        m = mm if m is None else (m & mm)
    return m if m is not None else np.array([], dtype=bool)


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    m = _finite_mask(y_true, y_pred)
    if not np.any(m):
        return float("nan")
    return float(mean_absolute_error(y_true[m], y_pred[m]))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Version-safe RMSE without deprecated mean_squared_error(squared=...)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    m = _finite_mask(y_true, y_pred)
    if not np.any(m):
        return float("nan")
    mse = float(mean_squared_error(y_true[m], y_pred[m]))  # MSE
    return float(np.sqrt(mse))


def probe_latest_measurement_date(pl_id: int, wc_id: int, ws_id: int):
    """
    Best-effort fetch of the latest measurement_date for (pl,wc,ws).
    Assumes clustering order is DESC on measurement_date.
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



def _row_get(r: Any, name: str, default: Any = None) -> Any:
    """Get field from either a Cassandra row object or a dict row."""
    try:
        if isinstance(r, dict):
            return r.get(name, default)
        return getattr(r, name, default)
    except Exception:
        return default

def _norm_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"0", "none", "null", "nan"}:
        return None
    return s


def fetch_rows_by_ws(
    pl_id: int,
    wc_id: int,
    ws_id: int,
    *,
    days: int,
    limit: int = 50000,
    st_no: str = "ALL",
    op_tc: str = "ALL",
    jsonl: str = "",
    time_min: str = "",
    time_max: str = "",
) -> List[Any]:
    """
    Fetch rows from dw_tbl_raw_data_by_ws anchored to event-tim

    # JSONL source mode (Kafka dump recordings)
    if jsonl:
        # Determine time window.
        if time_min and time_max:
            start_dt = parse_dt_like_to_utc(time_min)
            end_dt = parse_dt_like_to_utc(time_max)
        else:
            latest = jsonl_probe_latest_dt(jsonl, pl_id=int(pl_id), wc_id=int(wc_id), ws_id=int(ws_id), st_no=str(st_no), op_tc=str(op_tc))
            end_dt = latest if latest is not None else datetime.utcnow().replace(tzinfo=timezone.utc)
            start_dt = end_dt - timedelta(days=int(days))

        # Convert to row-like dicts compatible with downstream conversions.
        return jsonl_to_rows(
            jsonl,
            pl_id=int(pl_id),
            wc_id=int(wc_id),
            ws_id=int(ws_id),
            start_dt=start_dt,
            end_dt=end_dt,
            limit=int(limit),
            st_no=str(st_no),
            op_tc=str(op_tc),
        )
e.
    """
    latest = probe_latest_measurement_date(pl_id, wc_id, ws_id)
    end_dt = pd.to_datetime(latest, utc=True).to_pydatetime() if latest is not None else datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(days))

    chunk_hours = 12
    per_chunk_limit = max(5000, int(limit // max(1, (days * 24) // chunk_hours)))

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
        if st_no and str(st_no).upper() != "ALL":
            q = q.filter(produced_stock_no=str(st_no))
        if op_tc and str(op_tc).upper() != "ALL":
            q = q.filter(operationtaskcode=str(op_tc))

        rows_all.extend(list(q.limit(int(take))))
        cur = nxt

    return rows_all


def rows_to_df(rows: List[Any], *, segment_field: str) -> pd.DataFrame:
    """
    Convert Cassandra rows into a normalized long dataframe:
      ts, sensor, val, seg (optional)
    """
    if not rows:
        return pd.DataFrame()

    recs = []
    for r in rows:
        try:
            ts = _row_get(r, "measurement_date", None)
            sensor = _row_get(r, "equipment_name", None)
            val = _row_get(r, "counter_reading", None)
            seg_val = None
            if segment_field and segment_field not in {"", "AUTO"}:
                seg_val = _row_get(r, segment_field, None)
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
    """
    Create session ids based on gaps between consecutive timestamps in a sorted index.
    """
    if len(ts_index) == 0:
        return pd.Series(dtype="object")
    s = pd.Series(ts_index, index=ts_index).sort_index()
    gaps = s.diff().dt.total_seconds().fillna(0.0)
    new_sess = (gaps > float(session_gap_sec)).astype(int)
    sess_id = new_sess.cumsum()
    return sess_id.astype(str)


def build_univariate_dataset(
    df: pd.DataFrame,
    *,
    target: str,
    resample_sec: int,
    horizon_sec: int,
    n_lags: int,
    segment_field: str,
    session_gap_sec: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str], Dict[str, Any]]:
    """
    Returns:
      X, y, base0, seg_ids (or None), feature_names, info
    """
    sdf = df[df["sensor"] == str(target)][["ts", "val", "seg"]].copy()
    if sdf.empty:
        return np.empty((0, 0)), np.empty((0,)), np.empty((0,)), None, [], {"reason": "no_target_rows"}

    sdf = sdf.sort_values("ts")
    obs_ts = pd.DatetimeIndex(sdf["ts"].tolist())
    freq = f"{int(resample_sec)}s"

    s = pd.Series(sdf["val"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(sdf["ts"])).sort_index()
    s = s[~s.index.duplicated(keep="last")]

    sr = s.resample(freq).last()

    ffill_limit = _auto_ffill_limit(obs_ts, resample_sec=resample_sec, n_lags=n_lags)
    max_gap_sec = _auto_max_gap_sec(obs_ts)

    idx = pd.Series(sr.index, index=sr.index)
    last_obs = idx.where(sr.notna()).ffill()
    age_sec = (idx - last_obs).dt.total_seconds()
    sr = sr.where(age_sec <= float(max_gap_sec))
    sr = sr.ffill(limit=int(ffill_limit))

    horizon_steps = max(1, int(round(float(horizon_sec) / float(resample_sec))))

    # Lagged features
    X_df = pd.DataFrame(index=sr.index)
    for k in range(1, int(n_lags) + 1):
        X_df[f"lag_{k}"] = sr.shift(k)

    # Target at +horizon
    y = sr.shift(-horizon_steps)

    # baseline lag0: current sr(t)
    base0 = sr

    # Build seg_ids aligned to sr index
    seg_ids = None
    if str(segment_field).upper() == "SESSION":
        sess = _build_session_ids(sr.index, session_gap_sec=int(session_gap_sec))
        seg_ids = sess.reindex(sr.index)
    else:
        # derive seg series from raw (event-time) seg values
        seg_raw = sdf.dropna(subset=["seg"]).set_index("ts")["seg"].astype(str)
        seg_raw = seg_raw[~seg_raw.index.duplicated(keep="last")]
        seg_r = seg_raw.resample(freq).last()
        # gap guard and light forward fill
        idx2 = pd.Series(seg_r.index, index=seg_r.index)
        last_obs2 = idx2.where(seg_r.notna()).ffill()
        age_sec2 = (idx2 - last_obs2).dt.total_seconds()
        seg_r = seg_r.where(age_sec2 <= float(max_gap_sec))
        seg_r = seg_r.ffill(limit=int(ffill_limit))
        seg_ids = seg_r.reindex(sr.index)

    # Filter supervised rows
    mask = y.notna()
    for c in X_df.columns:
        # allow NaNs (imputer in pipeline), but require at least some features
        pass

    X = X_df.loc[mask].to_numpy(dtype="float32")
    yv = y.loc[mask].to_numpy(dtype="float32")
    b0 = base0.loc[mask].to_numpy(dtype="float32")
    seg = seg_ids.loc[mask].astype(str).fillna("MISSING").to_numpy() if seg_ids is not None else None

    feature_names = list(X_df.columns)

    info = {
        "rows_raw": int(len(sdf)),
        "resampled_n": int(len(sr)),
        "n_supervised": int(mask.sum()),
        "horizon_steps": int(horizon_steps),
        "ffill_limit": int(ffill_limit),
        "max_gap_sec": int(max_gap_sec),
        "segment_field": str(segment_field),
        "distinct_segments": int(len(set(seg.tolist()))) if seg is not None else 0,
        "missing_segment_ratio": float(np.mean(seg == "MISSING")) if seg is not None and len(seg) else 0.0,
    }
    return X, yv, b0, seg, feature_names, info


def build_multivariate_dataset(
    df: pd.DataFrame,
    *,
    target: str,
    resample_sec: int,
    horizon_sec: int,
    n_lags: int,
    segment_field: str,
    session_gap_sec: int,
    max_sensors: int,
    min_sensor_non_nan: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str], List[str], Dict[str, Any]]:
    """
    Build a multivariate lag dataset by processing each segment independently to avoid cross-batch leakage.

    IMPORTANT: The feature space must be identical across all segments; otherwise numpy.vstack will fail.
    We therefore select a GLOBAL sensor set (per ws+stock+opTc scope), then for each segment:
      - pivot/resample
      - ensure all selected sensors exist (missing -> NaN)
      - build lag features in a deterministic order: for sensor in sensors_used, for k=1..n_lags

    Returns:
      X, y, base0, seg_ids (or None), feature_names, sensors_used, info
    """
    if df.empty:
        return np.empty((0, 0)), np.empty((0,)), np.empty((0,)), None, [], [], {"reason": "empty_df"}

    freq = f"{int(resample_sec)}s"
    horizon_steps = max(1, int(round(float(horizon_sec) / float(resample_sec))))

    # Determine segment IDs for each row
    seg_series = df["seg"].copy()
    if str(segment_field).upper() == "SESSION":
        # Build sessions from TARGET sensor timestamps
        tdf = df[df["sensor"] == str(target)][["ts"]].copy()
        if tdf.empty:
            return np.empty((0, 0)), np.empty((0,)), np.empty((0,)), None, [], [], {"reason": "no_target_rows"}
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

    # GLOBAL sensor selection (consistent feature space across segments)
    # Use raw row counts as a robust proxy; segment-level sparsity is handled by skipping segments where target is too sparse.
    sensor_counts = df2["sensor"].value_counts().to_dict()
    sensors_sorted = sorted(sensor_counts.keys(), key=lambda s: int(sensor_counts.get(s, 0)), reverse=True)

    sensors_used: List[str] = []
    for s in sensors_sorted:
        if int(sensor_counts.get(s, 0)) < int(min_sensor_non_nan):
            continue
        sensors_used.append(str(s))
        if len(sensors_used) >= int(max_sensors):
            break

    if str(target) not in sensors_used:
        sensors_used = [str(target)] + [s for s in sensors_used if s != str(target)]
        sensors_used = sensors_used[: max(1, int(max_sensors))]

    if not sensors_used:
        return np.empty((0, 0)), np.empty((0,)), np.empty((0,)), None, [], [], {"reason": "no_sensors_selected"}

    feature_names: List[str] = []
    for s in sensors_used:
        for k in range(1, int(n_lags) + 1):
            feature_names.append(f"{s}__lag_{k}")
    expected_dim = int(len(feature_names))

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    b0_parts: List[np.ndarray] = []
    seg_parts: List[np.ndarray] = []

    for seg_id in segments:
        sdf = df2[df2["seg_id"] == seg_id]
        if sdf.empty:
            continue

        piv = sdf.pivot_table(index="ts", columns="sensor", values="val", aggfunc="last")
        if piv.empty:
            continue
        piv = piv.sort_index()
        rr = piv.resample(freq).last()

        if str(target) not in rr.columns:
            continue

        # Ensure all global sensors exist (missing -> NaN), then reorder
        for s in sensors_used:
            if s not in rr.columns:
                rr[s] = np.nan
        rr = rr[sensors_used]

        # Require target to have enough support in this segment (before fill)
        tgt_non_nan = int(rr[str(target)].notna().sum())
        if tgt_non_nan < int(min_sensor_non_nan):
            continue

        # gap/ffill heuristics based on target obs times in this segment
        tgt_obs = sdf[sdf["sensor"] == str(target)]["ts"].dropna()
        obs_ts = pd.DatetimeIndex(pd.to_datetime(tgt_obs, utc=True))
        ffill_limit = _auto_ffill_limit(obs_ts, resample_sec=resample_sec, n_lags=n_lags)
        max_gap_sec = _auto_max_gap_sec(obs_ts)

        idx = pd.Series(rr.index, index=rr.index)
        for s in rr.columns:
            col = rr[s]
            last_obs = idx.where(col.notna()).ffill()
            age_sec = (idx - last_obs).dt.total_seconds()
            col = col.where(age_sec <= float(max_gap_sec))
            col = col.ffill(limit=int(ffill_limit))
            rr[s] = col

        y = rr[str(target)].shift(-horizon_steps)
        base0 = rr[str(target)]

        X_df = pd.DataFrame(index=rr.index)
        # Deterministic ordering matches feature_names above
        for s in sensors_used:
            for k in range(1, int(n_lags) + 1):
                X_df[f"{s}__lag_{k}"] = rr[s].shift(k)

        mask = y.notna()
        if int(mask.sum()) <= 0:
            continue

        X_seg = X_df.loc[mask].to_numpy(dtype="float32")
        if int(X_seg.shape[1]) != expected_dim:
            # Should never happen now; guard anyway
            raise RuntimeError(f"MV feature_dim mismatch: got={X_seg.shape[1]} expected={expected_dim}")

        y_seg = y.loc[mask].to_numpy(dtype="float32")
        b0_seg = base0.loc[mask].to_numpy(dtype="float32")
        seg_seg = np.array([str(seg_id)] * int(mask.sum()), dtype=object)

        X_parts.append(X_seg)
        y_parts.append(y_seg)
        b0_parts.append(b0_seg)
        seg_parts.append(seg_seg)

    if not X_parts:
        return np.empty((0, 0)), np.empty((0,)), np.empty((0,)), None, [], [], {"reason": "no_segments_built"}

    X = np.vstack(X_parts)
    yv = np.concatenate(y_parts)
    b0 = np.concatenate(b0_parts)
    seg = np.concatenate(seg_parts) if seg_parts else None

    info = {
        "horizon_steps": int(horizon_steps),
        "segment_field": str(segment_field),
        "segments_total": int(len(segments)),
        "segments_used": int(len(X_parts)),
        "sensors_used": list(sensors_used),
        "feature_dim": int(X.shape[1]),
        "n_supervised": int(len(yv)),
    }
    return X, yv, b0, seg, feature_names, sensors_used, info


def _fit_pipeline_rf(random_state: int = 42) -> Pipeline:
    # Model is robust to NaNs via imputer. This keeps train/infer consistent.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(
                n_estimators=250,
                random_state=int(random_state),
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ]
    )


def _eval_time_split(
    X: np.ndarray,
    y: np.ndarray,
    base0: np.ndarray,
    *,
    test_frac: float = 0.20,
    min_test: int = 20,
    baseline_perfect_eps: float = 1e-12,
) -> Dict[str, Any]:
    n = int(len(y))
    n_test = max(int(round(n * float(test_frac))), int(min_test))
    n_test = min(n_test, n - 1) if n > 1 else 0
    if n_test <= 0 or (n - n_test) <= 0:
        return {"ok": False, "reason": "too_few_points", "n": n}

    split = n - n_test
    X_tr, y_tr, b0_tr = X[:split], y[:split], base0[:split]
    X_te, y_te, b0_te = X[split:], y[split:], base0[split:]

    # baseline check (lag0 persistence). If baseline is perfect, skip training.
    m0 = _finite_mask(y_te, b0_te)
    n_finite0 = int(m0.sum())
    if n_finite0 < int(min_test):
        return {
            "ok": False,
            "reason": "insufficient_finite_test",
            "n_test": int(len(y_te)),
            "n_test_finite": int(n_finite0),
        }

    y0 = y_te[m0]
    b0_0 = b0_te[m0]
    b_mae0 = _safe_mae(y0, b0_0)
    b_rmse0 = _safe_rmse(y0, b0_0)

    if float(b_mae0) <= float(baseline_perfect_eps):
        return {
            "ok": True,
            "mode": "time_split",
            "skipped": True,
            "reason": "baseline_perfect",
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "n_test_finite": int(n_finite0),
            "model_mae": None,
            "model_rmse": None,
            "baseline_mae_lag0": float(b_mae0),
            "baseline_rmse_lag0": float(b_rmse0),
            "delta_mae_vs_lag0": 0.0,
            "lift_vs_lag0": float("nan"),
        }

    pipe = _fit_pipeline_rf()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    # filter to finite points where both baseline and prediction are defined
    m = _finite_mask(y_te, pred, b0_te)
    n_finite = int(m.sum())
    if n_finite < int(min_test):
        return {"ok": False, "reason": "insufficient_finite_test", "n_test": int(len(y_te)), "n_test_finite": int(n_finite)}

    y_te_f = y_te[m]
    pred_f = pred[m]
    b0_te_f = b0_te[m]

    mae = _safe_mae(y_te_f, pred_f)
    rmse = _safe_rmse(y_te_f, pred_f)

    b_mae = _safe_mae(y_te_f, b0_te_f)
    b_rmse = _safe_rmse(y_te_f, b0_te_f)

    delta = float(b_mae - mae) if (np.isfinite(b_mae) and np.isfinite(mae)) else float("nan")
    lift = float(delta / b_mae) if (np.isfinite(delta) and np.isfinite(b_mae) and b_mae > 0.0) else float("nan")

    return {
        "ok": True,
        "mode": "time_split",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_test_finite": int(len(y_te_f)),
        "model_mae": float(mae),
        "model_rmse": float(rmse),
        "baseline_mae_lag0": float(b_mae),
        "baseline_rmse_lag0": float(b_rmse),
        "delta_mae_vs_lag0": float(delta),
        "lift_vs_lag0": float(lift),
    }


def _eval_batch_holdout(
    X: np.ndarray,
    y: np.ndarray,
    base0: np.ndarray,
    seg: np.ndarray,
    *,
    holdout_k: int = 3,
    min_points_per_batch: int = 50,
    min_test: int = 20,
    baseline_perfect_eps: float = 1e-12,
) -> Dict[str, Any]:
    if seg is None or len(seg) != len(y):
        return {"ok": False, "reason": "no_seg"}

    seg = np.asarray(seg, dtype=object)
    # filter batches with enough points and not MISSING
    seg_vals, seg_counts = np.unique(seg, return_counts=True)
    ok_segs = [s for (s, c) in zip(seg_vals, seg_counts) if c >= int(min_points_per_batch) and str(s) != "MISSING"]

    holdout_k_cfg = int(holdout_k)
    effective_k = min(holdout_k_cfg, int(len(ok_segs)) - 1)  # need at least 1 train batch
    if effective_k < 1:
        return {"ok": False, "reason": "insufficient_batches", "batches_ok": int(len(ok_segs)), "holdout_k_cfg": holdout_k_cfg}

    # approximate ordering by last occurrence index
    last_idx = {}
    for i, s in enumerate(seg):
        if s in ok_segs:
            last_idx[str(s)] = i
    ok_segs_sorted = sorted(ok_segs, key=lambda s: int(last_idx.get(str(s), -1)))
    test_segs = [str(s) for s in ok_segs_sorted[-int(effective_k):]]

    is_test = np.array([str(s) in set(test_segs) for s in seg], dtype=bool)
    # ensure test size
    if int(is_test.sum()) < int(min_test) or int((~is_test).sum()) < 1:
        return {"ok": False, "reason": "too_few_test_points", "n_test": int(is_test.sum())}

    X_tr, y_tr = X[~is_test], y[~is_test]
    X_te, y_te = X[is_test], y[is_test]
    b0_te = base0[is_test]

    # baseline check (lag0). If baseline is perfect, skip training.
    m0 = _finite_mask(y_te, b0_te)
    n_finite0 = int(m0.sum())
    if n_finite0 < int(min_test):
        return {"ok": False, "reason": "insufficient_finite_test", "n_test": int(len(y_te)), "n_test_finite": int(n_finite0)}

    y0 = y_te[m0]
    b0_0 = b0_te[m0]
    b_mae0 = _safe_mae(y0, b0_0)
    b_rmse0 = _safe_rmse(y0, b0_0)

    if float(b_mae0) <= float(baseline_perfect_eps):
        return {
            "ok": True,
            "mode": "batch_holdout",
            "skipped": True,
            "reason": "baseline_perfect",
            "holdout_k": int(effective_k),
            "holdout_k_cfg": int(holdout_k_cfg),
            "test_segs": test_segs,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "n_test_finite": int(n_finite0),
            "model_mae": None,
            "model_rmse": None,
            "baseline_mae_lag0": float(b_mae0),
            "baseline_rmse_lag0": float(b_rmse0),
            "delta_mae_vs_lag0": 0.0,
            "lift_vs_lag0": float("nan"),
            "per_holdout": [],
        }

    pipe = _fit_pipeline_rf()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    m = _finite_mask(y_te, pred, b0_te)
    n_finite = int(m.sum())
    if n_finite < int(min_test):
        return {"ok": False, "reason": "insufficient_finite_test", "n_test": int(len(y_te)), "n_test_finite": int(n_finite)}

    y_te_f = y_te[m]
    pred_f = pred[m]
    b0_te_f = b0_te[m]

    mae = _safe_mae(y_te_f, pred_f)
    rmse = _safe_rmse(y_te_f, pred_f)

    b_mae = _safe_mae(y_te_f, b0_te_f)
    b_rmse = _safe_rmse(y_te_f, b0_te_f)

    # per-batch breakdown (optional)
    per = []
    for s in test_segs:
        idx = np.array([str(x) == str(s) for x in seg], dtype=bool) & is_test
        if int(idx.sum()) <= 0:
            continue
        y_s = y[idx]
        p_s = pred[np.where(idx[is_test])[0]]  # mapping from global test mask to pred index
        b_s = base0[idx]
        per.append(
            {
                "seg_id": str(s),
                "n": int(len(y_s)),
                "model_mae": float(_safe_mae(y_s, p_s)),
                "baseline_mae_lag0": float(_safe_mae(y_s, b_s)),
            }
        )

    return {
        "ok": True,
        "mode": "batch_holdout",
        "holdout_k": int(effective_k),
        "holdout_k_cfg": int(holdout_k_cfg),
        "test_segs": test_segs,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_test_finite": int(len(y_te_f)),
        "model_mae": mae,
        "model_rmse": rmse,
        "baseline_mae_lag0": b_mae,
        "baseline_rmse_lag0": b_rmse,
        "delta_mae_vs_lag0": float(b_mae - mae),
        "lift_vs_lag0": float((b_mae - mae) / b_mae) if (b_mae and b_mae > 0.0) else float("nan"),
        "per_holdout": per,
    }


def choose_targets_by_trainability(
    df: pd.DataFrame,
    *,
    candidates: List[str],
    resample_sec: int,
    horizon_sec: int,
    n_lags: int,
    min_rows: int,
    segment_field: str,
    session_gap_sec: int,
) -> List[str]:
    """
    Deterministic target selection: prefer candidates with enough supervised points.
    """
    scored = []
    for t in candidates:
        X, y, _, _, _, info = build_univariate_dataset(
            df,
            target=t,
            resample_sec=resample_sec,
            horizon_sec=horizon_sec,
            n_lags=n_lags,
            segment_field=segment_field,
            session_gap_sec=session_gap_sec,
        )
        n_sup = int(info.get("n_supervised") or 0)
        if n_sup >= int(min_rows):
            scored.append((n_sup, t))
    scored.sort(reverse=True)
    return [t for (_, t) in scored]


def main():
    ap = argparse.ArgumentParser(description="Offline OUT_ONLY trainer (UNI/MV) with robust evaluation.")
    ap.add_argument("--plId", type=int, required=True)
    ap.add_argument("--wcId", type=int, required=True)
    ap.add_argument("--wsId", type=int, required=True)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--stNo", default="ALL")
    ap.add_argument("--opTc", default="ALL")

    ap.add_argument("--jsonl", default="", help="Optional JSONL path(s) to train without Cassandra (comma-separated).")
    ap.add_argument("--time_min", default="", help="Optional start datetime for JSONL source (ISO-ish).")
    ap.add_argument("--time_max", default="", help="Optional end datetime for JSONL source (ISO-ish).")

    ap.add_argument("--min_rows", type=int, default=80)
    ap.add_argument("--targets", default="", help="Comma separated target sensor names; empty => auto")
    ap.add_argument("--n_lags", type=int, default=6)
    ap.add_argument("--resample_sec", type=int, default=60)
    ap.add_argument("--horizon_sec", type=int, default=60)
    ap.add_argument("--segment_field", default="AUTO", help="AUTO | SESSION | specific field name")

    # MV controls
    ap.add_argument("--feature_mode", default="univariate", choices=["univariate", "multivariate"])
    ap.add_argument("--max_sensors", type=int, default=50, help="MV: cap number of sensors used")
    ap.add_argument("--min_sensor_non_nan", type=int, default=10, help="MV: min non-NaN per sensor after resample")

    # eval / acceptance
    ap.add_argument("--eval_mode", default="auto", choices=["auto", "time_split", "batch_holdout"])
    ap.add_argument("--holdout_k", type=int, default=3)
    ap.add_argument("--min_points_per_batch", type=int, default=50)
    ap.add_argument("--min_test", type=int, default=20)
    ap.add_argument("--baseline_perfect_eps", type=float, default=1e-12, help="Skip training when lag0 baseline MAE <= eps")
    ap.add_argument("--accept_min_lift", type=float, default=0.0)

    # sessionization
    ap.add_argument("--session_gap_sec", type=int, default=6 * 3600)

    # output
    ap.add_argument("--out_dir", default="./models/offline_outonly")
    ap.add_argument("--run_id", default="", help="Optional external run_id for sweep correlation")

    args = ap.parse_args()

    out_dir = str(args.out_dir or "./models/offline_outonly")
    os.makedirs(out_dir, exist_ok=True)

    run_id = str(args.run_id or "").strip()
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Fetch
    rows = fetch_rows_by_ws(
        pl_id=int(args.plId),
        wc_id=int(args.wcId),
        ws_id=int(args.wsId),
        days=int(args.days),
        limit=int(args.limit),
        st_no=str(args.stNo),
        op_tc=str(args.opTc),
        jsonl=str(args.jsonl),
        time_min=str(args.time_min),
        time_max=str(args.time_max),
    )
    if not rows:
        print("No rows fetched.")
        return 2

    # Segment field selection (AUTO -> profiler/policy)
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

    # Target list
    candidates = []
    if args.targets.strip():
        candidates = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        # default: top sensors by raw count
        vc = df["sensor"].value_counts()
        candidates = vc.head(50).index.tolist()

    # Filter to deterministic trainable subset using univariate builder
    selected = choose_targets_by_trainability(
        df,
        candidates=candidates,
        resample_sec=int(args.resample_sec),
        horizon_sec=int(args.horizon_sec),
        n_lags=int(args.n_lags),
        min_rows=int(args.min_rows),
        segment_field=str(segment_field),
        session_gap_sec=int(args.session_gap_sec),
    )
    if not selected:
        print("No trainable targets found (min_rows not met).")
        return 3

    wsuid_token = safe_token(f"{args.plId}_WC{args.wcId}_WS{args.wsId}", default="WS")
    st_tag = safe_token(str(args.stNo), default="ALL") if str(args.stNo).upper() != "ALL" else "ALL"
    op_tag = safe_token(str(args.opTc), default="ALL") if str(args.opTc).upper() != "ALL" else "ALL"

    feature_mode = "UNI" if args.feature_mode == "univariate" else "MV"
    resample_sec = int(args.resample_sec)
    horizon_sec = int(args.horizon_sec)
    n_lags = int(args.n_lags)

    summary = {
        "run_id": run_id,
        "ts_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "ws": {"pl_id": int(args.plId), "wc_id": int(args.wcId), "ws_id": int(args.wsId)},
        "st_no": str(args.stNo),
        "op_tc": str(args.opTc),
        "segment_field": str(segment_field),
        "segment_policy": ctx_policy,
        "feature_mode": feature_mode,
        "resample_sec": resample_sec,
        "horizon_sec": horizon_sec,
        "n_lags": n_lags,
        "targets": [],
    }

    # Train per target
    for target in selected:
        if feature_mode == "UNI":
            X, y, b0, seg, feature_names, info = build_univariate_dataset(
                df,
                target=target,
                resample_sec=resample_sec,
                horizon_sec=horizon_sec,
                n_lags=n_lags,
                segment_field=str(segment_field),
                session_gap_sec=int(args.session_gap_sec),
            )
            sensors_used = [str(target)]
        else:
            X, y, b0, seg, feature_names, sensors_used, info = build_multivariate_dataset(
                df,
                target=target,
                resample_sec=resample_sec,
                horizon_sec=horizon_sec,
                n_lags=n_lags,
                segment_field=str(segment_field),
                session_gap_sec=int(args.session_gap_sec),
                max_sensors=int(args.max_sensors),
                min_sensor_non_nan=int(args.min_sensor_non_nan),
            )

        n_sup = int(len(y))
        if n_sup < int(args.min_rows):
            summary["targets"].append(
                {
                    "target": target,
                    "ok": False,
                    "reason": "insufficient_supervised_rows",
                    "n_supervised": n_sup,
                    "info": info,
                }
            )
            continue

        # Decide eval mode
        eval_mode = str(args.eval_mode)
        eval_result: Dict[str, Any]
        if eval_mode == "auto":
            if seg is not None and str(segment_field).upper() != "SESSION":
                eval_result = _eval_batch_holdout(
                    X, y, b0, seg,
                    holdout_k=int(args.holdout_k),
                    min_points_per_batch=int(args.min_points_per_batch),
                    min_test=int(args.min_test),
                    baseline_perfect_eps=float(args.baseline_perfect_eps),
                )
                if not eval_result.get("ok"):
                    eval_result = _eval_time_split(X, y, b0, min_test=int(args.min_test), baseline_perfect_eps=float(args.baseline_perfect_eps))
            else:
                # SESSION or no seg => time split
                eval_result = _eval_time_split(X, y, b0, min_test=int(args.min_test), baseline_perfect_eps=float(args.baseline_perfect_eps))
        elif eval_mode == "batch_holdout":
            eval_result = _eval_batch_holdout(
                X, y, b0, seg,
                holdout_k=int(args.holdout_k),
                min_points_per_batch=int(args.min_points_per_batch),
                min_test=int(args.min_test),
                baseline_perfect_eps=float(args.baseline_perfect_eps),
            )
            if not eval_result.get("ok"):
                # fallback to time split to still emit diagnostics
                eval_result = _eval_time_split(X, y, b0, min_test=int(args.min_test), baseline_perfect_eps=float(args.baseline_perfect_eps))
        else:
            eval_result = _eval_time_split(X, y, b0, min_test=int(args.min_test), baseline_perfect_eps=float(args.baseline_perfect_eps))

        ok_eval = bool(eval_result.get("ok"))
        model_mae = eval_result.get("model_mae")
        baseline_mae = eval_result.get("baseline_mae_lag0")
        lift = eval_result.get("lift_vs_lag0")

        accepted = False
        reject_reason = None
        if ok_eval and model_mae is not None and baseline_mae is not None:
            n_test_i = int(eval_result.get("n_test") or 0)
            if n_test_i < int(args.min_test):
                reject_reason = "insufficient_test"
            else:
                lift_f = float(lift or 0.0)
                if lift_f < float(args.accept_min_lift):
                    reject_reason = "worse_than_baseline" if lift_f < 0.0 else "lift_below_threshold"
                else:
                    accepted = True
        elif not ok_eval:
            reject_reason = str(eval_result.get("reason") or "eval_failed")

        # If evaluation failed, do not train / persist an artifact.
        if not ok_eval:
            reason = str(eval_result.get("reason") or "eval_failed")
            print(f"[SKIP] target={target} FM={feature_mode} eval_failed reason={reason}")
            summary["targets"].append(
                {
                    "target": target,
                    "ok": False,
                    "accepted": False,
                    "skipped": False,
                    "skip_reason": reason,
                    "reject_reason": reason,
                    "model_mae": None,
                    "baseline_mae_lag0": None if baseline_mae is None else float(baseline_mae),
                    "lift_vs_lag0": None if lift is None else float(lift),
                    "eval_mode": str(eval_result.get("mode") or ""),
                    "n_test": int(eval_result.get("n_test") or 0),
                    "model_path": None,
                    "meta_path": None,
                    "eval": eval_result,
                }
            )
            continue

        # Baseline-perfect skip (or other intentional skips)
        if bool(eval_result.get("skipped")):
            reason = str(eval_result.get("reason") or "skipped")
            print(f"[SKIP] target={target} FM={feature_mode} skipped reason={reason} base_mae={baseline_mae}")
            summary["targets"].append(
                {
                    "target": target,
                    "ok": True,
                    "accepted": False,
                    "skipped": True,
                    "skip_reason": reason,
                    "reject_reason": reason,
                    "model_mae": None,
                    "baseline_mae_lag0": None if baseline_mae is None else float(baseline_mae),
                    "lift_vs_lag0": None if lift is None else float(lift),
                    "eval_mode": str(eval_result.get("mode") or ""),
                    "n_test": int(eval_result.get("n_test") or 0),
                    "model_path": None,
                    "meta_path": None,
                    "eval": eval_result,
                }
            )
            continue

        # Fit final model on all supervised data
        pipe = _fit_pipeline_rf()
        pipe.fit(X, y)

        key_prefix = (
            f"WSUID_{wsuid_token}_ST_{st_tag}_OPTC_{op_tag}__OUTONLY__TGT_{safe_token(target)}"
            f"__HSEC_{horizon_sec}__RSEC_{resample_sec}__FM_{feature_mode}"
        )
        model_path = os.path.join(out_dir, f"{key_prefix}__ALG_RF.pkl")
        meta_path = os.path.join(out_dir, f"{key_prefix}__meta.json")

        joblib.dump(pipe, model_path)

        metrics = {
            "n_supervised": int(n_sup),
            "n_train": int(eval_result.get("n_train") or 0),
            "n_test": int(eval_result.get("n_test") or 0),
            "model_mae": None if model_mae is None else float(model_mae),
            "model_rmse": None if eval_result.get("model_rmse") is None else float(eval_result.get("model_rmse")),
            "baseline_mae_lag0": None if baseline_mae is None else float(baseline_mae),
            "baseline_rmse_lag0": None if eval_result.get("baseline_rmse_lag0") is None else float(eval_result.get("baseline_rmse_lag0")),
            "lift_vs_lag0": None if lift is None else float(lift),
            "accepted": bool(accepted),
            "reject_reason": None if bool(accepted) else (str(reject_reason) if reject_reason is not None else "not_accepted"),
            "accept_min_lift": float(args.accept_min_lift),
            "eval_mode": str(eval_result.get("mode") or ""),
            # legacy compat:
            "mae": None if model_mae is None else float(model_mae),
            "feature_names": list(feature_names),
        }

        meta = {
            "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "wsuid_token": wsuid_token,
            "pl_id": int(args.plId),
            "wc_id": int(args.wcId),
            "ws_id": int(args.wsId),
            "st_no": str(args.stNo),
            "op_tc": str(args.opTc),
            "target_sensor": str(target),
            "feature_mode": feature_mode,
            "sensors_used": list(sensors_used),
            "resample_sec": int(resample_sec),
            "horizon_sec": int(horizon_sec),
            "n_lags": int(n_lags),
            "segment_field": str(segment_field),
            "segment_policy": ctx_policy,
            "data_info": info,
            "eval": eval_result,
            "metrics": metrics,
            "model_path": model_path,
            "meta_path": meta_path,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[OK] target={target} FM={feature_mode} accepted={accepted} mae={metrics['model_mae']} base_mae={metrics['baseline_mae_lag0']} lift={metrics['lift_vs_lag0']} meta={meta_path}")

        summary["targets"].append(
            {
                "target": target,
                "ok": True,
                "accepted": bool(accepted),
                "reject_reason": metrics.get("reject_reason"),
                "model_mae": metrics["model_mae"],
                "baseline_mae_lag0": metrics["baseline_mae_lag0"],
                "lift_vs_lag0": metrics["lift_vs_lag0"],
                "eval_mode": metrics["eval_mode"],
                "n_test": metrics["n_test"],
                "model_path": model_path,
                "meta_path": meta_path,
            }
        )

    # Write run summary (for sweep / orchestrator)
    summary_path = os.path.join(out_dir, f"run_outonly_{wsuid_token}_{st_tag}_{op_tag}_{feature_mode}_{run_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print lightweight acceptance totals
    tried = [t for t in summary["targets"] if t.get("ok")]
    accepted_n = sum(1 for t in tried if bool(t.get("accepted")))
    print(f"RUN_SUMMARY saved: {summary_path} targets_ok={len(tried)} accepted={accepted_n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------
# Backward-compat helpers (used by Phase2 offline trainer)
# ---------------------------------------------------------------------
def fetch_output_rows_by_ws(
    pl_id: int,
    wc_id: int,
    ws_id: int,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    allow_filtering: bool = False,
    st_no: Optional[str] = None,
    op_tc: Optional[str] = None,
    chunk_hours: int = 6,
    show_progress: bool = False,
    jsonl: str = "",
):
    """Legacy helper: fetch rows for a time window (wall-clock anchored)."""

    if jsonl:
        return jsonl_to_rows(
            jsonl,
            pl_id=int(pl_id),
            wc_id=int(wc_id),
            ws_id=int(ws_id),
            start_dt=start_dt if start_dt.tzinfo is not None else start_dt.replace(tzinfo=timezone.utc),
            end_dt=end_dt if end_dt.tzinfo is not None else end_dt.replace(tzinfo=timezone.utc),
            limit=int(limit),
            st_no=str(st_no or "ALL"),
            op_tc=str(op_tc or "ALL"),
        )

    rows_all: List[Any] = []
    cur = start_dt
    while cur < end_dt and len(rows_all) < int(limit):
        nxt = min(end_dt, cur + timedelta(hours=int(chunk_hours)))
        take = int(min(int(limit) - len(rows_all), 200000))
        q = dw_tbl_raw_data_by_ws.objects.filter(
            plant_id=int(pl_id),
            work_center_id=int(wc_id),
            work_station_id=int(ws_id),
            measurement_date__gte=cur,
            measurement_date__lt=nxt,
        )
        if st_no is not None:
            q = q.filter(produced_stock_no=str(st_no))
        if op_tc is not None:
            q = q.filter(operationtaskcode=str(op_tc))
        if allow_filtering:
            q = q.allow_filtering()    
        from cassandra import InvalidRequest

        try:
            chunk = list(q.limit(int(take)))
        except InvalidRequest as e:
            msg = str(e)
            if (not allow_filtering) and ("ALLOW FILTERING" in msg or "data filtering" in msg):
                print("CASSANDRA_ALLOW_FILTERING_RETRY: query requires ALLOW FILTERING; retrying with allow_filtering=True")
                q2 = q.allow_filtering()
                chunk = list(q2.limit(int(take)))
            else:
                raise

        rows_all.extend(chunk)
        if show_progress:
            print(f"FETCH chunk [{cur}..{nxt}) got={len(chunk)} total={len(rows_all)}")
        cur = nxt
    return rows_all


def rows_to_wide_df(rows: List[Any]) -> pd.DataFrame:
    """Legacy helper: convert dw_tbl_raw_data_by_ws rows to a wide DataFrame indexed by ts."""
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        try:
            recs.append(
                {
                    "ts": getattr(r, "measurement_date", None),
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
    df["sensor"] = df["sensor"].fillna("").astype(str)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="ts", columns="sensor", values="val", aggfunc="last").sort_index()
    return wide

