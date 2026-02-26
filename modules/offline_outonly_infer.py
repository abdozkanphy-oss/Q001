# modules/offline_outonly_infer.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from modules.model_registry import OutOnlyArtifact


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


def _parse_feature_names(feature_names: List[str]) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Returns (mode, specs)
      - mode='UNI': specs=[('', lag_k)] for lag_1..lag_n
      - mode='MV' : specs=[(sensor_name, lag_k)] using name format '<sensor>__lag_<k>'
    """
    if not feature_names:
        return ("UNI", [])

    names = [str(x) for x in feature_names if x is not None]
    if all(n.startswith("lag_") for n in names):
        specs = []
        for n in names:
            try:
                k = int(n.split("_", 1)[1])
            except Exception:
                continue
            specs.append(("", k))
        return ("UNI", specs)

    specs = []
    for n in names:
        m = n.rsplit("__lag_", 1)
        if len(m) != 2:
            continue
        s_name = m[0]
        try:
            k = int(m[1])
        except Exception:
            continue
        specs.append((s_name, k))
    return ("MV", specs)


def predict_outonly_from_seed(
    artifact: OutOnlyArtifact,
    target_var: str,
    seed_history,
    now_ts: datetime,
    current_value: Optional[float] = None,
) -> Optional[float]:
    """
    Predict using an offline OUT_ONLY model.

    The saved model is a sklearn Pipeline(imputer+rf), so NaNs in features are acceptable.
    We still require at least some non-NaN feature signal; otherwise return None.

    IMPORTANT:
      - resampling cadence must match what the model was trained on.
      - use artifact.resample_sec (coming from meta), do NOT force 60.
    """
    feature_names = list(getattr(artifact, "feature_names", []) or [])
    mode, specs = _parse_feature_names(feature_names)

    if mode == "UNI":
        n_lags = len(feature_names) if feature_names else 10
        required_sensors = [target_var]
        lag_specs = [(target_var, k) for k in range(1, n_lags + 1)]
    else:
        required_sensors = sorted({s for (s, _) in specs if s})
        if not required_sensors:
            n_lags = len(feature_names) if feature_names else 10
            required_sensors = [target_var]
            lag_specs = [(target_var, k) for k in range(1, n_lags + 1)]
        else:
            lag_specs = [(s, k) for (s, k) in specs]

    # Collect history into per-sensor series
    ts_by_sensor: Dict[str, List[pd.Timestamp]] = {s: [] for s in required_sensors}
    val_by_sensor: Dict[str, List[float]] = {s: [] for s in required_sensors}

    if seed_history:
        for ts, row in seed_history:
            if not isinstance(row, dict):
                continue
            try:
                t = pd.to_datetime(ts)
            except Exception:
                continue
            for s in required_sensors:
                if s not in row:
                    continue
                v = row.get(s)
                try:
                    fv = float(v)
                except Exception:
                    continue
                if not np.isfinite(fv):
                    continue
                ts_by_sensor[s].append(t)
                val_by_sensor[s].append(fv)

    if current_value is not None and target_var in ts_by_sensor:
        try:
            fv = float(current_value)
            if np.isfinite(fv):
                ts_by_sensor[target_var].append(pd.to_datetime(now_ts))
                val_by_sensor[target_var].append(fv)
        except Exception:
            pass

    resample_sec = int(getattr(artifact, "resample_sec", 60) or 60)
    freq = f"{int(resample_sec)}s"

    # Heuristics based on target timestamps
    tgt_ts = pd.DatetimeIndex(ts_by_sensor.get(target_var, []))
    if len(tgt_ts) < 3:
        any_ts = []
        for s in required_sensors:
            any_ts.extend(ts_by_sensor.get(s, []))
        tgt_ts = pd.DatetimeIndex(any_ts)

    if len(tgt_ts) < 3:
        return None

    max_lag = 0
    for _, k in lag_specs:
        max_lag = max(max_lag, int(k))
    ffill_limit = _auto_ffill_limit(tgt_ts, resample_sec=resample_sec, n_lags=max_lag)
    max_gap_sec = _auto_max_gap_sec(tgt_ts)

    series_by_sensor: Dict[str, pd.Series] = {}
    for s in required_sensors:
        ts_list = ts_by_sensor.get(s, [])
        val_list = val_by_sensor.get(s, [])
        if len(ts_list) < 2:
            series_by_sensor[s] = pd.Series(dtype="float64")
            continue

        sr = pd.Series(val_list, index=pd.DatetimeIndex(ts_list)).sort_index()
        sr = sr[~sr.index.duplicated(keep="last")]
        rr = sr.resample(freq).last()

        idx = pd.Series(rr.index, index=rr.index)
        last_obs = idx.where(rr.notna()).ffill()
        age_sec = (idx - last_obs).dt.total_seconds()
        rr = rr.where(age_sec <= float(max_gap_sec))
        rr = rr.ffill(limit=int(ffill_limit))

        series_by_sensor[s] = rr

    feats = []
    non_nan = 0
    for s, k in lag_specs:
        rr = series_by_sensor.get(s)
        v = np.nan
        if rr is not None and len(rr) > 0:
            try:
                v0 = rr.shift(int(k)).iloc[-1]
                if v0 is not None and np.isfinite(v0):
                    v = float(v0)
            except Exception:
                v = np.nan
        if np.isfinite(v):
            non_nan += 1
        feats.append(v)

    if non_nan == 0:
        return None

    X = np.array(feats, dtype="float32").reshape(1, -1)

    try:
        model = joblib.load(artifact.model_path)
        y_hat = model.predict(X)
        return float(y_hat[0])
    except Exception:
        return None
