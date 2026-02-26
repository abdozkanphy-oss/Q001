# modules/offline_mimo_rf_trainer.py
"""M5.0: Ultimate Offline MIMO Trainer & AutoML.

Modes
-----
1. Standard Mode: Trains a single configuration with optional GridSearch tuning.
2. AutoML Mode:   Searches for the best Data (Resample/Lags) & Model configuration
                  using strict Batch Holdout validation.

Features
--------
- Vectorized row processing (High Performance).
- Strict UTC timestamps.
- Model Factory: RF, HGB, XGBoost.
- AutoML Engine: Loops through Resample Rates, Lags, and Models.
- Artifacts: Saves the Winner Model + Leaderboard JSON.

CLI Examples
------------
1. AutoML Search (Best Practice):
   python -m modules.offline_mimo_rf_trainer --plId 149 --stNo "Antares PV" --automl --search_resample "30,60" --search_lags "6,12"

2. Manual Single Run:
   python -m modules.offline_mimo_rf_trainer --plId 149 --stNo "Antares PV" --model_type RF --tune
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Sklearn & Models
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Optional XGBoost Support
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# --- Helper Functions ---

def _get_dw_model():
    """Lazy import to avoid hard Cassandra dependency."""
    from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws
    return dw_tbl_raw_data_by_ws

_WSUID_RE = re.compile(r"^(?P<pl>\d+)_WC(?P<wc>\d+)_WS(?P<ws>\d+)$")

def ensure_utc(ts: Any) -> Optional[pd.Timestamp]:
    if ts is None: return None
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(t): return None
    return t

def _parse_wsuid(wsuid: str) -> Optional[Tuple[int, int, int]]:
    if not wsuid: return None
    m = _WSUID_RE.match(str(wsuid).strip())
    if not m: return None
    return int(m.group("pl")), int(m.group("wc")), int(m.group("ws"))

def _norm_id(v: Any) -> Optional[str]:
    if v is None: return None
    s = str(v).strip()
    if not s or s.lower() in {"0", "none", "null", "nan"}: return None
    return s

def _hash_list(items: List[str], *, n: int = 10) -> str:
    h = hashlib.sha1("|".join(items).encode("utf-8")).hexdigest()
    return h[: max(6, int(n))]

def probe_latest_measurement_date(pl_id, wc_id, ws_id):
    try:
        dw_tbl = _get_dw_model()
        q = dw_tbl.objects.filter(plant_id=int(pl_id), work_center_id=int(wc_id), work_station_id=int(ws_id)).limit(1)
        rows = list(q)
        return getattr(rows[0], "measurement_date", None) if rows else None
    except Exception:
        return None

def fetch_rows_by_ws_time_range(pl_id, wc_id, ws_id, *, time_min, time_max, days, limit=50000) -> List[Any]:
    dw_tbl = _get_dw_model()
    latest = probe_latest_measurement_date(pl_id, wc_id, ws_id)
    end_dt = ensure_utc(latest).to_pydatetime() if latest else datetime.now(pd.Timestamp.utcnow().tz)
    
    if time_max: end_dt = min(end_dt, time_max)
    start_dt = time_min if time_min else end_dt - timedelta(days=int(days))
    
    if start_dt >= end_dt: return []

    chunk_hours = 12
    span_hours = max(1.0, (end_dt - start_dt).total_seconds() / 3600.0)
    per_chunk_limit = max(2000, int(int(limit) // max(1, int(span_hours / chunk_hours))))

    rows_all = []
    cur = start_dt
    while cur < end_dt and len(rows_all) < int(limit):
        nxt = min(end_dt, cur + timedelta(hours=chunk_hours))
        take = min(per_chunk_limit, int(limit) - len(rows_all))
        q = dw_tbl.objects.filter(
            plant_id=int(pl_id), work_center_id=int(wc_id), work_station_id=int(ws_id),
            measurement_date__gte=cur, measurement_date__lt=nxt
        )
        rows_all.extend(list(q.limit(int(take))))
        cur = nxt
    return rows_all

def filter_rows_by_stock_op(rows, *, st_no, op_tc) -> List[Any]:
    if not rows: return []
    st_no = str(st_no or "ALL").strip()
    op_tc = str(op_tc or "ALL").strip()
    st_all, op_all = (st_no.upper() == "ALL" or st_no == ""), (op_tc.upper() == "ALL" or op_tc == "")
    out = []
    for r in rows:
        try:
            if not st_all:
                st1 = getattr(r, "produced_stock_no", None)
                st2 = getattr(r, "produced_stock_name", None)
                if str(st1) != st_no and str(st2) != st_no: continue
            if not op_all:
                op1 = getattr(r, "operationtaskcode", None)
                if str(op1) != op_tc: continue
            out.append(r)
        except Exception: continue
    return out

def rows_to_df(rows, *, segment_field) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    has_seg = (segment_field and segment_field not in {"", "AUTO"})
    data = []
    for r in rows:
        try:
            ts = r.measurement_date
            sensor = r.equipment_name
            val = r.counter_reading
            seg = getattr(r, segment_field, None) if has_seg else None
            data.append((ts, sensor, val, seg))
        except AttributeError:
            data.append((
                getattr(r, "measurement_date", None), getattr(r, "equipment_name", None),
                getattr(r, "counter_reading", None), getattr(r, segment_field, None) if has_seg else None
            ))
    df = pd.DataFrame(data, columns=["ts", "sensor", "val", "seg"])
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df["sensor"] = df["sensor"].fillna("").astype(str)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])
    if "seg" in df.columns: df["seg"] = df["seg"].map(_norm_id)
    return df

def _build_session_ids(ts_index, *, session_gap_sec) -> pd.Series:
    if len(ts_index) == 0: return pd.Series(dtype="object")
    s = pd.Series(ts_index, index=ts_index).sort_index()
    gaps = s.diff().dt.total_seconds().fillna(0.0)
    sess_id = (gaps > float(session_gap_sec)).astype(int).cumsum()
    return sess_id.astype(str)

def _auto_ffill_limit(obs_ts, resample_sec, n_lags, cap=600):
    if len(obs_ts) < 2: return min(cap, n_lags + 1)
    gaps = obs_ts.to_series().diff().dt.total_seconds().dropna()
    if len(gaps) == 0: return min(cap, n_lags + 1)
    buckets = max(1, int(round(float(gaps.median()) / float(resample_sec))))
    return min(cap, max(n_lags + 1, buckets))

def build_multitarget_dataset(df, *, targets, resample_sec, horizon_sec, n_lags, segment_field, session_gap_sec, max_sensors, min_sensor_non_nan):
    if df is None or df.empty: return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "empty_df"}
    
    targets = [str(t) for t in targets if str(t).strip()]
    targets = list(dict.fromkeys(targets))
    freq = f"{int(resample_sec)}s"
    horizon_steps = max(1, int(round(float(horizon_sec) / float(resample_sec))))

    if str(segment_field).upper() == "SESSION":
        tdf = df[df["sensor"].isin(targets)][["ts"]].copy().sort_values("ts")
        if tdf.empty: return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_target_rows"}
        sess_ids = _build_session_ids(pd.DatetimeIndex(tdf["ts"]), session_gap_sec=int(session_gap_sec))
        ts_sorted = pd.DatetimeIndex(tdf["ts"])
        sess_by_ts = pd.Series(sess_ids.values, index=ts_sorted)
        sess_by_ts = sess_by_ts[~sess_by_ts.index.duplicated(keep="last")].sort_index()
        all_ts = pd.DatetimeIndex(df["ts"])
        pos = sess_by_ts.index.searchsorted(all_ts, side="right") - 1
        valid_pos = (pos >= 0)
        out = np.full(len(df), "MISSING", dtype=object)
        if valid_pos.any(): out[valid_pos] = sess_by_ts.iloc[pos[valid_pos]].values
        seg_series = pd.Series(out, index=df.index)
    else:
        seg_series = df["seg"].copy() if "seg" in df.columns else pd.Series([None] * len(df))
        seg_series = seg_series.fillna("MISSING").astype(str)

    df2 = df.copy()
    df2["seg_id"] = seg_series
    segments = df2["seg_id"].unique().tolist()

    sensor_counts = df2["sensor"].value_counts().to_dict()
    sensors_sorted = sorted(sensor_counts.keys(), key=lambda s: int(sensor_counts.get(s, 0)), reverse=True)
    sensors_used = []
    for s in sensors_sorted:
        if int(sensor_counts.get(s, 0)) < int(min_sensor_non_nan): continue
        sensors_used.append(str(s))
        if len(sensors_used) >= max_sensors: break
    for t in targets:
        if t not in sensors_used: sensors_used = [t] + [x for x in sensors_used if x != t]
    sensors_used = sensors_used[:max(1, max_sensors)]
    if not sensors_used: return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_sensors"}

    feature_names = [f"{s}__lag_{k}" for s in sensors_used for k in range(1, n_lags + 1)]
    X_parts, Y_parts, B0_parts, SEG_parts = [], [], [], []
    seg_stats = {"segments_total": len(segments), "segments_used": 0}

    for seg_id in segments:
        sdf = df2[df2["seg_id"] == seg_id]
        if sdf.empty: continue
        piv = sdf.pivot_table(index="ts", columns="sensor", values="val", aggfunc="last").sort_index()
        if piv.empty: continue
        
        rr = piv.resample(freq).last()
        for s in sensors_used: 
            if s not in rr.columns: rr[s] = np.nan
        rr = rr[sensors_used]
        if any(t not in rr.columns for t in targets): continue

        tgt_obs = sdf[sdf["sensor"].isin(targets)]["ts"].dropna()
        obs_ts = pd.DatetimeIndex(tgt_obs)
        if len(obs_ts) < 2: continue

        ffill_limit = _auto_ffill_limit(obs_ts, resample_sec, n_lags)
        max_gap = 6 * 3600
        idx = rr.index.to_series()
        for s in rr.columns:
            last_valid = idx.where(rr[s].notna()).ffill()
            age = (idx - last_valid).dt.total_seconds()
            rr.loc[age > max_gap, s] = np.nan
            rr[s] = rr[s].ffill(limit=ffill_limit)
        
        shifts = {}
        for s in sensors_used:
            for k in range(1, n_lags + 1): shifts[f"{s}__lag_{k}"] = rr[s].shift(k)
        X_df = pd.DataFrame(shifts, index=rr.index)[feature_names]
        Y_df = rr[targets].shift(-horizon_steps)
        B0_df = rr[targets]
        
        mask = Y_df.notna().all(axis=1) & X_df.notna().all(axis=1)
        if mask.sum() < min_sensor_non_nan: continue

        X_parts.append(X_df.loc[mask].to_numpy(dtype="float32"))
        Y_parts.append(Y_df.loc[mask].to_numpy(dtype="float32"))
        B0_parts.append(B0_df.loc[mask].to_numpy(dtype="float32"))
        SEG_parts.append(np.full(mask.sum(), str(seg_id), dtype=object))
        seg_stats["segments_used"] += 1

    if not X_parts: return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, feature_names, sensors_used, {**seg_stats, "reason": "no_trainable"}
    
    return np.vstack(X_parts), np.vstack(Y_parts), np.vstack(B0_parts), np.concatenate(SEG_parts), feature_names, sensors_used, {
        "n_supervised": sum(len(x) for x in X_parts), "n_targets": len(targets)
    }

def get_model_pipeline(model_type: str, tune: bool = False) -> Tuple[Any, Dict]:
    model_type = model_type.upper()
    steps = [("imputer", SimpleImputer(strategy="median"))]
    param_grid = {}
    
    if model_type == "RF":
        est = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        steps.append(("est", est))
        if tune: param_grid = {"est__n_estimators": [100, 300], "est__min_samples_leaf": [1, 5]} 
    elif model_type == "HGB":
        hgb = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        steps.append(("est", MultiOutputRegressor(hgb)))
        if tune: param_grid = {"est__estimator__max_iter": [100, 300], "est__estimator__learning_rate": [0.01, 0.1]}
    elif model_type == "XGB":
        if not HAS_XGB: raise ValueError("XGBoost not installed.")
        steps.append(("est", XGBRegressor(n_jobs=-1, random_state=42, n_estimators=100)))
        if tune: param_grid = {"est__n_estimators": [100, 300], "est__learning_rate": [0.05, 0.1]}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline(steps), param_grid

def train_and_eval(X, Y, B0, seg, *, eval_mode, holdout_k, min_test, model_type, tune):
    if eval_mode == "batch_holdout":
        unique_segs = pd.unique(seg)
        if len(unique_segs) <= holdout_k:
            print(f"Warning: Not enough segments ({len(unique_segs)}) for holdout. Fallback time_split.")
            eval_mode = "time_split"
        else:
            test_segs = set(unique_segs[-holdout_k:])
            is_test = np.isin(seg, list(test_segs))
            Xtr, Xte = X[~is_test], X[is_test]
            Ytr, Yte = Y[~is_test], Y[is_test]
            B0te = B0[is_test]
            if len(Xte) < 10: eval_mode = "time_split"

    if eval_mode == "time_split":
        n_test = max(min_test, int(0.2 * len(X)))
        Xtr, Xte = X[:-n_test], X[-n_test:]
        Ytr, Yte = Y[:-n_test], Y[-n_test:]
        B0te = B0[-n_test:]

    pipe, grid = get_model_pipeline(model_type, tune=tune)
    best_params = None
    if tune and grid:
        search = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, scoring="neg_mean_absolute_error")
        search.fit(Xtr, Ytr)
        pipe, best_params = search.best_estimator_, search.best_params_
    else:
        pipe.fit(Xtr, Ytr)
        
    pred = pipe.predict(Xte)
    # MAE per target
    mae_base, mae_model = [], []
    for j in range(Yte.shape[1]):
        m = np.isfinite(Yte[:, j]) & np.isfinite(pred[:, j])
        if not m.any(): mae_base.append(np.nan); mae_model.append(np.nan)
        else:
            mae_base.append(mean_absolute_error(Yte[m, j], B0te[m, j]))
            mae_model.append(mean_absolute_error(Yte[m, j], pred[m, j]))
    
    lift = [(b - m) for b, m in zip(mae_base, mae_model)]
    
    return {
        "ok": True,
        "metrics": {
            "lift_mean": float(np.nanmean(lift)) if lift else 0.0,
            "baseline_mae": mae_base, "model_mae": mae_model,
            "n_test": len(Xte)
        },
        "pipeline": pipe, "best_params": best_params
    }

# --- AutoML Logic ---

def run_automl_loop(df_raw, args, targets):
    results = []
    resample_opts = [int(x) for x in args.search_resample.split(",")]
    lag_opts = [int(x) for x in args.search_lags.split(",")]
    model_opts = args.model_type.split(",") # Can accept "RF,HGB" if needed
    
    best_score, best_cfg, best_pipe = -9999, None, None

    print(f"AutoML Search: Resample={resample_opts}, Lags={lag_opts}, Models={model_opts}")

    for r_sec in resample_opts:
        for n_lag in lag_opts:
            print(f" > Prep: R={r_sec}s, L={n_lag} ... ", end="")
            X, Y, B0, seg, _, _, info = build_multitarget_dataset(
                df_raw, targets=targets, resample_sec=r_sec, horizon_sec=args.horizon_sec,
                n_lags=n_lag, segment_field=args.segment_field, session_gap_sec=21600,
                max_sensors=args.max_inputs, min_sensor_non_nan=10
            )
            
            if len(Y) < 100:
                print("Skipped (Data<100)")
                continue
            
            print(f"OK ({len(Y)} rows)")

            for m_type in model_opts:
                if m_type == "XGB" and not HAS_XGB: continue
                
                # Fast eval for AutoML (No deep tuning inside the loop for speed)
                res = train_and_eval(
                    X, Y, B0, seg, 
                    eval_mode="batch_holdout", holdout_k=3, min_test=20, min_points=20,
                    model_type=m_type, tune=False 
                )
                
                score = res["metrics"]["lift_mean"]
                print(f"   -> {m_type}: Lift={score:.4f}")
                
                cfg = {
                    "id": f"{m_type}_R{r_sec}_L{n_lag}",
                    "model": m_type, "resample": r_sec, "lags": n_lag,
                    "lift": score, "metrics": res["metrics"]
                }
                results.append(cfg)
                
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
                    best_pipe = res["pipeline"]
                    # Retrain on full data for artifact?
                    # For strictness, we keep the one trained on Train split, 
                    # or strictly refit on ALL X. Let's refit on ALL X.
                    best_pipe.fit(X, Y)

    return results, best_cfg, best_pipe

# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plId", type=int, default=0)
    ap.add_argument("--wcId", type=int, default=0)
    ap.add_argument("--wsId", type=int, default=0)
    ap.add_argument("--wsUid", default="")
    ap.add_argument("--stNo", required=True)
    ap.add_argument("--opTc", default="ALL")
    
    # Data Fetch
    ap.add_argument("--time_min", default="")
    ap.add_argument("--time_max", default="")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=50000)
    
    # Config
    ap.add_argument("--targets", default="")
    ap.add_argument("--horizon_sec", type=int, default=60)
    ap.add_argument("--max_inputs", type=int, default=40)
    
    # Modes
    ap.add_argument("--automl", action="store_true", help="Enable AutoML Loop")
    ap.add_argument("--search_resample", default="30,60")
    ap.add_argument("--search_lags", default="6,12")
    
    # Manual Mode Params
    ap.add_argument("--model_type", default="RF", help="RF, HGB, XGB (comma sep for AutoML)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--resample_sec", type=int, default=60)
    ap.add_argument("--n_lags", type=int, default=6)
    ap.add_argument("--eval_mode", default="time_split")

    ap.add_argument("--out_dir", default="./models/offline_mimo_rf")
    ap.add_argument("--run_id", default="")
    
    # Local imports mock
    try:
        from modules.context_profiler import profile_context_from_rows
        from modules.context_policy import select_context_policy
        from modules.model_registry import safe_token
    except ImportError:
        def profile_context_from_rows(r): return {}
        def select_context_policy(c, prefer_stock=True): return {"segment_field": "SESSION"}
        def safe_token(s, default=""): return re.sub(r'[^a-zA-Z0-9]', '', str(s)) or default

    args = ap.parse_args()
    
    # 1. Fetch
    pl, wc, ws = args.plId, args.wcId, args.wsId
    if args.wsUid:
        p = _parse_wsuid(str(args.wsUid))
        if p: pl, wc, ws = p
    
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tmin = ensure_utc(args.time_min).to_pydatetime() if args.time_min else None
    tmax = ensure_utc(args.time_max).to_pydatetime() if args.time_max else None
    
    print(f"[{run_id}] Fetching Data...")
    rows = fetch_rows_by_ws_time_range(pl, wc, ws, time_min=tmin, time_max=tmax, days=args.days, limit=args.limit)
    rows = filter_rows_by_stock_op(rows, st_no=args.stNo, op_tc=args.opTc)
    if not rows: return 2
    
    ctx = profile_context_from_rows(rows)
    policy = select_context_policy(ctx)
    seg_field = policy.get("segment_field", "SESSION")
    args.segment_field = seg_field # Injection for AutoML loop
    
    print(f"[{run_id}] Processing DataFrame (Seg={seg_field})...")
    df = rows_to_df(rows, segment_field=seg_field)
    
    targets = [t.strip() for t in args.targets.split(",")] if args.targets else []
    if not targets: targets = df["sensor"].value_counts().head(10).index.tolist()
    
    # 2. Execution Mode
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.automl:
        # --- AUTOML MODE ---
        print(f"[{run_id}] Starting AutoML...")
        leaderboard, winner_cfg, winner_pipe = run_automl_loop(df, args, targets)
        
        if not winner_cfg:
            print("AutoML Failed.")
            return 3
            
        print(f"WINNER: {winner_cfg['id']} (Lift={winner_cfg['lift']:.4f})")
        
        # Save
        prefix = f"AUTOML_{safe_token(args.stNo)}_{run_id}"
        model_path = os.path.join(args.out_dir, f"{prefix}_BEST.pkl")
        joblib.dump(winner_pipe, model_path)
        
        meta = {
            "run_id": run_id,
            "mode": "AUTOML",
            "winner": winner_cfg,
            "leaderboard": sorted(leaderboard, key=lambda x: x['lift'], reverse=True),
            "targets": targets,
            "model_path": model_path
        }
        with open(os.path.join(args.out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    else:
        # --- MANUAL MODE ---
        X, Y, B0, seg, feats, sensors, info = build_multitarget_dataset(
            df, targets=targets, resample_sec=args.resample_sec,
            horizon_sec=args.horizon_sec, n_lags=args.n_lags,
            segment_field=seg_field, session_gap_sec=21600,
            max_sensors=args.max_inputs, min_sensor_non_nan=10
        )
        if len(Y) < 50: return 3
        
        print(f"[{run_id}] Training Single {args.model_type}...")
        res = train_and_eval(
            X, Y, B0, seg, eval_mode=args.eval_mode, holdout_k=3, min_test=50,
            model_type=args.model_type, tune=args.tune
        )
        
        prefix = f"MANUAL_{safe_token(args.stNo)}_{run_id}"
        model_path = os.path.join(args.out_dir, f"{prefix}.pkl")
        joblib.dump(res["pipeline"], model_path)
        
        meta = {
            "run_id": run_id, "mode": "MANUAL", "config": vars(args),
            "metrics": res["metrics"], "data_info": info,
            "model_path": model_path
        }
        with open(os.path.join(args.out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
            
    print(f"Done. Saved to {args.out_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
