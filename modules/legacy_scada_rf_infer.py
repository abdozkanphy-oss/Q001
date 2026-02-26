# modules/legacy_scada_rf_infer.py
"""
Legacy SCADA-style MIMO RandomForest training + evaluation + artifact save.

Purpose
- Provide a v2.20-compatible, single-module CLI that reproduces the classic
  scada_product_training.py behavior (windowed next-step forecasting on a per-drug basis),
  while sourcing data from dw_tbl_raw_data_by_ws (preferred) or an exported CSV.

Key behaviors (matches scada_product_training.py)
- Segment by prod_order_reference_no (batch); do NOT mix batches when building windows.
- Per batch: pivot long->wide (ts index, equipment_name columns, counter_reading values)
- Resample to fixed grid (default 30s for Antares legacy), aggregate by last/mean
- Inactive strategy: FFILL (then fill remaining NaN with 0.0) or ZERO
- Optional time-elapsed feature within each batch
- Build windows: X[t-lookback:t], y[t] (next-step at the resample grid)
- Train: RobustScaler on flattened X, then MultiOutputRegressor(RandomForestRegressor)

Evaluation (added)
- Uses the last N batches (default 2) as validators, selected by batch start time.
- Reports overall R2 + masked MAPE, and per-target R2/MAPE.
- Also reports a lag0 persistence baseline from the last timestep in each window
  (useful sanity check for forecasting difficulty).

Notes
- MAPE is unstable near zero. Masked MAPE ignores |y| < mape_min_abs.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler

# Prefer DW projection table (partitioned by workstation + time) for offline work
try:
    from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws  # type: ignore
except Exception:
    dw_tbl_raw_data_by_ws = None  # allows CSV-only usage


# -----------------------------
# Helpers
# -----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(s: str) -> datetime:
    # Accept "YYYY-mm-dd HH:MM:SS" or ISO; assume local naive is UTC for offline use
    s = str(s).strip()
    if not s:
        raise ValueError("Empty datetime string")
    try:
        # pandas is flexible
        dt = pd.to_datetime(s, utc=True)
        if getattr(dt, "to_pydatetime", None):
            dt = dt.to_pydatetime()
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        raise ValueError(f"Failed to parse datetime: {s!r}: {e}") from e


def _safe_name(s: str) -> str:
    s = str(s)
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "_", "-", "."):
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "NA"


def _norm_str(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def _infer_ts_col(df: pd.DataFrame) -> str:
    # Prefer measurement_date, else create_date, else ts
    for c in ("measurement_date", "create_date", "ts"):
        if c in df.columns:
            return c
    raise ValueError("CSV missing a timestamp column: expected measurement_date or create_date (or ts)")


def _ensure_dt_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt


@dataclass
class BatchFrame:
    batch_id: str
    wide: pd.DataFrame  # indexed by ts
    cols: List[str]     # stable column order (including meta_time_elapsed_sec if used)


def build_wide_batch_frame(
    df_b: pd.DataFrame,
    *,
    resample_seconds: int,
    resample_method: str,
    inactive_strategy: str,
) -> pd.DataFrame:
    """
    df_b columns required: ts (datetime64[ns, UTC]), equipment_name, counter_reading
    """
    if df_b.empty:
        return pd.DataFrame()

    df_b = df_b.copy()
    df_b = df_b.dropna(subset=["ts", "equipment_name"])
    if df_b.empty:
        return pd.DataFrame()

    df_b["equipment_name"] = df_b["equipment_name"].astype(str)
    df_b["counter_reading"] = pd.to_numeric(df_b["counter_reading"], errors="coerce")

    # pivot (duplicate timestamps/sensors can exist -> take last by ts ordering)
    df_b = df_b.sort_values("ts")
    wide = df_b.pivot_table(
        index="ts",
        columns="equipment_name",
        values="counter_reading",
        aggfunc="last",
    )
    if wide.empty:
        return pd.DataFrame()

    wide = wide.sort_index()

    rule = f"{int(resample_seconds)}S"
    if str(resample_method).lower() == "mean":
        wide = wide.resample(rule).mean()
    else:
        # last
        wide = wide.resample(rule).last()

    if str(inactive_strategy).upper() == "FFILL":
        wide = wide.ffill()
        wide = wide.fillna(0.0)
    else:
        wide = wide.fillna(0.0)

    return wide


def make_windows(
    wide: pd.DataFrame,
    *,
    lookback: int,
    use_time_elapsed: bool,
    cols_ref: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X: [N, lookback, D]
      y: [N, D] (predict current-step)
      cols: list[str] (D columns, including meta_time_elapsed_sec if used)
    """
    if wide.empty:
        return np.zeros((0, lookback, 0)), np.zeros((0, 0)), []

    wide = wide.copy()
    wide = wide.sort_index()

    # Build stable columns list:
    sensor_cols = [c for c in wide.columns if c not in ("meta_time_elapsed_sec", "meta_time_elapsed")]
    sensor_cols = [str(c) for c in sensor_cols]

    if cols_ref is not None:
        # align to reference columns (excluding time feature which we will add deterministically)
        # Add any missing sensor columns as zeros.
        ref_sensors = [c for c in cols_ref if c not in ("meta_time_elapsed_sec", "meta_time_elapsed")]
        for c in ref_sensors:
            if c not in wide.columns:
                wide[c] = 0.0
        sensor_cols = ref_sensors

    # Add time-elapsed feature if enabled
    if use_time_elapsed:
        t0 = wide.index.min()
        wide["meta_time_elapsed_sec"] = (wide.index - t0).total_seconds().astype(float)

    cols = sensor_cols + (["meta_time_elapsed_sec"] if use_time_elapsed else [])
    wide = wide[cols]

    arr = wide.to_numpy(dtype=float)
    if arr.shape[0] <= int(lookback):
        return np.zeros((0, lookback, arr.shape[1])), np.zeros((0, arr.shape[1])), cols

    X_list = []
    y_list = []
    for i in range(int(lookback), arr.shape[0]):
        X_list.append(arr[i - int(lookback) : i, :])
        y_list.append(arr[i, :])

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    return X, y, cols


def _masked_mape(y_true: np.ndarray, y_pred: np.ndarray, *, min_abs: float = 1e-3) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    mask = denom >= float(min_abs)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / denom[mask])) * 100.0)


def _per_target_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mape_min_abs: float,
) -> Tuple[List[float], List[float]]:
    r2s = []
    mapes = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        try:
            r2j = float(r2_score(yt, yp))
        except Exception:
            r2j = float("nan")
        r2s.append(r2j)
        mapes.append(_masked_mape(yt, yp, min_abs=mape_min_abs))
    return r2s, mapes


def _lag0_baseline_from_X(X: np.ndarray) -> np.ndarray:
    # Predict y[t] = X[t-1] (last row in the lookback window)
    return X[:, -1, :]


def _select_batches_last_k(
    batch_ids: List[str],
    batch_start_ts: Dict[str, datetime],
    k: int,
) -> Tuple[List[str], List[str]]:
    ids = list(batch_ids)
    ids.sort(key=lambda b: batch_start_ts.get(b, datetime(1970, 1, 1, tzinfo=timezone.utc)))
    if k <= 0:
        return ids, []
    if len(ids) <= k:
        return [], ids
    return ids[:-k], ids[-k:]


def _rows_to_long_df(rows: List[object]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = {
            "measurement_date": getattr(r, "measurement_date", None),
            "create_date": getattr(r, "create_date", None),
            "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
            "prod_order_reference_no_txt": getattr(r, "prod_order_reference_no_txt", None),
            "produced_stock_name": getattr(r, "produced_stock_name", None),
            "produced_stock_no": getattr(r, "produced_stock_no", None),
            "equipment_name": getattr(r, "equipment_name", None),
            "counter_reading": getattr(r, "counter_reading", None),
            "operationtaskcode": getattr(r, "operationtaskcode", None),
        }
        out.append(d)
    df = pd.DataFrame(out)
    return df


def fetch_dw_rows(
    pl_id: int,
    wc_id: int,
    ws_id: int,
    *,
    time_min: datetime,
    time_max: datetime,
    limit: int,
) -> pd.DataFrame:
    if dw_tbl_raw_data_by_ws is None:
        raise RuntimeError("Cassandra model dw_tbl_raw_data_by_ws is unavailable; use --csv instead.")
    q = dw_tbl_raw_data_by_ws.objects.filter(
        plant_id=int(pl_id),
        work_center_id=int(wc_id),
        work_station_id=int(ws_id),
        measurement_date__gte=time_min,
        measurement_date__lt=time_max,
    ).limit(int(limit))
    rows = list(q)
    df = _rows_to_long_df(rows)
    return df


def filter_stock(df: pd.DataFrame, st_no: str) -> pd.DataFrame:
    st = _norm_str(st_no)
    if not st or st == "all":
        return df
    df = df.copy()
    df["produced_stock_name"] = df["produced_stock_name"].astype(str)
    df["produced_stock_no"] = df["produced_stock_no"].astype(str)
    m = (df["produced_stock_name"].map(_norm_str) == st) | (df["produced_stock_no"].map(_norm_str) == st)
    return df[m].copy()


def resolve_batch_id(df: pd.DataFrame, segment_field: str) -> pd.Series:
    seg = str(segment_field or "").strip()
    if seg.upper() in ("AUTO", ""):
        # Prefer txt if present
        if "prod_order_reference_no_txt" in df.columns and df["prod_order_reference_no_txt"].notna().any():
            return df["prod_order_reference_no_txt"].astype(str)
        return df["prod_order_reference_no"].astype(str)
    if seg not in df.columns:
        raise ValueError(f"segment_field {seg!r} not found in dataframe columns: {list(df.columns)}")
    return df[seg].astype(str)


def train_and_eval(
    df_long: pd.DataFrame,
    *,
    st_no: str,
    segment_field: str,
    resample_seconds: int,
    resample_method: str,
    inactive_strategy: str,
    lookback: int,
    use_time_elapsed: bool,
    val_last_k_batches: int,
    min_windows_per_batch: int,
    mape_min_abs: float,
    n_estimators: int,
    random_state: int,
) -> Dict:
    if df_long.empty:
        raise ValueError("No rows to train on after filtering.")

    ts_col = "measurement_date" if "measurement_date" in df_long.columns else _infer_ts_col(df_long)
    df_long = df_long.copy()
    df_long["ts"] = _ensure_dt_utc(df_long[ts_col])
    df_long = df_long.dropna(subset=["ts", "equipment_name", "counter_reading"])
    if df_long.empty:
        raise ValueError("No usable rows after dropping missing ts/equipment_name/counter_reading.")

    df_long["batch_id"] = resolve_batch_id(df_long, segment_field=segment_field)
    df_long["batch_id"] = df_long["batch_id"].astype(str)

    # Determine batch start times for ordering
    batch_start_ts: Dict[str, datetime] = {}
    for b, g in df_long.groupby("batch_id"):
        t0 = pd.to_datetime(g["ts"], utc=True, errors="coerce").min()
        if pd.isna(t0):
            continue
        batch_start_ts[str(b)] = t0.to_pydatetime()

    batch_ids = [b for b in sorted(df_long["batch_id"].unique().tolist()) if b in batch_start_ts]
    if len(batch_ids) < max(3, val_last_k_batches + 1):
        raise ValueError(f"Not enough batches for train/val split. Found {len(batch_ids)} batches.")

    train_batches, val_batches = _select_batches_last_k(batch_ids, batch_start_ts, k=int(val_last_k_batches))
    if not train_batches or not val_batches:
        raise ValueError("Train/val batch split produced empty side.")

    # Build windows per batch with stable column space based on first eligible batch in train
    cols_ref: Optional[List[str]] = None

    def _build_for_batches(batches: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
        Xs, ys = [], []
        per_batch_counts: Dict[str, int] = {}
        nonlocal cols_ref

        for b in batches:
            gb = df_long[df_long["batch_id"] == b][["ts", "equipment_name", "counter_reading"]].copy()
            if gb.empty:
                continue

            wide = build_wide_batch_frame(
                gb,
                resample_seconds=int(resample_seconds),
                resample_method=str(resample_method),
                inactive_strategy=str(inactive_strategy),
            )
            if wide.empty:
                continue

            X, y, cols = make_windows(
                wide,
                lookback=int(lookback),
                use_time_elapsed=bool(use_time_elapsed),
                cols_ref=cols_ref,
            )
            if X.shape[0] < int(min_windows_per_batch):
                continue

            if cols_ref is None:
                cols_ref = cols

            Xs.append(X)
            ys.append(y)
            per_batch_counts[b] = int(X.shape[0])

        if not Xs:
            return np.zeros((0, int(lookback), 0)), np.zeros((0, 0)), [], per_batch_counts
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        return X_all, y_all, (cols_ref or []), per_batch_counts

    X_train, y_train, cols, train_counts = _build_for_batches(train_batches)
    X_val, y_val, _, val_counts = _build_for_batches(val_batches)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError(
            f"No windows produced. train_windows={X_train.shape[0]} val_windows={X_val.shape[0]} "
            f"(min_windows_per_batch={min_windows_per_batch})."
        )

    # Fit scaler on training flat features
    scaler = RobustScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)

    # Train RF (multi-output)
    base = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_val_scaled)

    # Baseline (lag0)
    y_base = _lag0_baseline_from_X(X_val)

    # Metrics (overall)
    r2_overall = float(r2_score(y_val, y_pred, multioutput="variance_weighted"))
    r2_base = float(r2_score(y_val, y_base, multioutput="variance_weighted"))
    mape_overall = _masked_mape(y_val, y_pred, min_abs=mape_min_abs)
    mape_base = _masked_mape(y_val, y_base, min_abs=mape_min_abs)

    # Per target
    r2_per, mape_per = _per_target_stats(y_val, y_pred, mape_min_abs=mape_min_abs)
    r2_per_base, mape_per_base = _per_target_stats(y_val, y_base, mape_min_abs=mape_min_abs)

    # Lift in MAE (positive means model better than baseline)
    mae_model = float(np.mean(np.abs(y_pred - y_val)))
    mae_base = float(np.mean(np.abs(y_base - y_val)))
    lift_mae = float((mae_base - mae_model) / max(1e-12, mae_base))

    return {
        "model": model,
        "scaler": scaler,
        "cols": cols,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "train_windows": int(X_train.shape[0]),
        "val_windows": int(X_val.shape[0]),
        "per_batch_train_windows": train_counts,
        "per_batch_val_windows": val_counts,
        "metrics": {
            "r2": r2_overall,
            "mape_masked": mape_overall,
            "baseline_r2": r2_base,
            "baseline_mape_masked": mape_base,
            "mae": mae_model,
            "baseline_mae": mae_base,
            "lift_mae": lift_mae,
            "r2_per_target": r2_per,
            "baseline_r2_per_target": r2_per_base,
            "mape_per_target": mape_per,
            "baseline_mape_per_target": mape_per_base,
            "mape_min_abs": float(mape_min_abs),
        },
        "params": {
            "stNo": st_no,
            "segment_field": segment_field,
            "resample_seconds": int(resample_seconds),
            "resample_method": str(resample_method),
            "inactive_strategy": str(inactive_strategy),
            "lookback": int(lookback),
            "use_time_elapsed": bool(use_time_elapsed),
            "val_last_k_batches": int(val_last_k_batches),
            "min_windows_per_batch": int(min_windows_per_batch),
            "n_estimators": int(n_estimators),
            "random_state": int(random_state),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Legacy SCADA-style MIMO RF trainer + evaluator (batch holdout by last K batches).")
    ap.add_argument("--plId", type=int, required=True)
    ap.add_argument("--wcId", type=int, required=True)
    ap.add_argument("--wsId", type=int, required=True)

    ap.add_argument("--stNo", required=True, help='Stock name or stock no; e.g. "Antares PV"')

    ap.add_argument("--time_min", default="", help='UTC-ish datetime string, e.g. "2024-01-01 08:00:00"')
    ap.add_argument("--time_max", default="", help='UTC-ish datetime string, e.g. "2024-12-30 23:59:59"')
    ap.add_argument("--days", type=int, default=9999, help="If time_min/time_max not given, look back N days from latest seen ts in DW/CSV.")

    ap.add_argument("--limit", type=int, default=400000, help="Max rows fetched from DW within time window.")
    ap.add_argument("--csv", default="", help="Optional CSV instead of Cassandra DW. Must include produced_stock_name/no etc.")
    ap.add_argument("--outdir", default="./models/legacy_scada_rf", help="Output directory for model artifacts.")

    ap.add_argument("--segment_field", default="prod_order_reference_no", help="Batch/segment field; typically prod_order_reference_no")
    ap.add_argument("--val_last_k_batches", type=int, default=2, help="Use the last K batches as validation.")
    ap.add_argument("--min_windows_per_batch", type=int, default=1, help="Drop batches producing fewer windows than this.")

    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--resample_seconds", type=int, default=30)
    ap.add_argument("--resample_method", default="last", choices=["last", "mean"])
    ap.add_argument("--inactive_strategy", default="FFILL", choices=["FFILL", "ZERO"])
    ap.add_argument("--use_time_elapsed", action="store_true")
    ap.add_argument("--no_time_elapsed", action="store_true")

    ap.add_argument("--mape_min_abs", type=float, default=1e-3, help="Mask |y| < this when computing MAPE.")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()

    use_time = True
    if args.no_time_elapsed:
        use_time = False
    if args.use_time_elapsed:
        use_time = True

    # Load data
    if args.csv:
        df = pd.read_csv(args.csv)
        ts_col = _infer_ts_col(df)
        df["ts"] = _ensure_dt_utc(df[ts_col])
    else:
        # Determine time window
        # If time_min/max not provided: use a generous window anchored by now (UTC) minus days.
        if args.time_min and args.time_max:
            tmin = _parse_dt(args.time_min)
            tmax = _parse_dt(args.time_max)
        else:
            # Fallback window: [now-days, now]
            tmax = datetime.now(timezone.utc)
            tmin = tmax - timedelta(days=int(args.days))
        df = fetch_dw_rows(int(args.plId), int(args.wcId), int(args.wsId), time_min=tmin, time_max=tmax, limit=int(args.limit))
        df["ts"] = _ensure_dt_utc(df["measurement_date"])

    df = filter_stock(df, args.stNo)
    if df.empty:
        raise SystemExit(f"No rows found after filtering stNo={args.stNo!r}.")

    # If time_min/max provided and CSV used, filter the time window
    if args.csv and args.time_min and args.time_max:
        tmin = _parse_dt(args.time_min)
        tmax = _parse_dt(args.time_max)
        df = df[(df["ts"] >= tmin) & (df["ts"] < tmax)].copy()

    # Train & eval
    result = train_and_eval(
        df,
        st_no=args.stNo,
        segment_field=args.segment_field,
        resample_seconds=int(args.resample_seconds),
        resample_method=str(args.resample_method),
        inactive_strategy=str(args.inactive_strategy),
        lookback=int(args.lookback),
        use_time_elapsed=bool(use_time),
        val_last_k_batches=int(args.val_last_k_batches),
        min_windows_per_batch=int(args.min_windows_per_batch),
        mape_min_abs=float(args.mape_min_abs),
        n_estimators=int(args.n_estimators),
        random_state=int(args.random_state),
    )

    os.makedirs(args.outdir, exist_ok=True)
    safe_st = _safe_name(args.stNo)
    ws_token = f"WS_{int(args.plId)}_{int(args.wcId)}_{int(args.wsId)}"
    base_name = f"LEGACY_DRUG_{safe_st}__{ws_token}__RSEC_{int(args.resample_seconds)}__LB_{int(args.lookback)}__RF"

    model_path = os.path.join(args.outdir, f"{base_name}.pkl")
    scaler_path = os.path.join(args.outdir, f"{base_name}_scaler.pkl")
    meta_path = os.path.join(args.outdir, f"{base_name}_meta.json")

    joblib.dump(result["model"], model_path)
    joblib.dump(result["scaler"], scaler_path)

    meta = {
        "created_utc": _utc_now_iso(),
        "artifact_type": "legacy_scada_mimo_rf",
        "plId": int(args.plId),
        "wcId": int(args.wcId),
        "wsId": int(args.wsId),
        "stNo": str(args.stNo),
        "cols": result["cols"],
        "params": result["params"],
        "train_batches": result["train_batches"],
        "val_batches": result["val_batches"],
        "train_windows": result["train_windows"],
        "val_windows": result["val_windows"],
        "per_batch_train_windows": result["per_batch_train_windows"],
        "per_batch_val_windows": result["per_batch_val_windows"],
        "metrics": result["metrics"],
        "notes": {
            "windowing": "X[t-lookback:t], y[t] on resample grid",
            "flatten_order": "row-major (time-major) from wide[cols]",
            "baseline": "lag0 persistence y_hat = X_last_step",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Print evaluation summary (end of training)
    m = result["metrics"]
    print("[legacy_scada_rf] OK")
    print(f"Model : {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Meta  : {meta_path}")
    print(f"Train windows: {result['train_windows']}  Val windows: {result['val_windows']}")
    print(f"Train batches: {len(result['train_batches'])}  Val batches: {len(result['val_batches'])}")
    print(f"[legacy_scada_rf] R2={m['r2']:.6f}  MAPE(masked)={m['mape_masked']:.3f}%")
    print(f"[legacy_scada_rf] Baseline R2={m['baseline_r2']:.6f}  Baseline MAPE(masked)={m['baseline_mape_masked']:.3f}%")
    print(f"[legacy_scada_rf] MAE={m['mae']:.6f}  Baseline MAE={m['baseline_mae']:.6f}  Lift(MAE)={m['lift_mae']:.6f}")

    # Per-target
    cols = result["cols"]
    print("[legacy_scada_rf] Per-target R2:")
    for name, v in zip(cols, m["r2_per_target"]):
        print(f"  {name}: {v:.6f}")
    print("[legacy_scada_rf] Per-target MAPE(masked):")
    for name, v in zip(cols, m["mape_per_target"]):
        vv = "nan" if (v != v) else f"{v:.3f}%"
        print(f"  {name}: {vv}")

    # Baseline per-target (optional, but helpful)
    print("[legacy_scada_rf] Per-target baseline R2:")
    for name, v in zip(cols, m["baseline_r2_per_target"]):
        print(f"  {name}: {v:.6f}")

if __name__ == "__main__":
    main()
