# modules/offline_phase2_unsup_trainer.py
# M3.x: Offline Phase2 UNSUP (multivariate) training on dw_tbl_raw_data_by_ws
#
# Produces artifacts under models/phase2models/offline_unsup:
#   - <prefix>__model.keras
#   - <prefix>__scaler.pkl
#   - <prefix>__meta.json

from __future__ import annotations

import argparse
import re
import json
import os
import time
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from utils.logger_2 import setup_logger

log = setup_logger("phase2_offline_trainer", "logs/phase2_offline_trainer.log")


@dataclass
class Phase2Meta:
    trained_at_utc: str
    wsuid: str
    stock_key: str
    op_tc: str
    resample_sec: int
    timesteps: int
    sensor_cols: List[str]
    n_rows_raw: int
    n_rows_grid: int
    n_train_seq: int
    threshold: float
    model_path: str
    scaler_path: str


def _wsuid(pl: int, wc: int, ws: int) -> str:
    # Canonical workstation_uid (must match Stage0 / utils.identity):
    #   "<plId>_WC<wcId>_WS<wsId>"
    from utils.identity import get_workstation_uid
    return str(get_workstation_uid({"plId": pl, "wcId": wc, "wsId": ws}) or f"{pl}_WC{wc}_WS{ws}")


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:160] or "X"



def _parse_dt_naive(s: str) -> datetime:
    """Parse a datetime string into a naive `datetime` (local wall-clock).

    - Accepts ISO strings with timezone offsets (e.g. '2024-12-11 10:50:54+03:00').
    - If an offset is present, it is **dropped without conversion** (keeps wall-clock time).
      This matches existing trainer behavior which queries Cassandra using naive datetimes.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime string")
    ts = pd.to_datetime(s, errors="raise")
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)  # drop tz, keep wall-clock
    except Exception:
        pass
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime()
    if isinstance(ts, datetime):
        return ts
    return datetime.fromtimestamp(float(ts))


def _load_intervals_file(path: str) -> List[tuple[datetime, datetime]]:
    """Load training intervals from a file.

    Supported formats:
    1) JSON list: [{"time_min": "...", "time_max": "..."}, ...]
    2) JSONL: one JSON object per line with keys time_min/time_max (or start/end).
    3) Delimited text: each non-empty line "time_min,time_max" (comma/semicolon/tab).

    Returned datetimes are naive (see `_parse_dt_naive`).
    """
    p = os.path.abspath(str(path))
    with open(p, "r", encoding="utf-8") as f:
        raw = f.read()

    def _obj_to_pair(obj) -> Optional[tuple[datetime, datetime]]:
        if not isinstance(obj, dict):
            return None
        a = obj.get("time_min") or obj.get("start") or obj.get("from")
        b = obj.get("time_max") or obj.get("end") or obj.get("to")
        if not a or not b:
            return None
        return (_parse_dt_naive(str(a)), _parse_dt_naive(str(b)))

    intervals: List[tuple[datetime, datetime]] = []
    s = raw.lstrip()
    if s.startswith("[") or s.startswith("{"):
        # Try JSON first
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "intervals" in obj:
                obj = obj.get("intervals")
            if isinstance(obj, list):
                for it in obj:
                    pair = _obj_to_pair(it)
                    if pair:
                        intervals.append(pair)
            elif isinstance(obj, dict):
                pair = _obj_to_pair(obj)
                if pair:
                    intervals.append(pair)
        except Exception:
            # Fallback: JSONL
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pair = _obj_to_pair(obj)
                if pair:
                    intervals.append(pair)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,	;]+", line)
            if len(parts) < 2:
                continue
            a = parts[0].strip()
            b = parts[1].strip()
            if not a or not b:
                continue
            intervals.append((_parse_dt_naive(a), _parse_dt_naive(b)))

    out: List[tuple[datetime, datetime]] = []
    for (a, b) in intervals:
        if a >= b:
            continue
        out.append((a, b))
    out.sort(key=lambda x: x[0])
    return out

def _select_sensors_by_coverage(wide_grid: pd.DataFrame, min_cov: float, max_sensors: int) -> List[str]:
    stats = []
    for c in wide_grid.columns:
        cov = float(wide_grid[c].notna().mean())
        cnt = int(wide_grid[c].notna().sum())
        if cov >= float(min_cov):
            stats.append((cov, cnt, str(c)))
    stats.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _, _, c in stats[: int(max_sensors)]]


def _to_sequences(X: np.ndarray, timesteps: int) -> np.ndarray:
    N, F = X.shape
    T = int(timesteps)
    if N < T:
        return np.empty((0, T, F), dtype=float)
    out = np.zeros((N - T + 1, T, F), dtype=float)
    for i in range(N - T + 1):
        out[i] = X[i : i + T]
    return out


def _build_lstm_autoencoder(timesteps: int, n_features: int):
    from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout # type: ignore
    from tensorflow.keras.models import Model # type: ignore
    from tensorflow.keras.regularizers import l2 # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping # type: ignore

    inp = Input(shape=(timesteps, n_features))
    x = LSTM(64, activation="relu", kernel_regularizer=l2(1e-3), return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = LSTM(32, activation="relu", kernel_regularizer=l2(1e-3), return_sequences=False)(x)
    x = RepeatVector(timesteps)(x)
    x = LSTM(32, activation="relu", kernel_regularizer=l2(1e-3), return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, activation="relu", kernel_regularizer=l2(1e-3), return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    return model, es




try:
    # TF 2.19 + Keras 3: tf.keras callbacks are the correct base for Model.fit() in this project.
    from tensorflow.keras.callbacks import Callback as _KerasCallback  # type: ignore
except Exception:  # pragma: no cover
    _KerasCallback = object  # type: ignore


class _ProgressLoggerCallback(_KerasCallback):
    """Lightweight progress logger for Keras `fit()`.

    - No external deps (no tqdm).
    - Works with TF 2.19 / Keras 3 by subclassing `tf.keras.callbacks.Callback`.
    - Logs to project logger (file) while keeping `verbose=0` usable.
    """

    def __init__(
        self,
        log,
        epochs: int,
        steps_per_epoch: int,
        every_batches: int = 50,
        every_sec: float = 15.0,
    ):
        super().__init__()  # type: ignore[misc]
        self._log = log
        self._epochs = int(max(1, epochs))
        self._spe = int(max(1, steps_per_epoch))
        self._every_batches = int(max(1, every_batches))
        self._every_sec = float(max(0.5, every_sec))
        self._t0 = 0.0
        self._epoch_t0 = 0.0
        self._last_log_t = 0.0
        self._seen_batches = 0
        self._epoch = 0

    def on_train_begin(self, logs=None):
        now = time.time()
        self._t0 = now
        self._last_log_t = now
        # Prefer Keras-provided steps if available.
        try:
            steps = int(self.params.get("steps") or self._spe)  # type: ignore[attr-defined]
            if steps > 0:
                self._spe = steps
        except Exception:
            pass
        self._log.info(f"TRAIN_PROGRESS begin epochs={self._epochs} steps_per_epoch={self._spe}")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = int(epoch)
        self._epoch_t0 = time.time()
        self._last_log_t = self._epoch_t0
        self._seen_batches = 0
        self._log.info(f"TRAIN_PROGRESS epoch_start {epoch + 1}/{self._epochs}")

    def on_train_batch_end(self, batch, logs=None):
        self._seen_batches += 1
        now = time.time()
        due_batches = (self._seen_batches % self._every_batches) == 0
        due_time = (now - self._last_log_t) >= self._every_sec
        if not (due_batches or due_time):
            return
        self._last_log_t = now

        loss = None
        try:
            if logs and "loss" in logs:
                loss = float(logs["loss"])
        except Exception:
            loss = None

        elapsed = max(1e-6, now - self._epoch_t0)
        bps = self._seen_batches / elapsed
        rem = max(0, self._spe - self._seen_batches)
        eta = rem / max(1e-6, bps)
        loss_s = "" if loss is None else f" loss={loss:.6f}"
        self._log.info(
            f"TRAIN_PROGRESS epoch={self._epoch + 1}/{self._epochs} "
            f"step={self._seen_batches}/{self._spe}{loss_s} eta={eta:.1f}s"
        )

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        elapsed = now - self._epoch_t0
        loss = None
        val_loss = None
        try:
            if logs:
                if "loss" in logs:
                    loss = float(logs["loss"])
                if "val_loss" in logs:
                    val_loss = float(logs["val_loss"])
        except Exception:
            pass
        loss_s = "" if loss is None else f" loss={loss:.6f}"
        vloss_s = "" if val_loss is None else f" val_loss={val_loss:.6f}"
        self._log.info(f"TRAIN_PROGRESS epoch_end {epoch + 1}/{self._epochs} took={elapsed:.1f}s{loss_s}{vloss_s}")

    def on_train_end(self, logs=None):
        now = time.time()
        total = now - self._t0
        self._log.info(f"TRAIN_PROGRESS end total={total:.1f}s")


def _make_progress_logger_callback(
    log,
    epochs: int,
    steps_per_epoch: int,
    every_batches: int = 50,
    every_sec: float = 15.0,
):
    """Compatibility wrapper.

    Some local edits may call this helper instead of instantiating `_ProgressLoggerCallback` directly.
    """
    return _ProgressLoggerCallback(
        log=log,
        epochs=int(epochs),
        steps_per_epoch=int(steps_per_epoch),
        every_batches=int(every_batches),
        every_sec=float(every_sec),
    )

def train_phase2_unsup(
    pl_id: int,
    wc_id: int,
    ws_id: int,
    days: int,
    st_no: str,
    op_tc: str,
    resample_sec: int,
    timesteps: int,
    min_rows: int,
    min_cov: float,
    max_sensors: int,
    ffill_limit: int,
    threshold_q: float,
    out_dir: str,
    epochs: int,
    batch_size: int,
    keras_verbose: int,
    progress: bool,
    progress_every_batches: int,
    progress_every_sec: float,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    intervals_file: Optional[str] = None,
    jsonl: str = "",
) -> Optional[Phase2Meta]:
    """Train Phase2 LSTM-AE in an unsupervised manner.

    Time range options:
    - Default: last `--days` days (legacy behavior).
    - `--time_min/--time_max`: train on one explicit interval.
    - `--intervals_file`: train across multiple explicit intervals; each interval is resampled independently,
      and sequences are built within each interval only (no cross-interval stitching).
    """
    # Reuse Phase3-friendly DW fetch (dw_tbl_raw_data_by_ws)
    from modules.offline_outonly_trainer import fetch_output_rows_by_ws, rows_to_wide_df, safe_token

    # Resolve intervals
    intervals: List[tuple[datetime, datetime]] = []
    range_tag = ""
    if intervals_file:
        intervals = _load_intervals_file(str(intervals_file))
        range_tag = f"INTV_{len(intervals)}"
        if not intervals:
            log.warning(f"TRAIN_SKIPPED: intervals_file empty/invalid: {intervals_file}")
            return None
    elif time_min or time_max:
        if not (time_min and time_max):
            raise ValueError("--time_min and --time_max must be provided together")
        intervals = [(_parse_dt_naive(str(time_min)), _parse_dt_naive(str(time_max)))]
        range_tag = "RANGE_CUSTOM"
    else:
        # Default: last `--days` days.
        # For JSONL recordings, anchor to the latest event-time in the dump to avoid empty windows for historical data.
        if jsonl:
            from modules.offline_jsonl_source import jsonl_probe_latest_dt
            latest = jsonl_probe_latest_dt(jsonl, pl_id=int(pl_id), wc_id=int(wc_id), ws_id=int(ws_id), st_no=str(st_no), op_tc=str(op_tc))
            end_dt = latest if latest is not None else datetime.now(timezone.utc)
        else:
            end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=int(days))
        intervals = [(start_dt, end_dt)]
        range_tag = f"DAYS_{int(days)}"

    interval_grids_raw: List[pd.DataFrame] = []
    n_raw_total = 0

    for (start_dt, end_dt) in intervals:
        rows = fetch_output_rows_by_ws(
            pl_id=int(pl_id),
            wc_id=int(wc_id),
            ws_id=int(ws_id),
            start_dt=start_dt,
            end_dt=end_dt,
            limit=10**9,
            st_no=None if str(st_no).upper() == "ALL" else str(st_no),
            op_tc=None if str(op_tc).upper() == "ALL" else str(op_tc),
            chunk_hours=6,
            show_progress=True,
            allow_filtering=True,
            jsonl=str(jsonl or "")
        )

        n_raw_total += len(rows)
        log.info(
            f"FETCH_OK rows={len(rows)} start={start_dt} end={end_dt} "
            f"plId={pl_id} wcId={wc_id} wsId={ws_id} stNo={st_no} opTc={op_tc} range={range_tag}"
        )

        if not rows:
            continue

        wide = rows_to_wide_df(rows)
        if wide is None or wide.empty:
            continue

        grid_raw = wide.resample(f"{int(resample_sec)}s").last().sort_index()
        if grid_raw.empty:
            continue

        interval_grids_raw.append(grid_raw)

    n_raw = int(n_raw_total)
    if n_raw < int(min_rows):
        log.warning(f"TRAIN_SKIPPED: not enough rows (rows={n_raw} < min_rows={min_rows})")
        return None

    if not interval_grids_raw:
        log.warning("TRAIN_SKIPPED: no non-empty intervals after fetch/resample")
        return None

    # Concatenate interval grids ONLY for sensor coverage selection (no reindexing over gaps).
    grid_cov = pd.concat(interval_grids_raw, axis=0).sort_index()

    cols = _select_sensors_by_coverage(grid_cov, min_cov=float(min_cov), max_sensors=int(max_sensors))
    if not cols:
        log.warning("TRAIN_SKIPPED: no sensors meet coverage threshold")
        return None

    # Build per-interval filled grids (ffill does NOT cross interval boundaries).
    interval_grids: List[pd.DataFrame] = []
    for g in interval_grids_raw:
        g2 = g.reindex(columns=cols)
        if int(ffill_limit) > 0:
            g2 = g2.ffill(limit=int(ffill_limit))
        g2 = g2.fillna(0.0)
        interval_grids.append(g2)

    n_rows_grid = int(sum(int(g.shape[0]) for g in interval_grids))
    if n_rows_grid <= 0:
        log.warning("TRAIN_SKIPPED: resampled grids empty after selection")
        return None

    X_parts = [g.to_numpy(dtype=float) for g in interval_grids if not g.empty]
    if not X_parts:
        log.warning("TRAIN_SKIPPED: no grid points after selection")
        return None

    X_all = np.vstack(X_parts)
    if X_all.shape[0] < int(timesteps) + 1:
        log.warning(f"TRAIN_SKIPPED: insufficient grid rows for timesteps (rows={X_all.shape[0]} < {timesteps+1})")
        return None

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    # Build sequences within each interval only (no cross-interval stitching).
    X_seq_parts: List[np.ndarray] = []
    cursor = 0
    for g in interval_grids:
        n = int(g.shape[0])
        if n <= 0:
            continue
        Xi_scaled = X_all_scaled[cursor : cursor + n]
        cursor += n
        seq = _to_sequences(Xi_scaled, timesteps=int(timesteps))
        if seq.shape[0] > 0:
            X_seq_parts.append(seq)

    if not X_seq_parts:
        log.warning("TRAIN_SKIPPED: sequence build failed (no interval produced enough rows)")
        return None

    X_seq = np.concatenate(X_seq_parts, axis=0)

    n_seq = X_seq.shape[0]
    n_train = int(max(1, int(0.9 * n_seq)))
    X_train = X_seq[:n_train]
    X_val = X_seq[n_train:] if n_train < n_seq else X_seq[:1]

    model, es = _build_lstm_autoencoder(int(timesteps), int(X_seq.shape[2]))
    est_train_mb = float(getattr(X_train, "nbytes", 0) / (1024 * 1024))
    est_val_mb = float(getattr(X_val, "nbytes", 0) / (1024 * 1024))
    log.info(
        f"TRAIN_START seq={n_seq} train={X_train.shape} val={X_val.shape} sensors={len(cols)} "
        f"mem_est_mb train={est_train_mb:.1f} val={est_val_mb:.1f}"
    )

    callbacks = [es]
    if progress:
        steps_per_epoch = int(math.ceil(len(X_train) / max(1, int(batch_size))))
        callbacks.append(
            _make_progress_logger_callback(
                log,
                epochs=int(epochs),
                steps_per_epoch=steps_per_epoch,
                every_batches=int(progress_every_batches),
                every_sec=float(progress_every_sec),
            )
        )

    model.fit(
        X_train,
        X_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_data=(X_val, X_val),
        shuffle=True,
        callbacks=callbacks,
        verbose=int(keras_verbose),
    )

    pred = model.predict(X_train, verbose=0)
    errs = np.mean(np.square(pred - X_train), axis=(1, 2))
    thr = float(np.quantile(errs, float(threshold_q)))

    os.makedirs(out_dir, exist_ok=True)
    wsuid = _wsuid(pl_id, wc_id, ws_id)
    prefix = "__".join(
        [
            f"WSUID_{safe_token(wsuid)}",
            f"ST_{_safe_name(str(st_no))}",
            f"OPTC_{_safe_name(str(op_tc))}",
            f"RANGE_{_safe_name(str(range_tag))}",
            f"RSEC_{int(resample_sec)}",
            f"TSTEP_{int(timesteps)}",
        ]
    )

    model_path = os.path.join(out_dir, f"{prefix}__model.keras")
    scaler_path = os.path.join(out_dir, f"{prefix}__scaler.pkl")
    meta_path = os.path.join(out_dir, f"{prefix}__meta.json")  # runtime scans *_meta.json

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    meta = Phase2Meta(
        trained_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        wsuid=wsuid,
        stock_key=str(st_no),
        op_tc=str(op_tc),
        resample_sec=int(resample_sec),
        timesteps=int(timesteps),
        sensor_cols=[str(c) for c in cols],
        n_rows_raw=int(n_raw),
        n_rows_grid=int(n_rows_grid),
        n_train_seq=int(X_train.shape[0]),
        threshold=thr,
        model_path=model_path,
        scaler_path=scaler_path,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2, ensure_ascii=False)

    log.info(f"TRAIN_OK model={model_path}")
    log.info(f"TRAIN_OK meta={meta_path} threshold={thr:.6f}")
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plId", type=int, required=True)
    ap.add_argument("--wcId", type=int, required=True)
    ap.add_argument("--wsId", type=int, required=True)

    ap.add_argument("--jsonl", type=str, default="", help="Optional JSONL path(s) to train without Cassandra (comma-separated).")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--time_min", type=str, default=None, help="Optional start datetime for training window (naive local or with offset).")
    ap.add_argument("--time_max", type=str, default=None, help="Optional end datetime for training window (naive local or with offset).")
    ap.add_argument("--intervals_file", type=str, default=None, help="Optional file defining multiple training intervals (JSON/JSONL/CSV lines).")
    ap.add_argument("--stNo", type=str, default="ALL")
    ap.add_argument("--opTc", type=str, default="ALL")
    ap.add_argument("--resample_sec", type=int, default=60)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--min_rows", type=int, default=500)
    ap.add_argument("--min_cov", type=float, default=0.10)
    ap.add_argument("--max_sensors", type=int, default=32)
    ap.add_argument("--ffill_limit", type=int, default=0)
    ap.add_argument("--threshold_q", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--keras_verbose", type=int, default=0, help="Keras fit verbose (0/1/2). Default 0.")
    ap.add_argument("--progress", action="store_true", help="Emit periodic training progress logs (recommended for CPU runs).")
    ap.add_argument("--progress_every_batches", type=int, default=50, help="Log every N train batches when --progress is set.")
    ap.add_argument("--progress_every_sec", type=float, default=15.0, help="Also log if this many seconds passed since last progress log.")
    ap.add_argument("--out_dir", type=str, default="./models/phase2models/offline_unsup")
    args = ap.parse_args()
    if args.intervals_file and (args.time_min or args.time_max):
        raise SystemExit('ERROR: --intervals_file cannot be combined with --time_min/--time_max')
    if (args.time_min and not args.time_max) or (args.time_max and not args.time_min):
        raise SystemExit('ERROR: --time_min and --time_max must be provided together')

    train_phase2_unsup(
        pl_id=args.plId,
        wc_id=args.wcId,
        ws_id=args.wsId,
        jsonl=str(args.jsonl),
        days=args.days,
        st_no=args.stNo,
        op_tc=args.opTc,
        resample_sec=args.resample_sec,
        timesteps=args.timesteps,
        min_rows=args.min_rows,
        min_cov=args.min_cov,
        max_sensors=args.max_sensors,
        ffill_limit=args.ffill_limit,
        threshold_q=args.threshold_q,
        epochs=args.epochs,
        batch_size=args.batch_size,
        keras_verbose=args.keras_verbose,
        progress=bool(args.progress),
        progress_every_batches=args.progress_every_batches,
        progress_every_sec=args.progress_every_sec,
        time_min=args.time_min,
        time_max=args.time_max,
        intervals_file=args.intervals_file,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()