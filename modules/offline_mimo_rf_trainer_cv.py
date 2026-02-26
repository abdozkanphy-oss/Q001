"""modules/offline_mimo_rf_trainer.py

M4.x: Offline supervised Multi-Input / Multi-Output trainer (RandomForest + faster alternatives).

Why this file exists
--------------------
We need a production-safe offline trainer that can:
- Train ONE model per (workstation, selected stock) to predict multiple target sensors jointly (multi-output regression).
- Fetch from Cassandra WITHOUT relying on ALLOW FILTERING (query by WS partition keys + time range only).
- Be robust to irregular sampling (event-time anchored resampling + gap guard).
- Avoid cross-batch leakage via batch-holdout evaluation when batch IDs exist.
- Produce artifacts compatible with the rest of the system:
    * model serialized with joblib -> .pkl
    * sidecar metadata -> __meta.json
  NOTE: Artifact *types* MUST NOT change.

Key additions vs earlier draft
------------------------------
1) Faster ingestion: streaming fetch -> bulk-load DataFrame (no per-row dict building).
2) Model factory: --model_type in {RF,HGB,XGB}, with MultiOutput wrapper when needed.
3) Strict UTC: all timestamps are forced to UTC-aware; Cassandra range filters are sent as naive-UTC for driver safety.
4) Evaluation: time_split + batch_holdout (segment IDs ordered by time).
5) AutoML loop: grid over (resample_sec x n_lags x model_type) without refetching data.

CLI examples
------------
Single run:
python -m modules.offline_mimo_rf_trainer \
  --plId 149 --wcId 951 --wsId 441165 --stNo "Antares PV" \
  --time_min "2025-11-06 08:00:00.000000" --time_max "2025-12-06 08:00:00.000000" \
  --targets "S1,S2,S3" --n_lags 6 --resample_sec 60 --horizon_sec 60 \
  --model_type HGB --eval_mode batch_holdout --segment_field AUTO

AutoML:
python -m modules.offline_mimo_rf_trainer \
  --plId 149 --wcId 951 --wsId 441165 --stNo "Antares PV" \
  --days 21 --automl \
  --automl_resample_secs "30,60" --automl_n_lags "6,12" --automl_models "HGB,RF" \
  --eval_mode auto --segment_field AUTO
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# -----------------------------
# Helpers: tokens, UTC, parsing
# -----------------------------

_WSUID_RE = re.compile(r"^(?P<pl>\d+)_WC(?P<wc>\d+)_WS(?P<ws>\d+)$")


def _fallback_safe_token(s: str, *, default: str = "X") -> str:
    """Conservative filename tokenization (fallback only).

    NOTE: In the repo, we prefer modules.model_registry.safe_token to ensure
    registry/trainer tokenization is identical. This fallback is only used
    if that import is unavailable (e.g., isolated unit tests).
    """
    s = str(s or "").strip()
    if not s:
        return default
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    tok = "".join(out)
    tok = re.sub(r"_+", "_", tok).strip("_")
    return tok[:120] if tok else default


try:
    # IMPORTANT: must match registry tokenization exactly
    from modules.model_registry import safe_token as _safe_token  # type: ignore
except Exception:
    _safe_token = _fallback_safe_token  # type: ignore


def _hash_list(items: Sequence[str], *, n: int = 10) -> str:
    h = hashlib.sha1("|".join([str(x) for x in items]).encode("utf-8")).hexdigest()
    return h[: max(6, int(n))]


def _parse_wsuid(wsuid: str) -> Optional[Tuple[int, int, int]]:
    if not wsuid:
        return None
    m = _WSUID_RE.match(str(wsuid).strip())
    if not m:
        return None
    return int(m.group("pl")), int(m.group("wc")), int(m.group("ws"))


def ensure_utc(dt: Any) -> Optional[datetime]:
    """Return timezone-aware UTC datetime or None.

    - If dt is naive, it is assumed to be UTC (NOT local time).
    - If dt is str, parsed with pandas.
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        s = dt.strip()
        if not s:
            return None
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    if isinstance(dt, pd.Timestamp):
        if pd.isna(dt):
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).to_pydatetime()
        return dt.tz_convert("UTC").to_pydatetime()
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    # unknown -> best effort
    try:
        ts = pd.to_datetime(dt, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def to_naive_utc(dt: datetime) -> datetime:
    """Cassandra driver safety: send naive UTC datetime."""
    dt2 = ensure_utc(dt) or datetime.now(timezone.utc)
    return dt2.replace(tzinfo=None)


def _norm_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"0", "none", "null", "nan"}:
        return None
    return s


# -----------------------------
# Cassandra access (lazy import)
# -----------------------------

def _get_dw_model():
    """Lazy import to avoid Cassandra dependency during unit tests."""
    from cassandra_utils.models.dw_raw_by_ws import dw_tbl_raw_data_by_ws  # local import
    return dw_tbl_raw_data_by_ws


def probe_latest_measurement_date(pl_id: int, wc_id: int, ws_id: int) -> Optional[datetime]:
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
        return ensure_utc(getattr(rows[0], "measurement_date", None))
    except Exception:
        return None


@dataclass
class FetchSpec:
    pl_id: int
    wc_id: int
    ws_id: int
    time_min: Optional[datetime]
    time_max: Optional[datetime]
    days: int
    limit: int
    chunk_hours: int = 12


def iter_rows_by_ws_time_range(spec: FetchSpec) -> Iterable[Any]:
    """Yield rows from dw_tbl_raw_data_by_ws for WS partition + time range.

    IMPORTANT: We intentionally do NOT filter by stock/opTc at the Cassandra
    query level, to avoid relying on ALLOW FILTERING or secondary indexes.
    """
    dw_tbl_raw_data_by_ws = _get_dw_model()

    latest = probe_latest_measurement_date(spec.pl_id, spec.wc_id, spec.ws_id)
    end_dt = latest or ensure_utc(datetime.now(timezone.utc))

    if spec.time_max is not None:
        end_dt = min(end_dt, ensure_utc(spec.time_max) or end_dt)

    if spec.time_min is not None:
        start_dt = ensure_utc(spec.time_min)
    else:
        start_dt = end_dt - timedelta(days=int(spec.days))

    if start_dt is None or start_dt >= end_dt:
        return

    # Cassandra range filters: use naive UTC to avoid tz confusion in driver.
    start_dt_q = to_naive_utc(start_dt)
    end_dt_q = to_naive_utc(end_dt)

    # Adaptive per-chunk cap
    limit = int(spec.limit)
    unlimited = limit <= 0
    span_hours = max(1.0, (end_dt_q - start_dt_q).total_seconds() / 3600.0)
    n_chunks = max(1, int(np.ceil(span_hours / float(spec.chunk_hours))))
    per_chunk_limit = max(2000, int(limit // n_chunks)) if not unlimited else 10**9

    cur = start_dt_q
    yielded = 0
    while cur < end_dt_q and (unlimited or yielded < limit):
        nxt = min(end_dt_q, cur + timedelta(hours=int(spec.chunk_hours)))
        take = per_chunk_limit if unlimited else min(per_chunk_limit, limit - yielded)

        q = dw_tbl_raw_data_by_ws.objects.filter(
            plant_id=int(spec.pl_id),
            work_center_id=int(spec.wc_id),
            work_station_id=int(spec.ws_id),
            measurement_date__gte=cur,
            measurement_date__lt=nxt,
        )

        # Reduce payload when possible
        try:
            q = q.only(
                "measurement_date",
                "equipment_name",
                "counter_reading",
                "produced_stock_no",
                "produced_stock_name",
                "operationtaskcode",
                "prod_order_reference_no",
                "prod_order_reference_no_txt",
                "job_order_reference_no",
                "job_order_reference_no_txt",
                "job_order_operation_id",
                "job_order_operation_id_txt",
            )
        except Exception:
            pass

        for r in q.limit(int(take)):
            yield r
            yielded += 1
            if not unlimited and yielded >= limit:
                break

        cur = nxt


# ---------------------------------------
# Fast normalization: rows -> long DataFrame
# ---------------------------------------

_SEGMENT_CANDIDATES = [
    "prod_order_reference_no_txt",
    "prod_order_reference_no",
    "job_order_reference_no_txt",
    "job_order_reference_no",
    "job_order_operation_id_txt",
    "job_order_operation_id",
]



def _row_get(r: Any, name: str, default: Any = None) -> Any:
    """Get field from either a Cassandra row object or a dict row."""
    try:
        if isinstance(r, dict):
            return r.get(name, default)
        return getattr(r, name, default)
    except Exception:
        return default

def rows_to_long_df(
    rows: Iterable[Any],
    *,
    st_no: str,
    op_tc: str,
    segment_field_hint: str,
    extra_segment_fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Stream rows -> long df (ts,sensor,val + stock/op/segment fields).

    Performance notes:
    - We still need Python-level attribute reads, but we avoid per-row dict creation.
    - Filtering by stock/op is applied in Python during extraction (still compliant:
      NOT applied at Cassandra query level).
    """
    st_no = str(st_no or "ALL").strip()
    op_tc = str(op_tc or "ALL").strip()
    st_all = st_no.upper() == "ALL" or st_no == ""
    op_all = op_tc.upper() == "ALL" or op_tc == ""

    # Ensure we have the segment columns we might need for AUTO selection.
    seg_fields = set()
    if segment_field_hint and segment_field_hint.upper() not in {"", "AUTO"}:
        seg_fields.add(segment_field_hint)
    for s in (extra_segment_fields or []):
        if s:
            seg_fields.add(str(s))
    # Add candidates for AUTO so we can choose without refetching
    for s in _SEGMENT_CANDIDATES:
        seg_fields.add(s)

    ts_list: List[Any] = []
    sensor_list: List[Any] = []
    val_list: List[Any] = []
    stno_list: List[Any] = []
    stname_list: List[Any] = []
    optc_list: List[Any] = []
    seg_cols: Dict[str, List[Any]] = {s: [] for s in sorted(seg_fields)}

    # Localize getter for speed (supports dict rows from JSONL)
    _get = _row_get
    for r in rows:
        try:
            ts = _get(r, "measurement_date", None)
            sensor = _get(r, "equipment_name", None)
            val = _get(r, "counter_reading", None)

            st1 = _get(r, "produced_stock_no", None)
            st2 = _get(r, "produced_stock_name", None)
            op1 = _get(r, "operationtaskcode", None)

            if not st_all:
                if str(st1) != st_no and str(st2) != st_no:
                    continue
            if not op_all:
                if str(op1) != op_tc:
                    continue

            ts_list.append(ts)
            sensor_list.append(sensor)
            val_list.append(val)
            stno_list.append(st1)
            stname_list.append(st2)
            optc_list.append(op1)

            for s in seg_cols.keys():
                seg_cols[s].append(_get(r, s, None))
        except Exception:
            continue

    if not ts_list:
        return pd.DataFrame()

    data = {
        "ts": ts_list,
        "sensor": sensor_list,
        "val": val_list,
        "produced_stock_no": stno_list,
        "produced_stock_name": stname_list,
        "operationtaskcode": optc_list,
    }
    for s, vals in seg_cols.items():
        data[s] = vals

    df = pd.DataFrame(data)

    # Normalize types
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df["sensor"] = df["sensor"].fillna("").astype(str)
    df = df[df["sensor"].str.len() > 0]
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])

    for s in seg_cols.keys():
        df[s] = df[s].map(_norm_id)

    return df


def csv_to_long_df(
    csv_path: str,
    *,
    st_no: str,
    op_tc: str,
    segment_field_hint: str,
) -> pd.DataFrame:
    """Debug/test path: create the same long df from a CSV dump."""
    raw = pd.read_csv(csv_path)
    # Normalize columns to expected names if needed
    col_map = {}
    if "measurement_date" not in raw.columns and "measDt" in raw.columns:
        col_map["measDt"] = "measurement_date"
    if "equipment_name" not in raw.columns and "equipmentName" in raw.columns:
        col_map["equipmentName"] = "equipment_name"
    if col_map:
        raw = raw.rename(columns=col_map)

    # Filtering (vectorized)
    st_no = str(st_no or "ALL").strip()
    op_tc = str(op_tc or "ALL").strip()
    st_all = st_no.upper() == "ALL" or st_no == ""
    op_all = op_tc.upper() == "ALL" or op_tc == ""

    if not st_all:
        m = (raw.get("produced_stock_no").astype(str) == st_no) | (raw.get("produced_stock_name").astype(str) == st_no)
        raw = raw[m]
    if not op_all:
        raw = raw[raw.get("operationtaskcode").astype(str) == op_tc]

    # Build df with expected columns
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(raw.get("measurement_date"), utc=True, errors="coerce"),
            "sensor": raw.get("equipment_name").astype(str),
            "val": pd.to_numeric(raw.get("counter_reading"), errors="coerce"),
        }
    )

    # Segment candidates
    for s in _SEGMENT_CANDIDATES:
        if s in raw.columns:
            df[s] = raw[s].map(_norm_id)

    # Keep additional hint field
    if segment_field_hint and segment_field_hint not in {"AUTO", ""} and segment_field_hint in raw.columns:
        df[segment_field_hint] = raw[segment_field_hint].map(_norm_id)

    df = df.dropna(subset=["ts", "val"])
    df = df[df["sensor"].str.len() > 0]
    return df


# -----------------------------
# Segment policy (AUTO)
# -----------------------------

def select_segment_field_auto(
    df: pd.DataFrame,
    *,
    prefer_stock: bool = True,
    min_coverage: float = 0.60,
    min_avg_group_size: float = 25.0,
    max_distinct_ratio: float = 0.25,
) -> str:
    """Pick the best segment field for leakage-safe evaluation.

    Heuristic goal: prefer a column that:
    - exists and has high coverage
    - has a reasonable number of distinct IDs (not too fragmented)
    - yields groups large enough for holdout evaluation

    If nothing is suitable, return "SESSION".
    """
    if df is None or df.empty:
        return "SESSION"

    n = int(df.shape[0])
    if n <= 0:
        return "SESSION"

    candidates = [c for c in _SEGMENT_CANDIDATES if c in df.columns]
    if not candidates:
        return "SESSION"

    scores: List[Tuple[float, str, Dict[str, Any]]] = []
    for c in candidates:
        s = df[c]
        cov = float(s.notna().mean()) if n > 0 else 0.0
        if cov < float(min_coverage):
            continue
        # Distinct among non-null
        nn = s.dropna()
        if nn.empty:
            continue
        distinct = int(nn.nunique(dropna=True))
        distinct_ratio = float(distinct) / float(max(1, n))
        if distinct_ratio > float(max_distinct_ratio):
            continue
        avg_group = float(nn.shape[0]) / float(max(1, distinct))
        if avg_group < float(min_avg_group_size):
            continue

        # Prefer "prod_order" then "job_order" (common industrial semantics)
        prior = 0.0
        if "prod_order" in c:
            prior = 2.0
        elif "job_order_reference" in c:
            prior = 1.0
        elif "job_order_operation" in c:
            prior = 0.5

        # Score: higher is better
        score = prior + 2.0 * cov + 0.5 * np.log1p(avg_group) - 2.0 * distinct_ratio
        scores.append((float(score), c, {"coverage": cov, "distinct": distinct, "avg_group": avg_group, "distinct_ratio": distinct_ratio}))

    if not scores:
        return "SESSION"

    scores.sort(key=lambda x: x[0], reverse=True)
    return str(scores[0][1])


def build_session_ids(ts_index: pd.DatetimeIndex, *, session_gap_sec: int) -> pd.Series:
    """Sessionize timestamps by gap threshold."""
    if len(ts_index) == 0:
        return pd.Series(dtype="object")
    s = pd.Series(ts_index, index=ts_index).sort_index()
    gaps = s.diff().dt.total_seconds().fillna(0.0)
    new_sess = (gaps > float(session_gap_sec)).astype(int)
    sess_id = new_sess.cumsum()
    return sess_id.astype(str)


# -----------------------------
# Dataset building (MIMO supervised)
# -----------------------------

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


def _order_segments_by_time(df: pd.DataFrame, seg_col: str) -> List[str]:
    g = df.groupby(seg_col)["ts"].agg(["min", "max"]).reset_index()
    g = g.sort_values(["max", "min"], ascending=True)
    return [str(x) for x in g[seg_col].tolist()]


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
    sensors_fixed: Optional[List[str]] = None,
    pivot_agg: str = "last",
    resample_method: str = "last",
    inactive_strategy: str = "gap_guard",
    use_time_elapsed_feature: bool = False,
    include_time_elapsed_in_targets: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str], List[str], Dict[str, Any]]:
    """Build multi-target supervised dataset.

    Features:
      For each selected sensor s and lag k=1..n_lags => s__lag_k
    Targets:
      Each target sensor value at +horizon_steps
    Baseline:
      Persistence lag0 for each target at prediction time
    """
    if df is None or df.empty:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "empty_df"}

    targets = [str(t).strip() for t in targets if str(t).strip()]
    targets = list(dict.fromkeys(targets))
    if not targets:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_targets"}

    freq = f"{int(resample_sec)}s"
    horizon_steps = max(1, int(round(float(horizon_sec) / float(resample_sec))))

    df2 = df.copy()

    # Segment IDs: explicit field or SESSION
    if str(segment_field).upper() == "SESSION":
        # Use union of target timestamps to define sessions
        tdf = df2[df2["sensor"].isin(targets)][["ts"]].copy()
        if tdf.empty:
            return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_target_rows"}
        tdf = tdf.sort_values("ts")
        ts_sorted = pd.DatetimeIndex(tdf["ts"].tolist())
        sess_ids = build_session_ids(ts_sorted, session_gap_sec=int(session_gap_sec))
        sess_by_ts = pd.Series(sess_ids.values, index=ts_sorted)
        sess_by_ts = sess_by_ts[~sess_by_ts.index.duplicated(keep="last")].sort_index()

        all_ts = pd.DatetimeIndex(df2["ts"].tolist())
        pos = sess_by_ts.index.searchsorted(all_ts, side="right") - 1
        out = np.where(pos < 0, "MISSING", sess_by_ts.iloc[pos].astype(str).to_numpy())
        df2["seg_id"] = out
    else:
        if segment_field not in df2.columns:
            df2["seg_id"] = "MISSING"
        else:
            df2["seg_id"] = df2[segment_field].fillna("MISSING").astype(str)

    # Sensor selection (global)
    vc = df2["sensor"].value_counts()
    sensors_sorted = vc.index.tolist()
    sensors_used: List[str] = []

    # Optional: lock sensor set (useful to keep AutoML refit deterministic)
    if sensors_fixed:
        fixed = [str(s).strip() for s in sensors_fixed if str(s).strip()]
        fixed = list(dict.fromkeys(fixed))
        sensors_used = fixed[: max(1, int(max_sensors))]
        # Ensure all targets are included first
        for t in targets:
            if t not in sensors_used:
                sensors_used = [t] + [x for x in sensors_used if x != t]
        sensors_used = sensors_used[: max(1, int(max_sensors))]
        sensors_sorted = []
    for s in sensors_sorted:
        if int(vc.get(s, 0)) < int(min_sensor_non_nan):
            continue
        sensors_used.append(str(s))
        if len(sensors_used) >= int(max_sensors):
            break

    # Ensure all targets are included first
    for t in targets:
        if t not in sensors_used:
            sensors_used = [t] + [x for x in sensors_used if x != t]
    sensors_used = sensors_used[: max(1, int(max_sensors))]


    # Optional synthetic time feature (legacy scada_product_training style).
    # WARNING: Phase3 predictor must provide this sensor at inference time if used online.
    if bool(use_time_elapsed_feature):
        meta_name = "meta_time_elapsed"
        if meta_name not in sensors_used:
            sensors_used.append(meta_name)
        # If targets were auto-selected and user explicitly wants to predict it too
        if bool(include_time_elapsed_in_targets) and meta_name not in targets:
            targets = targets + [meta_name]
    if not sensors_used:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, [], [], {"reason": "no_sensors_selected"}

    # Feature names deterministic
    feature_names: List[str] = [f"{s}__lag_{k}" for s in sensors_used for k in range(1, int(n_lags) + 1)]
    expected_dim = int(len(feature_names))

    # Order segments by time (critical for batch_holdout and time_split consistency)
    segments = _order_segments_by_time(df2, "seg_id")
    seg_stats = {"segments_total": int(len(segments)), "segments_used": 0}

    X_parts: List[np.ndarray] = []
    Y_parts: List[np.ndarray] = []
    B0_parts: List[np.ndarray] = []
    SEG_parts: List[np.ndarray] = []

    # Pre-calc target indices in sensors_used
    target_idx = [sensors_used.index(t) for t in targets if t in sensors_used]
    if len(target_idx) != len(targets):
        # Should not happen due to earlier enforcement, but keep safe
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), None, feature_names, sensors_used, {"reason": "target_missing_in_sensors"}

    for seg_id in segments:
        sdf = df2[df2["seg_id"] == seg_id]
        if sdf.empty:
            continue

        # long -> wide
        tmp = sdf[["ts", "sensor", "val"]].sort_values("ts")

        if str(pivot_agg).lower() == "mean":
            # mean across duplicate (ts,sensor)
            piv = tmp.groupby(["ts", "sensor"], sort=False)["val"].mean().unstack("sensor")
        else:
            # default: keep last per (ts,sensor)
            tmp = tmp.drop_duplicates(["ts", "sensor"], keep="last")
            piv = tmp.pivot(index="ts", columns="sensor", values="val")

        if piv is None or piv.empty:
            continue

        piv = piv.sort_index()

        # resample
        if str(resample_method).lower() == "mean":
            rr = piv.resample(freq).mean()
        else:
            rr = piv.resample(freq).last()

        # Optional synthetic time feature (per-segment)
        if bool(use_time_elapsed_feature):
            try:
                t0 = rr.index.min()
                rr["meta_time_elapsed"] = (rr.index - t0).total_seconds().astype("float32")
            except Exception:
                rr["meta_time_elapsed"] = 0.0

        # Ensure columns exist and order
        for s in sensors_used:
            if s not in rr.columns:
                rr[s] = np.nan
        rr = rr[sensors_used]

        # Target support (use raw non-nan counts before any legacy fill)
        tgt_non_nan = [int(rr[t].notna().sum()) for t in targets]
        if tgt_non_nan and int(min(tgt_non_nan)) < int(min_sensor_non_nan):
            continue

        # Missing/inactive handling
        strat = str(inactive_strategy or "gap_guard").lower()
        if strat == "zero":
            rr = rr.fillna(0.0)
        elif strat == "ffill_zero":
            rr = rr.ffill().fillna(0.0)
        else:
            # gap_guard (default, safer)
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

        # Convert to numpy for fast lag building
        V = rr.to_numpy(dtype="float32")  # shape (T, n_sensors)
        T, S = V.shape

        # Need at least one supervised sample with full history and horizon
        if T < (int(n_lags) + int(horizon_steps)):
            continue

        # Build lag tensor with sensor-major layout:
        # lag_1 = current(t), lag_2 = t-1, ..., lag_n = t-(n-1)
        # X3: (T, S, n_lags) where last axis is lag index (0..n_lags-1)
        X3 = np.full((T, S, int(n_lags)), np.nan, dtype="float32")
        for d in range(int(n_lags)):  # d=0 => current
            X3[d:, :, d] = V[: T - d, :]

        # Flatten to (T, S*n_lags) in sensor-major order (matches feature_names you already build)
        Xmat = X3.reshape(T, S * int(n_lags))

        # Targets and baseline (aligned on prediction time t)
        Ymat = np.vstack([V[int(horizon_steps) :, i] for i in target_idx]).T  # (T-h, n_targets)
        B0mat = np.vstack([V[: T - int(horizon_steps), i] for i in target_idx]).T  # (T-h, n_targets)
        Xmat = Xmat[: T - int(horizon_steps), :]

        if Xmat.shape[0] != Ymat.shape[0]:
            continue

        # Mask: require finite targets AND finite baseline AND finite features
        mask = np.isfinite(Ymat).all(axis=1)
        mask &= np.isfinite(B0mat).all(axis=1)
        mask &= np.isfinite(Xmat).all(axis=1)

        if int(mask.sum()) < int(min_sensor_non_nan):
            continue

        X = Xmat[mask].astype("float32", copy=False)
        Y = Ymat[mask].astype("float32", copy=False)
        B0 = B0mat[mask].astype("float32", copy=False)

        if X.shape[1] != expected_dim:
            continue

        X_parts.append(X)
        Y_parts.append(Y)
        B0_parts.append(B0)
        SEG_parts.append(np.asarray([str(seg_id)] * X.shape[0], dtype=object))
        seg_stats["segments_used"] = int(seg_stats["segments_used"]) + 1

    if not X_parts:
        return (
            np.empty((0, 0)),
            np.empty((0, 0)),
            np.empty((0, 0)),
            None,
            feature_names,
            sensors_used,
            {"reason": "no_segments_trainable", **seg_stats},
        )

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


# -----------------------------
# Evaluation & metrics
# -----------------------------

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


def _baseline_perfect(base_mae: List[float], eps: float) -> bool:
    return bool(base_mae) and all(np.isfinite(x) and float(x) <= float(eps) for x in base_mae)


def _lift(base_mae: List[float], model_mae: List[float]) -> List[float]:
    out = []
    for a, b in zip(base_mae, model_mae):
        try:
            out.append(float(a) - float(b))
        except Exception:
            out.append(float("nan"))
    return out


def _mean_nan(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.nanmean(np.asarray(xs, dtype="float64")))


# -----------------------------
# Model factory + tuning
# -----------------------------

def _try_import_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


def make_estimator(model_type: str, *, random_state: int, n_jobs: int) -> Tuple[Any, Dict[str, List[Any]], bool]:
    """Return (estimator, param_grid, needs_multioutput_wrapper)."""
    mt = str(model_type or "RF").strip().upper()

    if mt == "RF":
        est = RandomForestRegressor(
            n_estimators=300,
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            max_depth=None,
            min_samples_leaf=1,
        )
        grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 12, 24],
            "model__min_samples_leaf": [1, 2, 4],
        }
        return est, grid, False

    if mt == "HGB":
        est = HistGradientBoostingRegressor(
            random_state=int(random_state),
            max_depth=None,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.0,
            early_stopping=True,
        )
        grid = {
            # Note: MultiOutputRegressor => model__estimator__*
            "model__estimator__max_depth": [None, 8, 16],
            "model__estimator__learning_rate": [0.03, 0.05, 0.1],
            "model__estimator__max_iter": [200, 400],
        }
        return est, grid, True

    if mt == "XGB":
        xgb = _try_import_xgb()
        if xgb is None:
            raise RuntimeError("xgboost is not installed, but --model_type XGB was requested.")
        est = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            tree_method="hist",
        )
        grid = {
            "model__estimator__n_estimators": [400, 800],
            "model__estimator__max_depth": [6, 10],
            "model__estimator__learning_rate": [0.03, 0.05, 0.1],
            "model__estimator__subsample": [0.8, 1.0],
        }
        return est, grid, True

    raise ValueError(f"Unknown model_type: {model_type}")


def make_pipeline(model_type: str, *, random_state: int, n_jobs: int, use_robust_scaler: bool, rf_multioutput_mode: str) -> Tuple[Pipeline, Dict[str, List[Any]], Dict[str, Any]]:
    est, grid, wrap = make_estimator(model_type, random_state=random_state, n_jobs=n_jobs)
    wrapped = MultiOutputRegressor(est, n_jobs=int(n_jobs)) if wrap else est
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", wrapped)])
    info = {"model_type": str(model_type).upper(), "wrapped_multioutput": bool(wrap)}
    return pipe, grid, info


def _unique_preserve_order(vals: Iterable[Any]) -> List[Any]:
    seen: set = set()
    out: List[Any] = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _groups_are_single_blocks(groups: np.ndarray) -> bool:
    """Return True if each group appears in a single contiguous block (no re-appears later)."""
    if groups is None:
        return False
    g = np.asarray(groups, dtype=object)
    if g.size == 0:
        return False
    closed: set = set()
    prev = g[0]
    for x in g[1:]:
        if x != prev:
            closed.add(prev)
            prev = x
            if prev in closed:
                return False
    return True


def _make_ordered_batch_splits(groups: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Chronological batch CV: expanding window over batch blocks.

    Assumes groups are roughly time-ordered. If groups are contiguous blocks (typical for our dataset builder),
    splits become simple contiguous ranges for speed. Otherwise falls back to isin-based index selection.
    """
    if groups is None:
        return []
    g = np.asarray(groups, dtype=object)
    n = int(g.size)
    if n <= 0:
        return []

    segs = _unique_preserve_order(g.tolist())
    n_segs = int(len(segs))
    if n_segs < 3:
        return []

    n_splits = int(max(2, min(6, n_splits, n_segs - 1)))

    # pick a segment-step so that we can create up to n_splits folds
    step = max(1, n_segs // (n_splits + 1))

    contiguous = _groups_are_single_blocks(g)
    blocks: Optional[List[Tuple[Any, int, int]]] = None
    if contiguous:
        blocks = []
        start = 0
        prev = g[0]
        for i in range(1, n):
            if g[i] != prev:
                blocks.append((prev, start, i))
                start = i
                prev = g[i]
        blocks.append((prev, start, n))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(1, n_splits + 1):
        train_seg_end = min(n_segs - 1, s * step)
        test_seg_end = min(n_segs, (s + 1) * step)

        if test_seg_end <= train_seg_end:
            continue

        train_segs = segs[:train_seg_end]
        test_segs = segs[train_seg_end:test_seg_end]
        if not test_segs:
            continue

        if contiguous and blocks is not None and len(blocks) == n_segs:
            train_end_row = int(blocks[train_seg_end - 1][2])
            test_end_row = int(blocks[test_seg_end - 1][2])
            if test_end_row - train_end_row <= 0:
                continue
            train_idx = np.arange(0, train_end_row, dtype=int)
            test_idx = np.arange(train_end_row, test_end_row, dtype=int)
        else:
            # fallback (slower): select by group membership
            train_idx = np.flatnonzero(np.isin(g, np.asarray(train_segs, dtype=object))).astype(int, copy=False)
            test_idx = np.flatnonzero(np.isin(g, np.asarray(test_segs, dtype=object))).astype(int, copy=False)
            if train_idx.size == 0 or test_idx.size == 0:
                continue

        splits.append((train_idx, test_idx))

    return splits


def fit_with_optional_gridsearch(
    pipe: Pipeline,
    param_grid: Dict[str, List[Any]],
    *,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    enable: bool,
    cv_splits: int,
    cv_mode: str,
    groups: Optional[np.ndarray],
    n_jobs: int,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fit pipeline, optionally with GridSearchCV using the selected CV strategy.

    cv_mode:
      - 'tscv': TimeSeriesSplit on row order (fastest, but not batch-safe)
      - 'batch_kfold': GroupKFold over batch ids (leakage-safe, but ignores order)
      - 'ordered_batch': chronological batch CV (leakage-safe + order-aware)
    """
    if not enable:
        pipe.fit(Xtr, Ytr)
        return pipe, {"tuned": False}

    cv_splits_in = int(cv_splits)
    cv_splits = int(max(2, min(6, cv_splits_in)))
    cv_mode_in = str(cv_mode or "tscv").strip().lower()

    cv = None
    cv_used = "tscv"
    n_groups = None

    if cv_mode_in == "batch_kfold" and groups is not None:
        g = np.asarray(groups, dtype=object)
        uniq = np.unique(g)
        n_groups = int(uniq.size)
        n_splits_eff = int(min(cv_splits, n_groups))
        if n_splits_eff >= 2:
            gkf = GroupKFold(n_splits=n_splits_eff)
            cv = gkf.split(Xtr, Ytr, groups=g)
            cv_used = "batch_kfold"
            cv_splits = n_splits_eff

    elif cv_mode_in == "ordered_batch" and groups is not None:
        splits = _make_ordered_batch_splits(groups, cv_splits)
        if len(splits) >= 2:
            cv = splits
            cv_used = "ordered_batch"
            cv_splits = int(len(splits))
            # n_groups not needed

    if cv is None:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv = tscv
        cv_used = "tscv"

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=int(n_jobs),
        refit=True,
        verbose=0,
    )
    gs.fit(Xtr, Ytr)
    best: Pipeline = gs.best_estimator_
    return best, {
        "tuned": True,
        "cv_mode_requested": str(cv_mode_in),
        "cv_mode_used": str(cv_used),
        "cv_splits_requested": int(cv_splits_in),
        "cv_splits_used": int(cv_splits),
        "n_groups": int(n_groups) if n_groups is not None else None,
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
    }


# -----------------------------
# Split strategies
# -----------------------------

def eval_time_split(
    X: np.ndarray,
    Y: np.ndarray,
    B0: np.ndarray,
    seg: Optional[np.ndarray] = None,
    *,
    pipe: Pipeline,
    grid: Dict[str, List[Any]],
    tune: bool,
    cv_splits: int,
    cv_mode: str = "tscv",
    n_jobs: int,
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
    if _baseline_perfect(base_mae, baseline_perfect_eps):
        return {
            "ok": False,
            "reason": "baseline_perfect",
            "baseline_mae_lag0_per_target": base_mae,
            "n_train": int(n_train),
            "n_test": int(n_test),
        }

    groups_tr: Optional[np.ndarray] = None
    if seg is not None:
        try:
            if int(np.asarray(seg).shape[0]) == int(n):
                groups_tr = np.asarray(seg[:n_train], dtype=object)
        except Exception:
            groups_tr = None

    fitted, tune_info = fit_with_optional_gridsearch(
        pipe,
        grid,
        Xtr=Xtr,
        Ytr=Ytr,
        enable=bool(tune),
        cv_splits=int(cv_splits),
        cv_mode=str(cv_mode),
        groups=groups_tr,
        n_jobs=int(n_jobs),
    )
    pred = fitted.predict(Xte)

    model_mae = _mae_per_target(Yte, pred)
    lift = _lift(base_mae, model_mae)

    return {
        "ok": True,
        "eval_mode": "time_split",
        "n_train": int(n_train),
        "n_test": int(n_test),
        "baseline_mae_lag0_per_target": base_mae,
        "model_mae_per_target": model_mae,
        "lift_vs_lag0_per_target": lift,
        "lift_vs_lag0_mean": _mean_nan(lift),
        "tuning": tune_info,
        "pipeline": fitted,
    }


def eval_batch_holdout(
    X: np.ndarray,
    Y: np.ndarray,
    B0: np.ndarray,
    seg: np.ndarray,
    *,
    pipe: Pipeline,
    grid: Dict[str, List[Any]],
    tune: bool,
    cv_splits: int,
    cv_mode: str = "tscv",
    n_jobs: int,
    holdout_k: int,
    min_points_per_batch: int,
    min_test: int,
    baseline_perfect_eps: float,
) -> Dict[str, Any]:
    if seg is None or len(seg) != int(Y.shape[0]):
        return {"ok": False, "reason": "no_seg"}

    seg = np.asarray(seg, dtype=object)

    # Segment order by appearance already follows time if dataset builder ordered segments by time.
    uniq = []
    seen = set()
    for s in seg.tolist():
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    if len(uniq) < int(holdout_k) + 1:
        return {"ok": False, "reason": "too_few_segments", "n_segments": int(len(uniq))}

    test_segs = list(uniq[-int(holdout_k) :])
    test_set = set(test_segs)

    m_test = np.array([s in test_set for s in seg.tolist()], dtype=bool)
    m_train = ~m_test

    # Enforce per-batch point minimum for test segments
    for s in test_segs:
        if int(np.sum(seg[m_test] == s)) < int(min_points_per_batch):
            return {"ok": False, "reason": "holdout_too_small"}
    if int(np.sum(m_test)) < int(min_test):
        return {"ok": False, "reason": "holdout_too_small"}

    Xtr, Xte = X[m_train], X[m_test]
    Ytr, Yte = Y[m_train], Y[m_test]
    B0te = B0[m_test]

    base_mae = _mae_per_target(Yte, B0te)
    if _baseline_perfect(base_mae, baseline_perfect_eps):
        return {
            "ok": False,
            "reason": "baseline_perfect",
            "baseline_mae_lag0_per_target": base_mae,
            "n_train": int(Ytr.shape[0]),
            "n_test": int(Yte.shape[0]),
        }

    groups_tr: Optional[np.ndarray] = None
    try:
        groups_tr = np.asarray(seg, dtype=object)[m_train]
    except Exception:
        groups_tr = None

    fitted, tune_info = fit_with_optional_gridsearch(
        pipe,
        grid,
        Xtr=Xtr,
        Ytr=Ytr,
        enable=bool(tune),
        cv_splits=int(cv_splits),
        cv_mode=str(cv_mode),
        groups=groups_tr,
        n_jobs=int(n_jobs),
    )
    pred = fitted.predict(Xte)

    model_mae = _mae_per_target(Yte, pred)
    lift = _lift(base_mae, model_mae)

    return {
        "ok": True,
        "eval_mode": "batch_holdout",
        "n_train": int(Ytr.shape[0]),
        "n_test": int(Yte.shape[0]),
        "holdout_k": int(holdout_k),
        "test_segments": sorted([str(s) for s in test_set]),
        "baseline_mae_lag0_per_target": base_mae,
        "model_mae_per_target": model_mae,
        "lift_vs_lag0_per_target": lift,
        "lift_vs_lag0_mean": _mean_nan(lift),
        "tuning": tune_info,
        "pipeline": fitted,
    }


# -----------------------------
# AutoML loop
# -----------------------------

def _parse_int_list(s: str, *, default: List[int]) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return default
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out or default


def _parse_str_list(s: str, *, default: List[str]) -> List[str]:
    s = str(s or "").strip()
    if not s:
        return default
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out or default


def run_automl(
    df_long: pd.DataFrame,
    *,
    targets: List[str],
    segment_field: str,
    session_gap_sec: int,
    horizon_sec: int,
    resample_secs: List[int],
    n_lags_list: List[int],
    model_types: List[str],
    max_sensors: int,
    min_sensor_non_nan: int,
    # legacy-blend knobs (MUST be passed from main)
    pivot_agg: str,
    resample_method: str,
    inactive_strategy: str,
    use_time_elapsed_feature: bool,
    include_time_elapsed_in_targets: bool,
    use_robust_scaler: bool,
    rf_multioutput_mode: str,
    # eval/tuning knobs
    min_rows: int,
    eval_mode: str,
    holdout_k: int,
    min_points_per_batch: int,
    min_test: int,
    baseline_perfect_eps: float,
    accept_min_lift_mean: float,
    tune: bool,
    cv_splits: int,
    cv_mode: str,
    n_jobs: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (best_result, leaderboard)."""

    leaderboard: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    pivot_agg = str(pivot_agg or "last")
    resample_method = str(resample_method or "last")
    inactive_strategy = str(inactive_strategy or "gap_guard")
    rf_multioutput_mode = str(rf_multioutput_mode or "native")

    for rsec in resample_secs:
        for nl in n_lags_list:
            # Dataset once per (rsec, nl)
            X, Y, B0, seg, feature_names, sensors_used, info = build_multitarget_dataset(
                df_long,
                targets=targets,
                resample_sec=int(rsec),
                horizon_sec=int(horizon_sec),
                n_lags=int(nl),
                segment_field=str(segment_field),
                session_gap_sec=int(session_gap_sec),
                max_sensors=int(max_sensors),
                min_sensor_non_nan=int(min_sensor_non_nan),
                pivot_agg=pivot_agg,
                resample_method=resample_method,
                inactive_strategy=inactive_strategy,
                use_time_elapsed_feature=bool(use_time_elapsed_feature),
                include_time_elapsed_in_targets=bool(include_time_elapsed_in_targets),
            )

            if int(Y.shape[0]) < int(min_rows):
                leaderboard.append(
                    {
                        "ok": False,
                        "reason": "insufficient_rows",
                        "resample_sec": int(rsec),
                        "horizon_sec": int(horizon_sec),
                        "n_lags": int(nl),
                        "n_supervised": int(Y.shape[0]),
                        "model_type": None,
                    }
                )
                continue

            for mt in model_types:
                pipe, grid, model_info = make_pipeline(
                    mt,
                    random_state=42,
                    n_jobs=int(n_jobs),
                    use_robust_scaler=bool(use_robust_scaler),
                    rf_multioutput_mode=rf_multioutput_mode,
                )

                if eval_mode == "batch_holdout":
                    ev = eval_batch_holdout(
                        X,
                        Y,
                        B0,
                        seg,
                        pipe=pipe,
                        grid=grid,
                        tune=bool(tune),
                        cv_splits=int(cv_splits),
                        cv_mode=str(cv_mode),
                        n_jobs=int(n_jobs),
                        holdout_k=int(holdout_k),
                        min_points_per_batch=int(min_points_per_batch),
                        min_test=int(min_test),
                        baseline_perfect_eps=float(baseline_perfect_eps),
                    )
                elif eval_mode == "time_split":
                    ev = eval_time_split(
                        X,
                        Y,
                        B0,
                        seg=seg,
                        cv_mode=str(cv_mode),
                        pipe=pipe,
                        grid=grid,
                        tune=bool(tune),
                        cv_splits=int(cv_splits),
                        n_jobs=int(n_jobs),
                        min_test=int(min_test),
                        baseline_perfect_eps=float(baseline_perfect_eps),
                    )
                else:
                    # auto
                    if seg is not None and str(segment_field).upper() != "SESSION":
                        ev = eval_batch_holdout(
                            X,
                            Y,
                            B0,
                            seg,
                            pipe=pipe,
                            grid=grid,
                            tune=bool(tune),
                            cv_splits=int(cv_splits),
                            cv_mode=str(cv_mode),
                            n_jobs=int(n_jobs),
                            holdout_k=int(holdout_k),
                            min_points_per_batch=int(min_points_per_batch),
                            min_test=int(min_test),
                            baseline_perfect_eps=float(baseline_perfect_eps),
                        )
                        if not ev.get("ok"):
                            ev = eval_time_split(
                                X,
                                Y,
                                B0,
                                seg=seg,
                                cv_mode=str(cv_mode),
                                pipe=pipe,
                                grid=grid,
                                tune=bool(tune),
                                cv_splits=int(cv_splits),
                                n_jobs=int(n_jobs),
                                min_test=int(min_test),
                                baseline_perfect_eps=float(baseline_perfect_eps),
                            )
                    else:
                        ev = eval_time_split(
                            X,
                            Y,
                            B0,
                            seg=seg,
                            cv_mode=str(cv_mode),
                            pipe=pipe,
                            grid=grid,
                            tune=bool(tune),
                            cv_splits=int(cv_splits),
                            n_jobs=int(n_jobs),
                            min_test=int(min_test),
                            baseline_perfect_eps=float(baseline_perfect_eps),
                        )

                rec: Dict[str, Any] = {
                    "ok": bool(ev.get("ok")),
                    "reason": ev.get("reason", ""),
                    "eval_mode": ev.get("eval_mode", ""),
                    "resample_sec": int(rsec),
                    "horizon_sec": int(horizon_sec),
                    "n_lags": int(nl),
                    "model_type": model_info.get("model_type"),
                    "wrapped_multioutput": bool(model_info.get("wrapped_multioutput")),
                    "n_supervised": int(Y.shape[0]),
                    "segments_used": int(info.get("segments_used", 0)),
                    "lift_mean": float(ev.get("lift_vs_lag0_mean")) if ev.get("ok") else float("nan"),
                    "accepted": False,
                }
                if ev.get("ok"):
                    lm = float(ev.get("lift_vs_lag0_mean", float("nan")))
                    rec["accepted"] = bool(np.isfinite(lm) and lm >= float(accept_min_lift_mean))

                leaderboard.append(rec)

                if rec["ok"] and rec["accepted"]:
                    if best is None or float(rec["lift_mean"]) > float(best.get("lift_mean", float("-inf"))):
                        best = {
                            **rec,
                            "pipeline": ev.get("pipeline"),
                            "feature_names": feature_names,
                            "sensors_used": sensors_used,
                            "targets": targets,
                            "data_info": info,
                        }

    if best is None:
        return {"ok": False, "reason": "no_accepted_candidate"}, leaderboard
    return best, leaderboard

# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Offline MIMO trainer (RF + HGB + XGB) per WS + stock.")
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
    ap.add_argument("--limit", type=int, default=0, help="0 => no hard limit (be careful).")
    ap.add_argument("--chunk_hours", type=int, default=12)

    # debug/test input
    ap.add_argument("--input_csv", default="", help="Optional CSV dump to train without Cassandra.")

    ap.add_argument("--jsonl", default="", help="Optional JSONL Kafka dump to train without Cassandra (comma-separated paths).")

    # model/data controls
    ap.add_argument("--model_type", default="RF", choices=["RF", "HGB", "XGB"])
    ap.add_argument("--targets", default="", help="Comma-separated target sensors; empty => auto")
    ap.add_argument("--max_targets", type=int, default=15)

    ap.add_argument("--n_lags", type=int, default=6)
    ap.add_argument("--resample_sec", type=int, default=60)
    ap.add_argument("--horizon_sec", type=int, default=60)

    # legacy-inspired knobs (scada_product_training.py compatibility)
    ap.add_argument("--pivot_agg", default="last", choices=["last", "mean"], help="How to aggregate duplicate (ts,sensor) rows before pivot.")
    ap.add_argument("--resample_method", default="last", choices=["last", "mean"], help="How to aggregate within each resample bin.")
    ap.add_argument(
        "--inactive_strategy",
        default="gap_guard",
        choices=["gap_guard", "ffill_zero", "zero"],
        help="gap_guard: age-based drop + limited ffill (default, safer). ffill_zero: full ffill then fill remaining with 0 (legacy-like). zero: fill all missing with 0.",
    )
    ap.add_argument("--use_robust_scaler", type=int, default=0, help="1 => add RobustScaler() before model (legacy-like).")
    ap.add_argument(
        "--rf_multioutput_mode",
        default="native",
        choices=["native", "wrapper"],
        help="native: RandomForestRegressor multioutput (default). wrapper: MultiOutputRegressor(RF) (legacy-like).",
    )
    ap.add_argument("--use_time_elapsed_feature", type=int, default=0, help="1 => add synthetic 'meta_time_elapsed' feature per segment (requires predictor support if used online).")
    ap.add_argument("--include_time_elapsed_in_targets", type=int, default=0, help="1 => include meta_time_elapsed also as a target when targets are auto-selected.")

    ap.add_argument("--segment_field", default="AUTO", help="AUTO | SESSION | specific field name")
    ap.add_argument("--session_gap_sec", type=int, default=6 * 3600)

    ap.add_argument("--max_sensors", type=int, default=50)
    ap.add_argument("--min_sensor_non_nan", type=int, default=10)
    ap.add_argument("--min_rows", type=int, default=200, help="Minimum supervised rows required")

    # evaluation
    ap.add_argument("--eval_mode", default="auto", choices=["auto", "time_split", "batch_holdout"])
    ap.add_argument("--holdout_k", type=int, default=3)
    ap.add_argument("--min_points_per_batch", type=int, default=50)
    ap.add_argument("--min_test", type=int, default=50)
    ap.add_argument("--baseline_perfect_eps", type=float, default=1e-12)
    ap.add_argument("--accept_min_lift_mean", type=float, default=-90.0)

    # tuning
    ap.add_argument("--tune", action="store_true", help="Enable GridSearchCV(TimeSeriesSplit) on training split.")
    ap.add_argument("--cv_splits", type=int, default=4)
    ap.add_argument(
        "--cv_mode",
        default="tscv",
        choices=["tscv", "batch_kfold", "ordered_batch"],
        help="CV strategy for --tune: tscv(TimeSeriesSplit) | batch_kfold(GroupKFold over batches) | ordered_batch(chronological batch CV)",
    )
    ap.add_argument("--n_jobs", type=int, default=-1)

    # automl
    ap.add_argument("--automl", action="store_true", help="Run AutoML loop and save best model.")
    ap.add_argument("--automl_resample_secs", default="30,60")
    ap.add_argument("--automl_n_lags", default="6,12")
    ap.add_argument("--automl_models", default="HGB,RF")

    # outputs

    # final fit (recommended): refit the chosen pipeline on ALL supervised rows after evaluation
    ap.add_argument("--final_fit_on_all", type=int, default=1, help="1 => refit on all rows after evaluation before saving (recommended).")

    # outputs
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

    wsuid = f"{pl_id}_WC{wc_id}_WS{ws_id}"

    out_dir = str(args.out_dir or "./models/offline_mimo_rf")
    os.makedirs(out_dir, exist_ok=True)

    run_id = str(args.run_id or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Parse time range (UTC)
    time_min = ensure_utc(args.time_min) if str(args.time_min).strip() else None
    time_max = ensure_utc(args.time_max) if str(args.time_max).strip() else None

    # Load long df (Cassandra or CSV)
    segment_hint = str(args.segment_field or "AUTO").strip()

    if str(args.jsonl).strip():
        # JSONL dump mode (Kafka recordings)
        from modules.offline_jsonl_source import jsonl_probe_latest_dt, jsonl_to_rows, parse_dt_like_to_utc

        if str(args.time_min).strip() and str(args.time_max).strip():
            j_start = parse_dt_like_to_utc(str(args.time_min))
            j_end = parse_dt_like_to_utc(str(args.time_max))
        else:
            latest = jsonl_probe_latest_dt(str(args.jsonl), pl_id=int(pl_id), wc_id=int(wc_id), ws_id=int(ws_id), st_no=str(args.stNo), op_tc=str(args.opTc))
            j_end = latest if latest is not None else datetime.now(timezone.utc)
            j_start = j_end - timedelta(days=int(args.days))

        rows = jsonl_to_rows(
            str(args.jsonl),
            pl_id=int(pl_id),
            wc_id=int(wc_id),
            ws_id=int(ws_id),
            start_dt=j_start,
            end_dt=j_end,
            limit=int(args.limit) if int(args.limit) > 0 else 0,
            st_no=str(args.stNo),
            op_tc=str(args.opTc),
        )
        df_long = rows_to_long_df(
            rows,
            st_no=str(args.stNo),
            op_tc=str(args.opTc),
            segment_field_hint=segment_hint,
            extra_segment_fields=_SEGMENT_CANDIDATES,
        )
    elif str(args.input_csv).strip():
        df_long = csv_to_long_df(str(args.input_csv), st_no=str(args.stNo), op_tc=str(args.opTc), segment_field_hint=segment_hint)
    else:
        spec = FetchSpec(
            pl_id=pl_id,
            wc_id=wc_id,
            ws_id=ws_id,
            time_min=time_min,
            time_max=time_max,
            days=int(args.days),
            limit=int(args.limit),
            chunk_hours=int(args.chunk_hours),
        )
        rows_iter = iter_rows_by_ws_time_range(spec)
        df_long = rows_to_long_df(
            rows_iter,
            st_no=str(args.stNo),
            op_tc=str(args.opTc),
            segment_field_hint=segment_hint,
            extra_segment_fields=_SEGMENT_CANDIDATES,
        )

    if df_long is None or df_long.empty:
        print("No usable rows after fetch/filter/normalization.")
        return 2

    # Choose segment field
    segment_field = segment_hint
    segment_policy: Optional[Dict[str, Any]] = None
    if segment_field.upper() == "AUTO":
        chosen = select_segment_field_auto(df_long)
        segment_field = chosen
        segment_policy = {"mode": "AUTO", "chosen": chosen, "candidates": [c for c in _SEGMENT_CANDIDATES if c in df_long.columns]}
    elif segment_field.upper() not in {"SESSION"} and segment_field not in df_long.columns:
        print(f"segment_field='{segment_field}' not present; falling back to SESSION.")
        segment_field = "SESSION"

    # Targets
    if str(args.targets).strip():
        targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    else:
        vc = df_long["sensor"].value_counts()
        targets = vc.head(int(args.max_targets)).index.tolist()
    targets = [t for t in targets if t in set(df_long["sensor"].unique().tolist())]
    targets = list(dict.fromkeys([str(t) for t in targets]))
    if len(targets) < 1:
        print("No usable targets selected.")
        return 3

    # Single-run vs AutoML
    if bool(args.automl):
        resample_secs = _parse_int_list(args.automl_resample_secs, default=[30, 60])
        n_lags_list = _parse_int_list(args.automl_n_lags, default=[6, 12])
        model_types = [s.strip().upper() for s in _parse_str_list(args.automl_models, default=["HGB", "RF"])]
        model_types = [m for m in model_types if m in {"RF", "HGB", "XGB"}]

        best, leaderboard = run_automl(
            df_long,
            targets=targets,
            segment_field=str(segment_field),
            session_gap_sec=int(args.session_gap_sec),
            horizon_sec=int(args.horizon_sec),
            resample_secs=resample_secs,
            n_lags_list=n_lags_list,
            model_types=model_types,
            max_sensors=int(args.max_sensors),
            min_sensor_non_nan=int(args.min_sensor_non_nan),
            pivot_agg=str(getattr(args, "pivot_agg", "last")),
            resample_method=str(getattr(args, "resample_method", "last")),
            inactive_strategy=str(getattr(args, "inactive_strategy", "gap_guard")),
            use_time_elapsed_feature=bool(int(getattr(args, "use_time_elapsed_feature", 0) or 0) == 1),
            include_time_elapsed_in_targets=bool(int(getattr(args, "include_time_elapsed_in_targets", 0) or 0) == 1),
            use_robust_scaler=bool(int(getattr(args, "use_robust_scaler", 0) or 0) == 1),
            rf_multioutput_mode=str(getattr(args, "rf_multioutput_mode", "native")),
            min_rows=int(args.min_rows),
            eval_mode=str(args.eval_mode),
            holdout_k=int(args.holdout_k),
            min_points_per_batch=int(args.min_points_per_batch),
            min_test=int(args.min_test),
            baseline_perfect_eps=float(args.baseline_perfect_eps),
            accept_min_lift_mean=float(args.accept_min_lift_mean),
            tune=bool(args.tune),
            cv_splits=int(args.cv_splits),
            cv_mode=str(args.cv_mode),
            n_jobs=int(args.n_jobs),
        )

        # Save leaderboard
        lb_path = os.path.join(out_dir, f"WSUID_{_safe_token(wsuid)}__ST_{_safe_token(str(args.stNo))}__automl_leaderboard.json")
        with open(lb_path, "w", encoding="utf-8") as f:
            json.dump(leaderboard, f, indent=2, sort_keys=True)

        if not best.get("ok"):
            print(f"AutoML: no accepted candidate. Leaderboard saved: {lb_path}")
            return 4

        pipe = best.pop("pipeline")
        feature_names = best.pop("feature_names")
        sensors_used = best.pop("sensors_used")
        data_info = best.pop("data_info")

        # IMPORTANT compatibility: keep artifact naming scheme stable (still "__MIMO_RF__...__ALG_RF.pkl")
        resample_sec = int(best["resample_sec"])
        horizon_sec = int(best["horizon_sec"])
        n_lags = int(best["n_lags"])
        chosen_model_type = str(best["model_type"])

        # Refit on ALL supervised rows for the winning config (keeps evaluation separate)
        if int(getattr(args, "final_fit_on_all", 1) or 1) == 1:
            X_all, Y_all, _, _, _, _, _ = build_multitarget_dataset(
                df_long,
                targets=targets,
                resample_sec=int(resample_sec),
                horizon_sec=int(horizon_sec),
                n_lags=int(n_lags),
                segment_field=str(segment_field),
                session_gap_sec=int(args.session_gap_sec),
                max_sensors=int(args.max_sensors),
                min_sensor_non_nan=int(args.min_sensor_non_nan),
                pivot_agg=str(getattr(args, 'pivot_agg', 'last')),
                resample_method=str(getattr(args, 'resample_method', 'last')),
                inactive_strategy=str(getattr(args, 'inactive_strategy', 'gap_guard')),
                use_time_elapsed_feature=bool(int(getattr(args, 'use_time_elapsed_feature', 0) or 0) == 1),
                include_time_elapsed_in_targets=bool(int(getattr(args, 'include_time_elapsed_in_targets', 0) or 0) == 1),
                sensors_fixed=sensors_used,
            )
            if int(Y_all.shape[0]) >= int(args.min_rows):
                pipe.fit(X_all, Y_all)

    else:
        # Build dataset for requested config
        resample_sec = int(args.resample_sec)
        horizon_sec = int(args.horizon_sec)
        n_lags = int(args.n_lags)

        X, Y, B0, seg, feature_names, sensors_used, data_info = build_multitarget_dataset(
            df_long,
            targets=targets,
            resample_sec=resample_sec,
            horizon_sec=horizon_sec,
            n_lags=n_lags,
            segment_field=str(segment_field),
            session_gap_sec=int(args.session_gap_sec),
            max_sensors=int(args.max_sensors),
            min_sensor_non_nan=int(args.min_sensor_non_nan),
            pivot_agg=str(getattr(args, 'pivot_agg', 'last')),
            resample_method=str(getattr(args, 'resample_method', 'last')),
            inactive_strategy=str(getattr(args, 'inactive_strategy', 'gap_guard')),
            use_time_elapsed_feature=bool(int(getattr(args, 'use_time_elapsed_feature', 0) or 0) == 1),
            include_time_elapsed_in_targets=bool(int(getattr(args, 'include_time_elapsed_in_targets', 0) or 0) == 1),
        )

        if int(Y.shape[0]) < int(args.min_rows):
            print(f"Insufficient supervised rows: {int(Y.shape[0])} < {int(args.min_rows)}")
            return 3

        pipe0, grid, model_info = make_pipeline(str(args.model_type), random_state=42, n_jobs=int(args.n_jobs), use_robust_scaler=bool(int(getattr(args,'use_robust_scaler',0) or 0)==1), rf_multioutput_mode=str(getattr(args,'rf_multioutput_mode','native')))

        eval_mode = str(args.eval_mode)
        if eval_mode == "batch_holdout":
            ev = eval_batch_holdout(
                X,
                Y,
                B0,
                seg,
                pipe=pipe0,
                grid=grid,
                tune=bool(args.tune),
                cv_splits=int(args.cv_splits),
                cv_mode=str(args.cv_mode),
                n_jobs=int(args.n_jobs),
                holdout_k=int(args.holdout_k),
                min_points_per_batch=int(args.min_points_per_batch),
                min_test=int(args.min_test),
                baseline_perfect_eps=float(args.baseline_perfect_eps),
            )
        elif eval_mode == "time_split":
            ev = eval_time_split(
                X,
                Y,
                B0,
                seg=seg,
                cv_mode=str(args.cv_mode),
                pipe=pipe0,
                grid=grid,
                tune=bool(args.tune),
                cv_splits=int(args.cv_splits),
                                n_jobs=int(args.n_jobs),
                min_test=int(args.min_test),
                baseline_perfect_eps=float(args.baseline_perfect_eps),
            )
        else:
            # auto
            if seg is not None and str(segment_field).upper() != "SESSION":
                ev = eval_batch_holdout(
                    X,
                    Y,
                    B0,
                    seg,
                    pipe=pipe0,
                    grid=grid,
                    tune=bool(args.tune),
                    cv_splits=int(args.cv_splits),
                    n_jobs=int(args.n_jobs),
                    holdout_k=int(args.holdout_k),
                    min_points_per_batch=int(args.min_points_per_batch),
                    min_test=int(args.min_test),
                    baseline_perfect_eps=float(args.baseline_perfect_eps),
                )
                if not ev.get("ok"):
                    ev = eval_time_split(
                        X,
                        Y,
                        B0,
                        seg=seg,
                        cv_mode=str(args.cv_mode),
                        pipe=pipe0,
                        grid=grid,
                        tune=bool(args.tune),
                        cv_splits=int(args.cv_splits),
                        n_jobs=int(args.n_jobs),
                        min_test=int(args.min_test),
                        baseline_perfect_eps=float(args.baseline_perfect_eps),
                    )
            else:
                ev = eval_time_split(
                    X,
                    Y,
                    B0,
                    seg=seg,
                    cv_mode=str(args.cv_mode),
                    pipe=pipe0,
                    grid=grid,
                    tune=bool(args.tune),
                    cv_splits=int(args.cv_splits),
                    n_jobs=int(args.n_jobs),
                    min_test=int(args.min_test),
                    baseline_perfect_eps=float(args.baseline_perfect_eps),
                )

        if not ev.get("ok"):
            print(f"Training skipped: {ev.get('reason')}")
            return 4

        pipe = ev.pop("pipeline")
        chosen_model_type = str(model_info.get("model_type", str(args.model_type))).upper()

        # Refit on ALL supervised rows after evaluation (recommended).
        if int(getattr(args, "final_fit_on_all", 1) or 1) == 1:
            try:
                pipe.fit(X, Y)
            except Exception:
                pass

        best = {
            "eval_mode": ev.get("eval_mode"),
            "baseline_mae_lag0_per_target": ev.get("baseline_mae_lag0_per_target"),
            "model_mae_per_target": ev.get("model_mae_per_target"),
            "lift_vs_lag0_per_target": ev.get("lift_vs_lag0_per_target"),
            "lift_vs_lag0_mean": ev.get("lift_vs_lag0_mean"),
            "accepted": bool(np.isfinite(float(ev.get("lift_vs_lag0_mean", float("nan")))) and float(ev.get("lift_vs_lag0_mean")) >= float(args.accept_min_lift_mean)),
            "tuning": ev.get("tuning", {}),
        }

    # Final artifact paths (compatibility kept)
    wsuid_token = _safe_token(wsuid, default="WS")
    st_tag = _safe_token(str(args.stNo), default="ALL")
    op_tag = _safe_token(str(args.opTc), default="ALL") if str(args.opTc).upper() != "ALL" else "ALL"
    t_hash = _hash_list([_safe_token(t) for t in targets], n=10)

    prefix = (
        f"WSUID_{wsuid_token}_ST_{st_tag}_OPTC_{op_tag}"
        f"__MIMO_RF__HSEC_{int(horizon_sec)}__RSEC_{int(resample_sec)}__NLAG_{int(n_lags)}__TSET_{t_hash}"
    )
    model_path = os.path.join(out_dir, prefix + "__ALG_RF.pkl")
    meta_path = os.path.join(out_dir, prefix + "__meta.json")

    joblib.dump(pipe, model_path)

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ws": {"pl_id": pl_id, "wc_id": wc_id, "ws_id": ws_id, "wsuid": wsuid},
        "stNo": str(args.stNo),
        "opTc": str(args.opTc),
        "segment_field": str(segment_field),
        "segment_policy": segment_policy,
        "resample_sec": int(resample_sec),
        "horizon_sec": int(horizon_sec),
        "n_lags": int(n_lags),
        "tuning_config": {"tune_enabled": bool(args.tune), "cv_splits": int(args.cv_splits), "cv_mode": str(getattr(args, "cv_mode", "tscv"))},
        "targets": [str(t) for t in targets],
        "sensors_used": [str(s) for s in sensors_used],
        "feature_names": [str(x) for x in feature_names],
        "data_info": data_info,
        "model_path": model_path,
        # IMPORTANT: keep downstream stable by storing model_type in meta (filename stays as ALG_RF)
        "model_type": str(chosen_model_type),
        "metrics": {
            "accepted": bool(best.get("accepted", False)),
            "lift_vs_lag0_mean": float(best.get("lift_vs_lag0_mean", float("nan"))),
            "baseline_mae_lag0_mean": _mean_nan(best.get("baseline_mae_lag0_per_target") or []),
            "model_mae_mean": _mean_nan(best.get("model_mae_per_target") or []),
            "eval_mode": best.get("eval_mode", ""),
            "baseline_mae_lag0_per_target": best.get("baseline_mae_lag0_per_target"),
            "model_mae_per_target": best.get("model_mae_per_target"),
            "lift_vs_lag0_per_target": best.get("lift_vs_lag0_per_target"),
            "tuning": best.get("tuning", {}),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("OK")
    print(f"Model: {model_path}")
    print(f"Meta : {meta_path}")
    print(f"ModelType(meta): {chosen_model_type} (artifact suffix kept as ALG_RF for compatibility)")
    print(f"Targets: {len(targets)}  SensorsUsed: {len(sensors_used)}")
    print(f"Lift(mean): {meta['metrics']['lift_vs_lag0_mean']}  Accepted: {meta['metrics']['accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
