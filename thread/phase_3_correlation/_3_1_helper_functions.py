import numpy as np
import pandas as pd
from utils.config_reader import ConfigReader

from datetime import datetime, timezone
from scipy.stats import spearmanr

# Cassandra models are loaded lazily (only when persistence is enabled)
ScadaCorrelationMatrix = None
ScadaCorrelationMatrixSummary = None
_CASSANDRA_CORR_MODELS_OK = False
_CASSANDRA_CORR_MODELS_ERR = None

def _ensure_corr_models(p3_1_log=None):
    global ScadaCorrelationMatrix, ScadaCorrelationMatrixSummary
    global _CASSANDRA_CORR_MODELS_OK, _CASSANDRA_CORR_MODELS_ERR
    if _CASSANDRA_CORR_MODELS_OK:
        return True
    if _CASSANDRA_CORR_MODELS_ERR is not None:
        return False
    try:
        from cassandra_utils.models.scada_correlation_matrix import ScadaCorrelationMatrix as _SCM
        from cassandra_utils.models.scada_correlation_matrix_summary import ScadaCorrelationMatrixSummary as _SCMS
        ScadaCorrelationMatrix = _SCM
        ScadaCorrelationMatrixSummary = _SCMS
        _CASSANDRA_CORR_MODELS_OK = True
        return True
    except Exception as e:  # pragma: no cover
        _CASSANDRA_CORR_MODELS_ERR = e
        if p3_1_log:
            p3_1_log.warning(
                f"[compute_correlation] Cassandra models import failed; persistence disabled: {e}"
            )
        return False

cfg = ConfigReader()

PHASE3_DERIVED_PERSIST_ENABLED = bool(getattr(cfg, "phase3_derived_persist_enabled", True))
PHASE3_CORR_WRITE_LEGACY_ENABLED = bool(getattr(cfg, "phase3_corr_write_legacy_enabled", True))
PHASE3_CORR_WRITE_V2_ENABLED = bool(getattr(cfg, "phase3_corr_write_v2_enabled", False))

# Patch-2: allow enabling v2 writes separately for GLOBAL vs BATCH
PHASE3_CORR_WRITE_V2_GLOBAL_ENABLED = bool(getattr(cfg, 'phase3_corr_write_v2_global_enabled', PHASE3_CORR_WRITE_V2_ENABLED))
PHASE3_CORR_WRITE_V2_BATCH_ENABLED  = bool(getattr(cfg, 'phase3_corr_write_v2_batch_enabled',  PHASE3_CORR_WRITE_V2_ENABLED))

# Patch-2: avoid hot partitions on batch tables (never write prod_order_reference_no='0' unless explicitly allowed)
PHASE3_CORR_V2_REQUIRE_BATCH_ID = bool(getattr(cfg, 'phase3_corr_v2_require_batch_id', True))

try:
    from utils.keypoint_recorder import KP
except Exception:  # pragma: no cover
    KP = None

from utils.identity import get_stock_key

_NULL_ID_TOKENS = {'', '0', 'none', 'null', 'nan', 'n/a', 'na', 'unknown', 'unk'}

def _norm_nonempty_id(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in _NULL_ID_TOKENS:
        return None
    return s

def _kp_inc(name: str, n: int = 1):
    try:
        if KP is not None:
            KP.inc(name, n)
    except Exception:
        pass


def _kp_observe(name: str, v: float):
    try:
        if KP is not None:
            KP.observe(name, float(v))
    except Exception:
        pass


def _kp_set_gauge(name: str, v: float):
    try:
        if KP is not None:
            KP.set_gauge(name, float(v))
    except Exception:
        pass

def _derive_session_batch_id(msg: dict, stock_no: str) -> str:
    # Deterministic-ish fallback to avoid hot partitions when batch id is absent.
    ws_uid = (msg or {}).get('_workstation_uid') or (msg or {}).get('workstation_uid') or (msg or {}).get('wsNo') or (msg or {}).get('wsId') or 'UNK_WS'
    ts_ms = (msg or {}).get('measDt') or (msg or {}).get('crDt')
    try:
        ts_i = int(ts_ms)
    except Exception:
        ts_i = 0
    hour_bucket = 0
    if ts_i > 0:
        if ts_i < 10**11:
            ts_i *= 1000
        hour_bucket = int(ts_i // (1000 * 60 * 60))
    return f"SESSION|{ws_uid}|{stock_no}|{hour_bucket}"

def _choose_corr_batch_id(msg: dict):
    msg = msg or {}

    # 1) Prefer true prod_order_reference_no if present
    prod_ref = _norm_nonempty_id(msg.get('prod_order_reference_no'))
    if prod_ref is None:
        prod_ref = _norm_nonempty_id(msg.get('refNo'))

    # 2) Otherwise prefer computed batch id from M1 batch assigner
    batch_id = _norm_nonempty_id(msg.get('_batch_id'))

    # 3) Decide
    if prod_ref is not None:
        return prod_ref
    if batch_id is not None:
        return batch_id

    # 4) Last resort: deterministic session-like id (keeps partitions non-hot)
    stock_no = str(get_stock_key(msg, default='ALL') or 'ALL')
    return _derive_session_batch_id(msg, stock_no)


# --------------------------- utilities ---------------------------

def map_to_text(obj):
    if obj is None or not isinstance(obj, dict):
        return {}
    return {k: str(v) if v is not None else '' for k, v in obj.items()}


def _to_epoch_ms(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    # numeric?
    try:
        iv = int(v)
        # if it's clearly in seconds, convert to ms
        return iv * 1000 if iv < 10**11 else iv
    except Exception:
        pass

    # ISO string?
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _coerce_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _unique_names(names):
    """
    Make sensor names unique (Sensor, Sensor_2, Sensor_3, ...)
    if needed. Keep for future use if you want to deduplicate.
    """
    seen = {}
    out = []
    for n in names:
        base = (n or "Sensor").strip() or "Sensor"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def _corr_sensor_key(d: dict):
    """Return a stable, unique-ish sensor key for correlation.

    IMPORTANT:
      - Correlation storage must not collapse distinct sensors into the same label.
      - Prefer a composite key when possible: "<parameter>-<equipment_name>".

    Inputs seen in this project:
      - parameter / eqNo: numeric sensor id
      - equipment_name / eqNm: human-readable sensor name

    Strategy:
      1) if both parameter and equipment_name exist -> "param-eqNm"
      2) else if parameter exists -> "param"
      3) else if equipment_name exists -> "eqNm"
      4) else -> None

    Note: We do not do any lossy renaming for persistence. Display renaming belongs
    in the visualization layer, not the stored matrix.
    """
    if not isinstance(d, dict):
        return None

    param = d.get('parameter') or d.get('param') or d.get('eqNo')
    eqnm = d.get('equipment_name') or d.get('eqNm')

    param_s = str(param).strip() if param is not None else ''
    eqnm_s = str(eqnm).strip() if eqnm is not None else ''

    if param_s and eqnm_s:
        return f"{param_s}-{eqnm_s}"
    if param_s:
        return param_s
    if eqnm_s:
        return eqnm_s
    return None


# Backwards-compatible alias used across DF builders in this module
_sensor_name = _corr_sensor_key

def _sensor_value(d: dict):
    """
    Extract numeric reading from sensor dict.
    Supports keys: 'counter_reading', 'cntRead', 'value'.
    """
    v = d.get('counter_reading', d.get('cntRead', d.get('value', None)))
    if v is None:
        return None
    try:
        return float(str(v).replace(',', '.'))
    except Exception:
        return None


def _to_equipment_label(var_name: str) -> str:
    """
    'parameter-equipment_name' -> 'equipment_name'
    'parameter' (no '-')      -> 'parameter' (fallback)
    """
    if not isinstance(var_name, str):
        return str(var_name)
    if "-" in var_name:
        # ilk '-' sonrası ekipman adı
        return var_name.split("-", 1)[1]
    return var_name


def _extract_output_stock_from_message(message: dict):
    """
    Extract (output_stock_no, output_stock_name) from message.
    - Prefer prodList[*].stNo / stNm
    - Fallback to direct fields: output_stock_no / output_stock_name
    """
    if not isinstance(message, dict):
        return None, None

    # 1) prodList içinden
    prod_list = message.get("prodList") or []
    if isinstance(prod_list, list):
        for p in prod_list:
            if not isinstance(p, dict):
                continue
            st_no = p.get("stNo") or p.get("stockNo") or p.get("st_no")
            if st_no not in (None, ""):
                st_nm = p.get("stNm") or p.get("stockName") or p.get("st_name")
                return str(st_no), (str(st_nm) if st_nm is not None else None)

    # 2) direkt alanlardan
    st_no = (
        message.get("output_stock_no")
        or message.get("stNo")
        or message.get("stock_no")
    )
    st_nm = (
        message.get("output_stock_name")
        or message.get("stNm")
        or message.get("stock_name")
    )
    if st_no not in (None, ""):
        return str(st_no), (str(st_nm) if st_nm is not None else None)

    return None, None

def _is_input_type(v) -> bool:
    """
    Normalize equipment_type from many possible representations.
    True means INPUT.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) == 1
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "input", "in", "i")


def _is_output_sensor(sensor_dict: dict) -> bool:
    """
    OUTPUT = not INPUT
    """
    return not _is_input_type(sensor_dict.get("equipment_type"))


# --------------------------- DF builders ---------------------------

def extract_cntReads_to_df(sensor_values):
    """
    Build a tidy DataFrame:
      columns = ['crDt'] + sorted(unique sensor names across all timestamps)
      rows    = one per timestamp; values = float or NaN

    Sensor key = "<parameter>-<equipment_name>" when both exist, otherwise falls back to the available identifier.
    """
    if not sensor_values or not isinstance(sensor_values, list):
        return pd.DataFrame()

    # 1) collect all distinct sensor names
    name_set = set()
    for grp in sensor_values:
        if not isinstance(grp, (list, tuple)) or len(grp) < 2:
            continue
        for sensor in grp[1:]:
            d = dict(sensor) if sensor is not None else {}
            if not _is_output_sensor(d):  # ← INPUT'ları skip et
                continue
            nm = _sensor_name(d)
            if nm:
                name_set.add(nm)

    sensor_names = sorted(name_set)
    if not sensor_names:
        return pd.DataFrame()

    columns = ['crDt'] + sensor_names
    rows = []

    # 2) build rows
    for grp in sensor_values:
        if not isinstance(grp, (list, tuple)) or len(grp) < 2:
            continue

        meta = dict(grp[0]) if grp[0] is not None else {}
        crdt_ms = (
            _to_epoch_ms(meta.get('crDt'))
            or _to_epoch_ms(meta.get('measurement_date'))
            or _to_epoch_ms(meta.get('mdate'))
            or 0
        )

        row = {'crDt': crdt_ms, **{nm: np.nan for nm in sensor_names}}

        # last value wins; (you can switch to averaging if you want)
        for sensor in grp[1:]:
            d = dict(sensor) if sensor is not None else {}
            if not _is_output_sensor(d):  # ← INPUT'ları skip et
                continue
            nm = _sensor_name(d)
            if nm and (nm in row):
                val = _sensor_value(d)
                if val is not None:
                    row[nm] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # enforce column order
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[columns]


def extract_cntReads_to_df_with_message(sensor_values, message) -> pd.DataFrame:
    """
    1) History (sensor_values) -> df_hist
    2) Mesaj (outVals)        -> df_msg
    3) Kolonları birleştirip concat et
    """
    df_hist = extract_cntReads_to_df(sensor_values)

    row_msg = _row_from_message_out_for_corr(message)
    df_msg = pd.DataFrame([row_msg]) if row_msg else pd.DataFrame()

    if df_hist.empty and df_msg.empty:
        return pd.DataFrame()

    if df_hist.empty:
        return df_msg

    if df_msg.empty:
        return df_hist

    # kolonları hizala
    all_cols = sorted(set(df_hist.columns) | set(df_msg.columns))
    df_hist = df_hist.reindex(columns=all_cols)
    df_msg  = df_msg.reindex(columns=all_cols)

    df = pd.concat([df_hist, df_msg], ignore_index=True)

    if "crDt" in df.columns:
        df = df.sort_values("crDt").reset_index(drop=True)

    return df


def extract_cntReads_to_df_with_stock(sensor_values, message) -> pd.DataFrame:
    """
    Same as extract_cntReads_to_df, but also attaches:
      - output_stock_no
      - output_stock_name
    as metadata columns for each row.

    Used for WS-scope, per-stock correlation.
    """
    if not sensor_values or not isinstance(sensor_values, list):
        return pd.DataFrame()

    # 1) collect sensor names
    name_set = set()
    for grp in sensor_values:
        if not isinstance(grp, (list, tuple)) or len(grp) < 2:
            continue
        for sensor in grp[1:]:
            d = dict(sensor) if sensor is not None else {}
            if not _is_output_sensor(d):  # ← INPUT'ları skip et
                continue
            nm = _sensor_name(d)
            if nm:
                name_set.add(nm)

    sensor_names = sorted(name_set)
    if not sensor_names:
        return pd.DataFrame()

    columns = ['crDt', 'output_stock_no', 'output_stock_name'] + sensor_names
    rows = []

    # 2) build rows with stock info
    msg_stock_no, msg_stock_name = _extract_output_stock_from_message(message or {})

    for grp in sensor_values:
        if not isinstance(grp, (list, tuple)) or len(grp) < 2:
            continue

        meta = dict(grp[0]) if grp[0] is not None else {}

        crdt_ms = (
            _to_epoch_ms(meta.get('crDt'))
            or _to_epoch_ms(meta.get('measurement_date'))
            or _to_epoch_ms(meta.get('mdate'))
            or 0
        )

        # prefer stock from meta if present, else fall back to message
        st_no = (
            meta.get("output_stock_no")
            or meta.get("stock_no")
            or meta.get("stNo")
            or msg_stock_no
        )
        st_nm = (
            meta.get("output_stock_name")
            or meta.get("stNm")
            or msg_stock_name
        )

        row = {
            'crDt': crdt_ms,
            'output_stock_no': str(st_no) if st_no not in (None, "") else None,
            'output_stock_name': str(st_nm) if st_nm not in (None, "") else None,
        }
        row.update({nm: np.nan for nm in sensor_names})

        for sensor in grp[1:]:
            d = dict(sensor) if sensor is not None else {}
            if not _is_output_sensor(d):  # ← INPUT'ları skip et
                continue
            nm = _sensor_name(d)
            if nm and (nm in row):
                val = _sensor_value(d)
                if val is not None:
                    row[nm] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    if "crDt" in df.columns:
        df = df.sort_values("crDt").reset_index(drop=True)

    return df[columns]


def _to_dt_safe(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        iv = int(v)
        if iv > 10**12:
            return datetime.fromtimestamp(iv / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(iv, tz=timezone.utc)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _bundle_from_message(message):
    out_vals = message.get("outVals") or []
    if not out_vals:
        return None

    meas_ms = out_vals[0].get("measDt") or message.get("crDt")
    dt = _to_dt_safe(meas_ms) or datetime.now(timezone.utc)

    meta = {
        "measurement_date": dt,
        "crDt": str(int(dt.timestamp() * 1000)),
        "good": message.get("goodCnt") if "goodCnt" in message else message.get("good"),
        "prSt": message.get("prSt"),
        "job_order_reference_no": message.get("joRef") or message.get("job_order_reference_no"),
        "prod_order_reference_no": message.get("refNo") or message.get("prod_order_reference_no"),
        "output_stock_no": message.get("output_stock_no"),
        "output_stock_name": message.get("output_stock_name"),
    }

    sensors = []
    for ov in out_vals:
        sensors.append({
            "parameter": str(ov.get("eqNo")),
            "counter_reading": str(ov.get("cntRead")) if ov.get("cntRead") is not None else "0",
            "equipment_name": str(ov.get("eqNm")),
        })

    return [meta] + sensors


def _extract_stock_from_bundle(bundle):
    if not bundle or not isinstance(bundle, list) or not isinstance(bundle[0], dict):
        return None
    st = bundle[0].get("output_stock_no")
    return None if st in (None, "", "None") else str(st)


# ------------------- correlation matrix helpers -------------------


def _sanitize_axis_names(names):
    """Deterministically sanitize axis names without changing matrix dimension.

    Requirements:
      - Never drop variables (matrix size must be stable)
      - Produce non-empty, printable strings
      - Enforce uniqueness (append stable suffixes)

    Returns:
      - sanitized_names: list[str]
      - mapping: dict[old->new] for changed names only
      - had_collisions: bool
    """
    invalid = {"", "none", "null", "nan", "n/a", "na", "unknown", "unk"}
    seen = {}
    out = []
    mapping = {}
    had_collisions = False

    for i, raw in enumerate(list(names)):
        if raw is None:
            base = ""
        else:
            base = str(raw).replace("\n", " ").replace("\r", " ").strip()

        if base.lower() in invalid:
            base = f"__invalid__{i}"

        # enforce uniqueness deterministically
        k = base
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > 1:
            had_collisions = True
            base = f"{base}__{seen[k]}"

        out.append(base)

        if (raw is None and base != "") or (raw is not None and str(raw) != base):
            mapping[str(raw) if raw is not None else "None"] = base

    return out, mapping, had_collisions


def _sanitize_corr_df(df_corr: pd.DataFrame) -> pd.DataFrame:
    """Sanitize correlation matrix without changing its size.

    Ensures:
      - numeric dtype
      - symmetric
      - diagonal=1.0
      - no NaN/Inf (converted to 0.0)
      - axis labels are valid strings and unique (no drops)
    """
    if df_corr is None or df_corr.empty:
        return df_corr

    # Ensure square and aligned axes (best effort)
    try:
        # If columns differ from index, align to intersection in order (avoid silent reordering)
        if list(df_corr.columns) != list(df_corr.index):
            common = [c for c in df_corr.index if c in set(df_corr.columns)]
            if common:
                df_corr = df_corr.loc[common, common]
    except Exception:
        pass

    # Sanitize axis names deterministically (NO DROPS)
    names = list(df_corr.index)
    sani, _, _ = _sanitize_axis_names(names)
    if sani != names:
        df_corr = df_corr.copy()
        df_corr.index = sani
        df_corr.columns = sani

    # Coerce to float matrix
    df_corr = df_corr.astype(float, copy=False)

    # Replace inf/-inf -> NaN, then fill with 0
    df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_corr.fillna(0.0, inplace=True)

    # Force symmetry
    df_corr = (df_corr + df_corr.T) / 2.0

    # Force diagonal to 1.0
    np.fill_diagonal(df_corr.values, 1.0)

    # Final pass
    df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_corr.fillna(0.0, inplace=True)

    return df_corr

def _sanitize_corr_df2(df_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure: numeric dtype, diagonal=1.0, no NaN/Inf (converted to 0.0).
    """
    # Coerce to float matrix
    df_corr = df_corr.astype(float, copy=False)

    # Replace inf/-inf -> NaN, then fill with 0
    df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_corr.fillna(0.0, inplace=True)

    # Force symmetry just in case (average with its transpose)
    df_corr = (df_corr + df_corr.T) / 2.0

    # Force diagonal to 1.0
    np.fill_diagonal(df_corr.values, 1.0)

    # Final pass to guarantee no NaNs remain (paranoia)
    df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_corr.fillna(0.0, inplace=True)

    return df_corr


def _matrix_to_frozen(df_numeric: pd.DataFrame):
    """
    Convert DataFrame to frozen list-of-maps format.
    FIXED: Handle duplicate index names.
    """
    frozen = []
    
    # Ensure unique index (deduplicate if needed)
    if df_numeric.index.duplicated().any():
        # Keep first occurrence
        df_numeric = df_numeric[~df_numeric.index.duplicated(keep='first')]
    
    for row_var in df_numeric.index:
        row = {}
        # Use .loc with single index value
        row_values = df_numeric.loc[row_var]
        
        # Handle both Series and scalar
        if isinstance(row_values, pd.Series):
            for col_var, val in row_values.items():
                # Sanitize per-value
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    row[str(col_var)] = 0.0
                else:
                    row[str(col_var)] = float(val)
        else:
            # Scalar (shouldn't happen but handle it)
            row[str(df_numeric.columns[0])] = float(row_values) if pd.notna(row_values) else 0.0
        
        frozen.append({str(row_var): row})
    
    return frozen

def _matrix_to_frozen2(df_numeric: pd.DataFrame):
    """
    Convert DataFrame to the frozen list-of-maps format expected by
    ScadaCorrelationMatrix.correlation_data.

    Result shape:
      [
        {"var1": {"var1": 1.0, "var2": 0.3, ...}},
        {"var2": {"var1": 0.3, "var2": 1.0, ...}},
        ...
      ]
    """
    frozen = []
    for row_var in df_numeric.index:
        row = {}
        for col_var, val in df_numeric.loc[row_var].items():
            # sanitize per-value
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                row[col_var] = 0.0
            else:
                row[col_var] = float(val)
        frozen.append({row_var: row})
    return frozen


def convert_corr_matrix_to_frozen_structure(corr_df: pd.DataFrame):
    # sanitize before freezing
    corr_df = _sanitize_corr_df(corr_df)
    return _matrix_to_frozen(corr_df)


def _row_from_message_out_for_corr(message: dict) -> dict:
    """
    outVals'tan correlation için tek satır üretir.
    Kolon adları extract_cntReads_to_df ile uyumlu olsun diye
    _sensor_name / _sensor_value kullanıyoruz.
    """
    row = {}

    out_list = message.get("outVals") or []
    if not isinstance(out_list, list):
        return row

    for ov in out_list:
        if not isinstance(ov, dict):
            continue

        fake_sensor = {
            "parameter": ov.get("eqNo") or ov.get("param") or ov.get("eqNm"),
            "equipment_name": ov.get("eqNm"),
            "counter_reading": ov.get("cntRead"),
            "equipment_type": False,
        }
        nm = _sensor_name(fake_sensor)
        if not nm:
            continue

        val = _sensor_value(fake_sensor)
        if val is not None:
            row[nm] = val

    # zaman
    ts = (
        message.get("crDt")
        or (out_list[0].get("measDt") if out_list else None)
    )
    row["crDt"] = _to_epoch_ms(ts)

    return row


# ---------------- internal core: compute & save for one DF  ----------------

def _compute_and_save_from_df(
    df: pd.DataFrame,
    message: dict,
    p3_1_log,
    algorithm: str,
    scope: str,
    scope_id=None
):
    """
    Core correlation: uses df, skips metadata columns, saves to Cassandra.
    """
    if df is None or df.empty:
        if p3_1_log:
            p3_1_log.warning(
                f"[compute_correlation] empty DataFrame; skipping "
                f"(scope={scope}, scope_id={scope_id})"
            )
        return

    if p3_1_log:
        p3_1_log.info(f"[compute_correlation] DF columns: {df.columns.tolist()}")

    # Metadata columns that should NOT be used as variables
    meta_cols = {'crDt', 'output_stock_no', 'output_stock_name'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if len(feature_cols) < 2:
        if p3_1_log:
            p3_1_log.warning(
                f"[compute_correlation] not enough distinct sensors after "
                f"dropping metadata; skipping (scope={scope}, scope_id={scope_id})"
            )
        return

    # Order columns: metadata first (if present), then features
    ordered_cols = []
    for m in ('crDt', 'output_stock_no', 'output_stock_name'):
        if m in df.columns:
            ordered_cols.append(m)
    ordered_cols.extend(feature_cols)
    df = df[ordered_cols]

    # numeric matrix
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    p3_1_log.info(
        f"[compute_correlation] Computing correlation matrix for "
        f"scope={scope}, scope_id={scope_id} on {X.shape[0]} rows and "
        f"{X.shape[1]} variables"
    )

    cols = X.columns
    n = len(cols)
    corr = pd.DataFrame(0.0, index=cols, columns=cols)

    if algorithm.upper() != "SPEARMAN":
        if p3_1_log:
            p3_1_log.warning(
                f"[compute_correlation] Unsupported algorithm={algorithm}, "
                f"falling back to SPEARMAN"
            )

    
    # Minimum overlapping points required per pair (default: 3)
    min_overlap = int(getattr(cfg, "phase3_corr_min_overlap", 3) or 3)

    if p3_1_log:
        p3_1_log.info("[compute_correlation] Computing Spearman correlations")

    zero_corr_count = 0
    low_overlap_pairs = []

    overlap_counts = []  # off-diagonal overlaps (nij)
    constant_pair_count = 0

# --- compute only upper triangle, mirror to keep symmetry ---
    for i in range(n):
        xi = X.iloc[:, i].to_numpy(dtype=float)
        for j in range(i, n):
            yj = X.iloc[:, j].to_numpy(dtype=float)

            mask = ~np.isnan(xi) & ~np.isnan(yj)
            nij = int(mask.sum())

            if i != j:
                overlap_counts.append(nij)

            if i == j:
                # self-correlation = 1 by definition (even if constant)
                r = 1.0
            elif nij >= min_overlap and np.nanstd(xi[mask]) > 0 and np.nanstd(yj[mask]) > 0:
                r, _ = spearmanr(xi[mask], yj[mask], nan_policy='omit')
                # clamp/clean
                if r is None or np.isnan(r) or np.isinf(r):
                    r = 0.0
                else:
                    r = float(np.clip(r, -1.0, 1.0))
            
            else:
                r = 0.0
                zero_corr_count += 1
                if nij < min_overlap:
                    low_overlap_pairs.append((cols[i], cols[j], nij))
                else:
                    # overlap is sufficient but one/both series are constant (std==0)
                    constant_pair_count += 1

            corr.iat[i, j] = r
            corr.iat[j, i] = r  # mirror

    # --- overlap diagnostics (do not change output semantics) ---
    if overlap_counts:
        oc = np.array(overlap_counts, dtype=float)
        oc_min = float(np.min(oc))
        oc_p10 = float(np.quantile(oc, 0.10))
        oc_p50 = float(np.quantile(oc, 0.50))
        oc_p90 = float(np.quantile(oc, 0.90))
        oc_max = float(np.max(oc))
        low_frac = float(len(low_overlap_pairs)) / float(max(1, len(overlap_counts)))
        const_frac = float(constant_pair_count) / float(max(1, len(overlap_counts)))

        _kp_observe('phase3.corr.overlap_min', oc_min)
        _kp_observe('phase3.corr.overlap_p10', oc_p10)
        _kp_observe('phase3.corr.overlap_p50', oc_p50)
        _kp_observe('phase3.corr.overlap_p90', oc_p90)
        _kp_observe('phase3.corr.overlap_max', oc_max)
        _kp_observe('phase3.corr.low_overlap_frac', low_frac)
        _kp_observe('phase3.corr.constant_pair_frac', const_frac)

        if p3_1_log:
            p3_1_log.info(
                f"[compute_correlation] overlap stats (min/p10/p50/p90/max)="
                f"{oc_min:.0f}/{oc_p10:.0f}/{oc_p50:.0f}/{oc_p90:.0f}/{oc_max:.0f} "
                f"min_overlap={min_overlap} low_overlap_frac={low_frac:.3f} constant_pair_frac={const_frac:.3f}"
            )

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Raw correlation matrix for scope={scope}, "
            f"scope_id={scope_id}:\n{corr}"
        )

    if p3_1_log and zero_corr_count > 0:
        p3_1_log.info(
            f"[compute_correlation] {zero_corr_count} pairs had zero correlation "
            f"(low overlap or constant values)"
        )
        if low_overlap_pairs and len(low_overlap_pairs) <= 10:
            p3_1_log.debug(
                f"[compute_correlation] Low overlap examples: {low_overlap_pairs[:10]}"
            )

    # --- sanitize matrix (diag=1.0, no NaN/Inf) ---
    corr = _sanitize_corr_df(corr)

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Sanitized correlation matrix for scope={scope}, "
            f"scope_id={scope_id}:\n{corr}"
        )


# ------------------ PERSISTENCE LABELS ------------------
# IMPORTANT: Persist using the original correlation variable keys (no lossy renaming).
# Any display-only renaming (e.g., equipment_name) must happen downstream in the visualization layer.

# Build representations for saving (using stable, unique keys)

    # 1) frozen list-of-maps for ScadaCorrelationMatrix (PID-based)
    frozen_corr = convert_corr_matrix_to_frozen_structure(corr)

    # 2) dict-of-dicts for ScadaCorrelationMatrixSummary (WS-based)
    corr_dict = {
        row_var: {
            col_var: float(val) if not (val is None or np.isnan(val) or np.isinf(val)) else 0.0
            for col_var, val in corr.loc[row_var].items()
        }
        for row_var in corr.index
    }

    # --- persist according to scope ---
    if not PHASE3_DERIVED_PERSIST_ENABLED:
        if p3_1_log:
            p3_1_log.info(
                f"[compute_correlation] persist disabled; skipping Cassandra write (scope={scope}, scope_id={scope_id})"
            )
        return corr

    # Lazy-import Cassandra persistence models only when needed
    if not _ensure_corr_models(p3_1_log=p3_1_log):
        return corr

    try:
        # Work on a copy to avoid mutating original message everywhere
        save_msg = dict(message or {})

        if scope == "pid":
            # Patch-2: ensure batch correlation writes do not collapse into hot partition (prod_order_reference_no="0")
            corr_batch_id = _choose_corr_batch_id(save_msg)
            if corr_batch_id:
                save_msg['prod_order_reference_no'] = str(corr_batch_id)
            elif PHASE3_CORR_V2_REQUIRE_BATCH_ID:
                _kp_inc('phase3.corr.skip.v2_batch.no_batch_id', 1)

            # PID scope correlates over the current production window → treat as BATCH.
            if PHASE3_CORR_WRITE_LEGACY_ENABLED:
                if p3_1_log:
                    p3_1_log.info(
                        f"[compute_correlation] (legacy) Saving PID-level correlation "
                        f"(pid={scope_id}) via ScadaCorrelationMatrix"
                    )
                if ScadaCorrelationMatrix is None:
                    if p3_1_log:
                        p3_1_log.warning("[compute_correlation] Cassandra models unavailable; skipping ScadaCorrelationMatrix write")
                    _kp_inc('phase3.corr.skip.cass_models', 1)
                else:
                    ScadaCorrelationMatrix.saveData(save_msg, frozen_corr, p3_1_log=p3_1_log)
                _kp_inc('phase3.corr.write.legacy_batch.ok', 1)

            if PHASE3_CORR_WRITE_V2_BATCH_ENABLED:
                if PHASE3_CORR_V2_REQUIRE_BATCH_ID and not _norm_nonempty_id(save_msg.get('prod_order_reference_no')):
                    if p3_1_log:
                        p3_1_log.warning('[compute_correlation] (v2) Skipping BATCH correlation write: missing non-empty batch id')
                    _kp_inc('phase3.corr.skip.v2_batch.no_batch_id', 1)
                else:
                    from cassandra_utils.models.scada_correlation_matrix_ws_stock_batch import (
                        ScadaCorrelationMatrixWsStockBatch,
                    )

                    if p3_1_log:
                        p3_1_log.info(
                            f"[compute_correlation] (v2) Saving BATCH correlation "
                            f"(pid={scope_id}) via ScadaCorrelationMatrixWsStockBatch"
                        )
                    ScadaCorrelationMatrixWsStockBatch.saveData(
                        save_msg,
                        frozen_corr,
                        algorithm=algorithm,
                        p3_1_log=p3_1_log,
                    )
                    _kp_inc('phase3.corr.write.v2_batch.ok', 1)

        elif scope == "ws":
            # WS scope correlates over WS+STOCK window → treat as GLOBAL.
            # Ensure each run creates a fresh row.
            from datetime import datetime, timezone
            save_msg["partition_date"] = datetime.now(timezone.utc)

            if PHASE3_CORR_WRITE_LEGACY_ENABLED:
                if p3_1_log:
                    p3_1_log.info(
                        f"[compute_correlation] (legacy) Saving WS-level correlation "
                        f"(wsId={scope_id}) via ScadaCorrelationMatrixSummary"
                    )
                if ScadaCorrelationMatrixSummary is None:
                    if p3_1_log:
                        p3_1_log.warning("[compute_correlation] Cassandra models unavailable; skipping ScadaCorrelationMatrixSummary write")
                    _kp_inc('phase3.corr.skip.cass_models', 1)
                else:
                    ScadaCorrelationMatrixSummary.saveData(save_msg, corr_dict, p3_1_log=p3_1_log)
                _kp_inc('phase3.corr.write.legacy_global.ok', 1)

            if PHASE3_CORR_WRITE_V2_GLOBAL_ENABLED:
                from cassandra_utils.models.scada_correlation_matrix_ws_stock_global import (
                    ScadaCorrelationMatrixWsStockGlobal,
                )

                if p3_1_log:
                    p3_1_log.info(
                        f"[compute_correlation] (v2) Saving GLOBAL correlation "
                        f"(wsId={scope_id}) via ScadaCorrelationMatrixWsStockGlobal"
                    )
                ScadaCorrelationMatrixWsStockGlobal.saveData(
                    save_msg,
                    frozen_corr,
                    algorithm=algorithm,
                    p3_1_log=p3_1_log,
                )
                _kp_inc('phase3.corr.write.v2_global.ok', 1)

        else:
            if p3_1_log:
                p3_1_log.warning(f"[compute_correlation] Unknown scope={scope}; nothing persisted.")

    except Exception as e:
        if p3_1_log:
            p3_1_log.error(
                f"[compute_correlation] Error saving correlation matrix "
                f"(scope={scope}, scope_id={scope_id}): {e}",
                exc_info=True
            )
        raise

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Saved correlation matrix to Cassandra "
            f"(scope={scope}, scope_id={scope_id})"
        )


# --------------------------- main API ---------------------------
def compute_correlation(
    sensor_values,
    message,
    p3_1_log=None,
    algorithm: str = "SPEARMAN",
    scope: str = "pid",
    scope_id=None,
    group_by_output_stock: bool = False,
):
    """
    Compute correlation matrix for the given sensor_values + CURRENT message and save it to Cassandra.

    - scope="pid": per-process (joOpId) → ScadaCorrelationMatrix
    - scope="ws":  per-workstation
        - group_by_output_stock=False → global WS correlation
        - group_by_output_stock=True  → ONLY current message's stock correlation (no jumping)
    """

    # ---------------- small local debug helpers ----------------
    def _dbg_df_stats(df, tag: str):
        if not p3_1_log or df is None or df.empty:
            return

        meta_cols = {'crDt', 'output_stock_no', 'output_stock_name'}
        feature_cols = [c for c in df.columns if c not in meta_cols]

        p3_1_log.info(f"[compute_correlation][dbg] {tag}: df shape={df.shape}")
        p3_1_log.info(f"[compute_correlation][dbg] {tag}: cols={df.columns.tolist()}")
        p3_1_log.info(
            f"[compute_correlation][dbg] {tag}: feature_cols={len(feature_cols)} "
            f"meta_present={[c for c in df.columns if c in meta_cols]}"
        )

        if not feature_cols:
            return

        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

        nn = X.notna().sum().sort_values()
        p3_1_log.debug(f"[compute_correlation][dbg] {tag}: non-null counts min..max={int(nn.min())}..{int(nn.max())}")
        #p3_1_log.debug(f"[compute_correlation][dbg] {tag}: non-null counts head:\n{nn.head(10)}")
        #p3_1_log.debug(f"[compute_correlation][dbg] {tag}: non-null counts tail:\n{nn.tail(10)}")

        stds = X.std(skipna=True).sort_values()
        p3_1_log.debug(f"[compute_correlation][dbg] {tag}: std min..max={float(stds.min())}..{float(stds.max())}")
        #p3_1_log.debug(f"[compute_correlation][dbg] {tag}: std head:\n{stds.head(10)}")

        # pairwise overlap quick check
        cols = X.columns.tolist()
        overlaps = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                nij = ((~X.iloc[:, i].isna()) & (~X.iloc[:, j].isna())).sum()
                overlaps.append(int(nij))
        if overlaps:
            p3_1_log.debug(f"[compute_correlation][dbg] {tag}: pairwise overlap off-diag min={min(overlaps)}, max={max(overlaps)}")
        else:
            p3_1_log.debug(f"[compute_correlation][dbg] {tag}: pairwise overlap off-diag min=0, max=0")

def _prepare_df_for_corr(df, tag: str, scope: str, p3_1_log=None):
    """Prepare a wide DF for correlation.

    Goals:
      - Keep metadata columns untouched
      - Improve pairwise overlap (optional time-binning)
      - Keep matrix dimension stable (do not drop constant sensors by default)

    Notes:
      - For WS scope we allow a minimal forward-fill (limit=1) to align close samples.
      - Time-binning is controlled by config key `phase3_corr_resample_sec`.
    """
    if df is None or df.empty:
        return df

    # always sort by time if present
    if "crDt" in df.columns:
        df = df.sort_values("crDt").reset_index(drop=True)

    meta_cols = {'crDt', 'output_stock_no', 'output_stock_name'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if len(feature_cols) < 2:
        if p3_1_log:
            p3_1_log.warning(f"[compute_correlation][dbg] {tag}: not enough sensors (<2) after meta drop.")
        return df

    # numeric features
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # --- optional event-time binning to improve pairwise overlap ---
    corr_resample_sec = int(getattr(cfg, "phase3_corr_resample_sec", 0) or 0)
    if corr_resample_sec > 0 and "crDt" in df.columns:
        try:
            tms = pd.to_numeric(df["crDt"], errors="coerce")
            bucket = (tms // float(corr_resample_sec * 1000)).astype("Int64")
            df2 = df.copy()
            df2["__corr_bucket"] = bucket
            df2 = df2.dropna(subset=["__corr_bucket"])

            if not df2.empty:
                before_n = int(len(df2))
                # groupby.last() returns last non-null per column
                g = df2.groupby("__corr_bucket", sort=True, dropna=True)
                df2 = g.last().reset_index()
                # represent time as bucket start (ms)
                df2["crDt"] = (df2["__corr_bucket"].astype("int64") * int(corr_resample_sec) * 1000)
                df2 = df2.drop(columns=["__corr_bucket"])

                after_n = int(len(df2))
                _kp_observe('phase3.corr.rows_before_resample', before_n)
                _kp_observe('phase3.corr.rows_after_resample', after_n)
                _kp_set_gauge('phase3.corr.resample_sec', float(corr_resample_sec))

                if p3_1_log:
                    p3_1_log.info(
                        f"[compute_correlation][dbg] {tag}: time-binned rows {before_n} -> {after_n} "
                        f"(phase3_corr_resample_sec={corr_resample_sec})"
                    )

                df = df2
                feature_cols = [c for c in df.columns if c not in meta_cols]
                X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            if p3_1_log:
                p3_1_log.warning(f"[compute_correlation][dbg] {tag}: time-binning failed: {e}")

    # --- minimal time alignment ---
    if scope == "ws":
        X = X.ffill(limit=1)

    # drop all-null sensors
    non_all_null = X.columns[X.notna().any()].tolist()
    X = X[non_all_null]

    # NOTE: Do NOT drop constant sensors by default; keep dimension stable.
    # If you later decide to drop, ensure downstream expects reduced dimension.

    # rebuild df preserving metadata columns EXACTLY as-is
    out = df[[c for c in df.columns if c in meta_cols]].copy()
    out = pd.concat([out.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return out


    # ---------------- tiny helpers for stock gating ----------------
    def _clean_stock(v):
        if v is None:
            return None
        sv = str(v)
        if sv in ("", "None", "nan", "NaN"):
            return None
        return sv

    # ------------------ start function logging ------------------
    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] START (scope={scope}, scope_id={scope_id}, algorithm={algorithm}, "
            f"group_by_output_stock={group_by_output_stock})"
        )

    # IMPORTANT: do NOT touch/normalize stock fields; just ensure keys exist if extract function returns them
    if isinstance(message, dict):
        st_no, st_nm = _extract_output_stock_from_message(message)
        if st_no is not None:
            message.setdefault("output_stock_no", st_no)
        if st_nm is not None:
            message.setdefault("output_stock_name", st_nm)

    # ------------------ WS current-stock-only path ------------------
    if scope == "ws" and group_by_output_stock:
        if p3_1_log:
            p3_1_log.info("[compute_correlation] PATH: WS current-stock ONLY (group_by_output_stock=True)")

        current_stock = _clean_stock((message or {}).get("output_stock_no"))
        if current_stock is None:
            if p3_1_log:
                p3_1_log.warning(
                    "[compute_correlation] group_by_output_stock=True but current message has no valid output_stock_no; "
                    "SKIPPING to avoid mixing stocks."
                )
            return

        # Build DF that includes history + current message (your existing helper does this)
        df_stock = extract_cntReads_to_df_with_stock(sensor_values, message)
        _dbg_df_stats(df_stock, "WS_STOCK_DF(full)")

        if df_stock is None or df_stock.empty or "output_stock_no" not in df_stock.columns:
            if p3_1_log:
                p3_1_log.warning(
                    "[compute_correlation] Could not build per-stock DF; SKIPPING (do not fallback to global) "
                    "because group_by_output_stock=True."
                )
            return

        # Filter to ONLY current stock (no jumping to A/C when current is B)
        df_one = df_stock[df_stock["output_stock_no"].astype(str) == str(current_stock)].copy()
        _dbg_df_stats(df_one, f"WS_STOCK_DF(current_stock={current_stock})[filtered]")

        # If <2 time points -> wait for next message (do not compute)
        # Use crDt if present; otherwise row count.
        if df_one.empty:
            n_points = 0
        elif "crDt" in df_one.columns:
            n_points = df_one["crDt"].dropna().nunique()
        else:
            n_points = len(df_one)

        if n_points < 2:
            if p3_1_log:
                p3_1_log.info(
                    f"[compute_correlation] WS current-stock gating: stock={current_stock} points={n_points} (<2). "
                    "Waiting for next message; skipping correlation."
                )
            return

        # apply fixes on this current-stock DF BEFORE compute
        _dbg_df_stats(df_one, f"WS_STOCK_DF(stock={current_stock})[before_fix]")
        df_one_fixed = _prepare_df_for_corr(df_one, f"WS_STOCK_DF(stock={current_stock})")
        _dbg_df_stats(df_one_fixed, f"WS_STOCK_DF(stock={current_stock})[after_fix]")

        # build msg_copy with current stock (keep as-is)
        msg_copy = dict(message or {})
        msg_copy["output_stock_no"] = str(current_stock)

        if "output_stock_name" in df_one.columns:
            names = df_one["output_stock_name"].dropna().unique().tolist()
            if names:
                msg_copy["output_stock_name"] = str(names[0])

        if p3_1_log:
            p3_1_log.info(
                f"[compute_correlation] WS-level CURRENT-stock correlation "
                f"(wsId={scope_id}, output_stock_no={current_stock})"
            )

        _compute_and_save_from_df(
            df_one_fixed,
            msg_copy,
            p3_1_log=p3_1_log,
            algorithm=algorithm,
            scope=scope,
            scope_id=scope_id,
        )
        return  # done (no other stocks, no fallback)

    # ------------------ default path: PID or global WS ------------------
    if p3_1_log:
        p3_1_log.info("[compute_correlation] PATH: global correlation (message + history)")

    # THIS already includes CURRENT message + fetched raw history
    df = extract_cntReads_to_df_with_message(sensor_values, message)

    _dbg_df_stats(df, "GLOBAL_DF(before_fix)")
    df_fixed = _prepare_df_for_corr(df, "GLOBAL_DF")
    _dbg_df_stats(df_fixed, "GLOBAL_DF(after_fix)")

    _compute_and_save_from_df(
        df_fixed,
        message,
        p3_1_log=p3_1_log,
        algorithm=algorithm,
        scope=scope,
        scope_id=scope_id,
    )


def compute_correlation2(
    sensor_values,
    message,
    p3_1_log=None,
    algorithm: str = "SPEARMAN",
    scope: str = "pid",
    scope_id=None,
    group_by_output_stock: bool = False,
):
    """
    Compute correlation matrix for the given sensor_values and save it to Cassandra.

    - scope="pid": per-process (joOpId) → ScadaCorrelationMatrix
    - scope="ws":  per-workstation (and optionally per output_stock_no)
                   → ScadaCorrelationMatrixSummary
    """

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Starting Correlation Computation "
            f"(scope={scope}, scope_id={scope_id}, algorithm={algorithm}, "
            f"group_by_output_stock={group_by_output_stock})"
        )

    # Make sure message has output_stock_no / output_stock_name for DB columns
    if isinstance(message, dict):
        st_no, st_nm = _extract_output_stock_from_message(message)
        if st_no is not None:
            message.setdefault("output_stock_no", st_no)
        if st_nm is not None:
            message.setdefault("output_stock_name", st_nm)

    # --------- WS scope: per-output_stock_no correlation ---------
    if scope == "ws" and group_by_output_stock:
        if p3_1_log:
            p3_1_log.info("[compute_correlation] Using per-output_stock_no WS correlation")

        df_stock = extract_cntReads_to_df_with_stock(sensor_values, message)

        if df_stock is None or df_stock.empty or "output_stock_no" not in df_stock.columns:
            if p3_1_log:
                p3_1_log.warning(
                    "[compute_correlation] Could not build per-stock DF; "
                    "falling back to global WS correlation"
                )
        else:
            unique_stocks = [
                s for s in df_stock["output_stock_no"].dropna().unique().tolist() if s not in ("", None)
            ]

            if not unique_stocks and p3_1_log:
                p3_1_log.warning(
                    "[compute_correlation] No non-empty output_stock_no values; "
                    "falling back to global WS correlation"
                )
            else:
                for st_no in unique_stocks:
                    df_one = df_stock[df_stock["output_stock_no"] == st_no].copy()
                    if df_one.empty:
                        continue

                    msg_copy = dict(message or {})
                    msg_copy["output_stock_no"] = str(st_no)

                    if "output_stock_name" in df_one.columns:
                        names = df_one["output_stock_name"].dropna().unique().tolist()
                        if names:
                            msg_copy["output_stock_name"] = str(names[0])

                    if p3_1_log:
                        p3_1_log.info(
                            f"[compute_correlation] WS-level per-stock correlation for "
                            f"wsId={scope_id}, output_stock_no={st_no}"
                        )

                    _compute_and_save_from_df(
                        df_one,
                        msg_copy,
                        p3_1_log=p3_1_log,
                        algorithm=algorithm,
                        scope=scope,
                        scope_id=scope_id,
                    )

                # we've already computed per-stock correlations; we're done
                return

    # --------- default path: global correlation (pid or ws) ---------
    df = extract_cntReads_to_df_with_message(sensor_values, message)
    _compute_and_save_from_df(
        df,
        message,
        p3_1_log=p3_1_log,
        algorithm=algorithm,
        scope=scope,
        scope_id=scope_id,
    )


#### Helper Functions for Correlation Matrix Summary (unchanged)
from collections import defaultdict

def _frozen_list_to_dict(frozen_list):
    """
    Convert stored frozen list format:
      [{A:{B:r,...}}, {C:{...}}, ...]  ->  {A:{B:r,...}, C:{...}}
    """
    out = {}
    for item in (frozen_list or []):
        if isinstance(item, dict):
            for k, v in item.items():
                out[k] = v or {}
    return out

def aggregate_correlation_data(corr_list, p3_1_log=None):
    """
    corr_list: List of frozen correlation matrices
               each like [{A:{B:ρ,...}}, {B:{...}}, ...]
    Returns: nested dict {A:{B: ρ*}} aggregated via equal-weight Fisher z.
    """
    if p3_1_log:
        p3_1_log.info("[aggregate_correlation_data] Starting Correlation Aggregation")

    acc = defaultdict(lambda: defaultdict(lambda: {'sum_z': 0.0, 'k': 0}))

    for corr_frozen in corr_list:
        C = _frozen_list_to_dict(corr_frozen)
        for s1, row in (C or {}).items():
            if not isinstance(row, dict):
                continue
            for s2, r in row.items():
                if r is None or np.isnan(r):
                    continue
                # clamp to avoid atanh(±1)
                r_clip = float(np.clip(r, -0.999999, 0.999999))
                z = np.arctanh(r_clip)
                acc[s1][s2]['sum_z'] += z
                acc[s1][s2]['k']     += 1

    out = {}
    for s1, row in acc.items():
        out[s1] = {}
        for s2, v in row.items():
            if v['k'] > 0:
                out[s1][s2] = float(np.tanh(v['sum_z'] / v['k']))

    if p3_1_log:
        p3_1_log.info(f"[aggregate_correlation_data] Completed Correlation Aggregation: {out}")
    return out
