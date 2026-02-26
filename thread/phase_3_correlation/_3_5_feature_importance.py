# thread/phase_3_correlation/_3_5_feature_importance.py
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from utils.config_reader import ConfigReader
cfg = ConfigReader()


from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None


# Cassandra models are loaded lazily (only when persistence is enabled)
ScadaFeatureImportanceValues = None
ScadaFeatureImportanceValuesSummary = None
_to_utc_fi = None
_to_utc_fi_sum = None
_CASSANDRA_FI_MODELS_OK = False
_CASSANDRA_FI_MODELS_ERR = None

def _ensure_fi_models(p3_1_log=None):
    global ScadaFeatureImportanceValues, ScadaFeatureImportanceValuesSummary
    global _to_utc_fi, _to_utc_fi_sum
    global _CASSANDRA_FI_MODELS_OK, _CASSANDRA_FI_MODELS_ERR
    if _CASSANDRA_FI_MODELS_OK:
        return True
    if _CASSANDRA_FI_MODELS_ERR is not None:
        return False
    try:
        from cassandra_utils.models.scada_feature_importance_values import (
            ScadaFeatureImportanceValues as _FIV,
            _to_utc as _to_utc_fi_local,
        )
        from cassandra_utils.models.scada_feature_importance_values_summary import (
            ScadaFeatureImportanceValuesSummary as _FIVS,
            _to_utc as _to_utc_fi_sum_local,
        )
        ScadaFeatureImportanceValues = _FIV
        ScadaFeatureImportanceValuesSummary = _FIVS
        _to_utc_fi = _to_utc_fi_local
        _to_utc_fi_sum = _to_utc_fi_sum_local
        _CASSANDRA_FI_MODELS_OK = True
        return True
    except Exception as e:  # pragma: no cover
        _CASSANDRA_FI_MODELS_ERR = e
        if p3_1_log:
            p3_1_log.warning(
                f"[feature_importance] Cassandra models import failed; persistence disabled: {e}"
            )
        return False

# ----------------- GLOBALS -----------------
PHASE3_DERIVED_PERSIST_ENABLED = bool(getattr(cfg, "phase3_derived_persist_enabled", True))

def _is_input_type(equipment_type) -> bool:
    """
    Check if equipment_type indicates INPUT variable.
    
    FIXED: Now handles both boolean and string values.
    
    Returns:
        True if INPUT, False if OUTPUT, False if unknown
    """
    # Handle boolean (direct from DW)
    if equipment_type is True:
        return True
    if equipment_type is False:
        return False
    
    # ===== FIX: Handle string (after _map_to_text) =====
    if isinstance(equipment_type, str):
        lower_val = equipment_type.lower().strip()
        if lower_val == "true":
            return True
        if lower_val == "false":
            return False
    
    # Default: OUTPUT
    return False

# ----------------- helpers for names/selection -----------------

def _normalize_prefixed(names, prefix):
    if not names:
        return []
    out = []
    for n in names:
        s = str(n)
        out.append(s if s.startswith(prefix) else f"{prefix}{s}")
    return out

def _intersect_or_fallback(df_cols, wanted, fallback_prefixes):
    cols = set(df_cols)
    chosen = [c for c in wanted if c in cols]
    if chosen:
        return chosen
    # fallback: take *all* columns with the given prefixes
    fallback = []
    for pref in fallback_prefixes:
        fallback.extend([c for c in df_cols if c.startswith(pref)])
    return fallback

def _coerce_to_bool(series: pd.Series) -> pd.Series:
    """
    'true'/'false', 'TRUE'/'FALSE', 1/0, True/False vs. her şeyi temiz bool'a çevir.
    Anlaşılmayanları None bırak.
    """
    def _conv(v):
        # Already boolean
        if isinstance(v, bool):
            return v
        
        # None or NaN
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        
        # String conversion (case-insensitive)
        s = str(v).strip().upper()
        if s in ("TRUE", "1", "T", "YES", "Y"):
            return True
        if s in ("FALSE", "0", "F", "NO", "N", ""):
            return False
        
        # Numeric conversion
        try:
            num = float(s)
            if num > 0:
                return True
            elif num == 0:
                return False
        except:
            pass
        
        return None

    return series.map(_conv)

# ----------------- 1) derive goodCnt by rule -----------------

def derive_goodcnt(produce, good):
    """
    if produce == 'PRODUCTION' and good == True   -> True
    if produce == 'PRODUCTION' and good == False  -> False
    if produce != 'PRODUCTION' or good is null -> None
    """
    p = pd.Series(produce, dtype="object")
    g = pd.Series(good, dtype="object")

    # ===== CRITICAL FIX: Convert 'good' to boolean FIRST =====
    g_series = _coerce_to_bool(g)  # ← BU ÇOK ÖNEMLİ!
    
    # production mask
    is_prod = (p.astype(str).str.upper().eq("PRODUCTION") & p.notna())
    is_prod = is_prod.fillna(False).to_numpy(dtype=bool)

    # ===== NOW use boolean comparison (g_series is already bool/None) =====
    g_is_true  = (g_series == True).fillna(False).to_numpy(dtype=bool)
    g_is_false = (g_series == False).fillna(False).to_numpy(dtype=bool)

    # allocate output
    y = np.empty(len(p), dtype=object)
    y[:] = None
    y[is_prod & g_is_true]  = True
    y[is_prod & g_is_false] = False
    return y

# ----------------- 2) (opsiyonel) dataset helper -----------------
# Şu an kullanılmıyor ama ileride lazım olursa diye bırakıyorum.

def build_dataset(sensor_values: Dict[str, List[Any]],
                  produce_col: str = "produce",
                  good_col: str    = "good",
                  drop_cols: Tuple[str, ...] = ("ts",)) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    sensor_values: {col -> list/array} as you already use for correlation.
    Returns X (DataFrame of numeric features) and y_bool (np.ndarray of True/False after dropping None).
    """
    df = pd.DataFrame(sensor_values)

    # basic cleaning
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    if produce_col not in df.columns or good_col not in df.columns:
        raise ValueError(
            f"Expected columns '{produce_col}' and '{good_col}' not found in sensor_values keys: "
            f"{list(df.columns)}"
        )

    y_tri = derive_goodcnt(df[produce_col].values,
                           df[good_col].values)

    # keep only rows with True/False
    mask = pd.Series(y_tri).notna().values
    df = df.loc[mask].copy()
    y_bool = np.array([bool(v) for v in y_tri[mask]])

    # drop target helpers and any obvious non-features
    cols_to_drop = set(drop_cols) | {produce_col, good_col}
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # keep numeric only
    X = (X.select_dtypes(include=[np.number])
           .replace([np.inf, -np.inf], np.nan)
           .ffill()
           .bfill())

    if X.shape[1] == 0 or X.shape[0] < 1:
        raise ValueError("Not enough numeric features/rows for feature-importance.")

    return X, y_bool

# ----------------- 3a) eski: permutation importance (RF) -----------------
# İstersen hâlâ kullanabil diye bırakıyorum ama XGBOOST case'inde bunu kullanmayacağız.

def compute_perm_importance(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> dict:
    if X is None or X.shape[1] == 0:
        return {}

    X = (X.select_dtypes(include=[np.number])
           .replace([np.inf, -np.inf], np.nan)
           .ffill()
           .bfill())

    if X.shape[0] < 1:
        return {}

    # target tek class ise stratify kullanma
    stratify = y if (y.sum() > 0 and (~y).sum() > 0) else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=stratify
    )

    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=random_state
    )
    clf.fit(X_tr, y_tr)
    _ = f1_score(y_val, clf.predict(X_val), zero_division=0)

    pi = permutation_importance(
        clf, X_val, y_val, n_repeats=8, scoring="f1",
        random_state=random_state, n_jobs=-1
    )
    raw = {feat: float(m) for feat, m in zip(X_val.columns.tolist(), pi.importances_mean)}
    total = sum(v for v in raw.values() if v > 0)
    if total <= 0:
        k = len(raw) or 1
        return {k_: 1.0 / k for k_ in raw}
    return {k: (v / total if v > 0 else 0.0) for k, v in raw.items()}

# ----------------- 3b) yeni: XGBoost feature importance -----------------

def compute_xgb_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 4,
    n_jobs: int = 4,
) -> dict:
    """
    XGBoost'un kendi feature_importances_ çıktısını kullanarak önem hesaplar.
    Hiç permutation yok, joblib yok, dolayısıyla "can't start new thread" yok.
    """
    if X is None or X.shape[1] == 0:
        return {}

    # Sadece numerik kolonlar + basic temizlik
    X = (X.select_dtypes(include=[np.number])
           .replace([np.inf, -np.inf], np.nan)
           .ffill()
           .bfill())

    if X.shape[0] < 5:
        # çok az satır varsa model zaten anlamlı olmaz
        return {}

    # y sadece tek class ise importance anlamsız -> boş dön
    if len(np.unique(y)) < 2:
        return {}

    if XGBClassifier is None:
        # xgboost is optional; fall back to RF to keep compute-only mode usable
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            n_jobs=n_jobs,
            random_state=random_state,
        )

    # Basit: tüm veriye fit ediyoruz, importance için valid set şart değil
    clf.fit(X, y)

    # feature_importances_ doğrudan sütun sırası ile hizalı
    importances = clf.feature_importances_
    raw = {feat: float(score) for feat, score in zip(X.columns.tolist(), importances)}

    total = sum(v for v in raw.values() if v > 0)
    if total <= 0:
        k = len(raw) or 1
        return {k_: 1.0 / k for k_ in raw}
    return {k: (v / total if v > 0 else 0.0) for k, v in raw.items()}

# ----------------- 4) persistence helper (PID vs WS) -----------------

def save_feature_importance(message: dict,
                            imp_input: dict,
                            imp_output: dict,
                            algorithm: str = "rf_perm",
                            scope: str = "pid",
                            scope_id=None,
                            now_ts=None,
                            p3_1_log=None) -> None:

    """
    Decide which table to use based on scope:

    - scope="pid": ScadaFeatureImportanceValues (per process / joOpId)
    - scope="ws":  ScadaFeatureImportanceValuesSummary (per workstation + stock)
    """
    # Global persist gate (covers both PID and WS scopes)
    if not PHASE3_DERIVED_PERSIST_ENABLED:
        if p3_1_log:
            p3_1_log.info('[feature_importance] persist disabled; skipping Cassandra write')
        return

    if not _ensure_fi_models(p3_1_log=p3_1_log):
        return

    if scope == 'pid':
        if ScadaFeatureImportanceValues is None:
            if p3_1_log:
                p3_1_log.warning('[feature_importance] Cassandra models unavailable; skipping ScadaFeatureImportanceValues write')
            return
        ScadaFeatureImportanceValues.saveData(
            message=message,
            imp_input=imp_input,
            imp_output=imp_output,
            algorithm=algorithm,
            now_ts=now_ts,
            p3_1_log=p3_1_log,
        )
    elif scope == 'ws':
        if ScadaFeatureImportanceValuesSummary is None:
            if p3_1_log:
                p3_1_log.warning('[feature_importance] Cassandra models unavailable; skipping ScadaFeatureImportanceValuesSummary write')
            return
        ScadaFeatureImportanceValuesSummary.saveData(
            message=message,
            imp_input=imp_input,
            imp_output=imp_output,
            algorithm=algorithm,
            now_ts=now_ts,
            p3_1_log=p3_1_log,
        )
    else:
        if p3_1_log:
            p3_1_log.warning(f'[feature_importance] Unknown scope={scope}; nothing saved.')

# ----------------- 5) building rows from message / bundles -----------------

def _num_text(val):
    try:
        if val is None or (isinstance(val, str) and not val.strip()):
            return None
        return float(val)
    except Exception:
        return None

def _row_from_message_outputs(message: dict) -> dict:
    """
    UPDATED: Extract output variables from message with proper field priority.
    """
    row = {}
    
    # Try both possible keys
    ov_list = message.get("outVals") or message.get("outputValueList") or []
    
    for ov in ov_list:
        if not isinstance(ov, dict):
            continue
        
        # Name priority: eqNo > parameter > eqNm > param > eqId
        key = (
            ov.get("eqNo")
            or ov.get("parameter")
            or ov.get("eqNm")
            or ov.get("param")
            or ov.get("eqId")
        )
        if key is None or str(key).strip() == "":
            continue
        
        # Value priority: cntRead > counter_reading > genReadVal > value
        val = (
            ov.get("cntRead")
            or ov.get("counter_reading")
            or ov.get("genReadVal")
            or ov.get("value")
        )
        
        row[f"out_{str(key)}"] = _num_text(val)
    
    return row


def _row_from_message_inputs(message: dict) -> dict:
    """
    UPDATED: Extract input variables from message with proper field priority.
    """
    row = {}
    
    # Try both possible keys
    iv = message.get("inVars") or message.get("inputVariableList")
    if not iv:
        return row

    # Handle both list and single dict
    if isinstance(iv, dict):
        iv_list = [iv]
    elif isinstance(iv, list):
        iv_list = iv
    else:
        return row

    for it in iv_list:
        if not isinstance(it, dict):
            continue

        # Name priority: varNm > varId > name > param > eqNo
        key = (
            it.get("varNo")
            or it.get("varNm")
            or it.get("varId")
            or it.get("eqNo")
        )
        if key is None or str(key).strip() == "":
            continue

        # Value priority: genReadVal > actVal > value > cntRead
        val = (
            it.get("genReadVal")
            or it.get("actVal")
            or it.get("value")
            or it.get("cntRead")
        )

        row[f"in_{str(key)}"] = _num_text(val)
    
    return row


def _bundles_to_wide_df(sensor_values, message: dict) -> pd.DataFrame:
    # ---- 1) history (DW/RAW/buffer) -> wide ----
    df_hist = pd.DataFrame()
    if isinstance(sensor_values, list) and len(sensor_values) > 0:
        rows = []
        for bundle in sensor_values:
            if not bundle:
                continue

            meta, *sensors = bundle
            row = {}

            if isinstance(meta, dict):
                # ts
                crdt = meta.get("crDt")
                if crdt is not None:
                    try:
                        row["ts"] = int(crdt)
                    except Exception:
                        pass

                # row bazlı label / id alanları
                if "prSt" in meta:
                    row["prSt"] = meta["prSt"]
                if "good" in meta:
                    row["good"] = meta["good"]

                # joOpId / wsId
                if "joOpId" in meta:
                    row["joOpId"] = meta["joOpId"]
                elif "job_order_operation_id" in meta:
                    row["joOpId"] = meta["job_order_operation_id"]

                if "wsId" in meta:
                    row["wsId"] = meta["wsId"]
                elif "workstation_id" in meta:
                    row["wsId"] = meta["workstation_id"]
                elif "work_station_id" in meta:
                    row["wsId"] = meta["work_station_id"]

            # ===== SENSOR VALUES - NEW: Check equipment_type =====
            # ===== SENSOR VALUES - Use _is_input_type() helper =====
            for s in sensors:
                if not isinstance(s, dict):
                    continue
                
                # Get sensor name
                key = (
                    s.get("equipment_name")
                    or s.get("eqNm")
                    or s.get("parameter")
                    or s.get("eqNo")
                    or s.get("param")
                )
                if key is None:
                    continue
                
                # ===== FIX: Use _is_input_type() helper =====
                equipment_type = s.get("equipment_type")
                
                if _is_input_type(equipment_type):
                    # INPUT
                    val = (
                        s.get("gen_read_val") 
                        or s.get("genReadVal") 
                        or s.get("counter_reading") 
                        or s.get("cntRead") 
                        or s.get("value")
                    )
                    if val is not None:
                        row[f"in_{str(key)}"] = _num_text(val)
                
                else:
                    # OUTPUT (default)
                    val = (
                        s.get("counter_reading") 
                        or s.get("cntRead") 
                        or s.get("value")
                    )
                    if val is not None:
                        row[f"out_{str(key)}"] = _num_text(val)
                
                # else: equipment_type is None or other -> SKIP completely

            if row:
                rows.append(row)

        if rows:
            df_hist = pd.DataFrame(rows)
            if "ts" in df_hist.columns:
                df_hist = df_hist.sort_values("ts").reset_index(drop=True)

    # ---- 2) bu mesajdan tek satır çıkar (inVars + outVals + label + id) ----
    row_msg = {}
    
    # INPUTS (inVars)
    row_msg.update(_row_from_message_inputs(message))
    
    # OUTPUTS (outVals)
    row_msg.update(_row_from_message_outputs(message))

    # ts
    if "crDt" in message:
        try:
            row_msg.setdefault("ts", int(message["crDt"]))
        except Exception:
            pass

    # label alanları
    if "prSt" in message:
        row_msg.setdefault("prSt", message["prSt"])
    if "good" in message:
        row_msg.setdefault("good", message["good"])
    elif "goodCnt" in message:
        row_msg.setdefault("good", message["goodCnt"])

    # id alanları
    if "joOpId" in message:
        row_msg.setdefault("joOpId", message["joOpId"])
    if "wsId" in message:
        row_msg.setdefault("wsId", message["wsId"])

    df_msg = pd.DataFrame([row_msg]) if row_msg else pd.DataFrame()

    # ---- 3) history + mesajı birleştir ----
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

    # history + en son mesaj
    df = pd.concat([df_hist, df_msg], ignore_index=True)
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    return df

def _bundles_to_wide_df2(sensor_values, message: dict) -> pd.DataFrame:
    # ---- 1) history (DW/RAW/buffer) -> wide ----
    df_hist = pd.DataFrame()
    if isinstance(sensor_values, list) and len(sensor_values) > 0:
        rows = []
        for bundle in sensor_values:
            if not bundle:
                continue

            meta, *sensors = bundle
            row = {}

            if isinstance(meta, dict):
                # ts
                crdt = meta.get("crDt")
                if crdt is not None:
                    try:
                        row["ts"] = int(crdt)
                    except Exception:
                        pass

                # row bazlı label / id alanları
                if "prSt" in meta:
                    row["prSt"] = meta["prSt"]
                if "good" in meta:
                    row["good"] = meta["good"]

                # joOpId / wsId
                if "joOpId" in meta:
                    row["joOpId"] = meta["joOpId"]
                elif "job_order_operation_id" in meta:
                    row["joOpId"] = meta["job_order_operation_id"]

                if "wsId" in meta:
                    row["wsId"] = meta["wsId"]
                elif "workstation_id" in meta:
                    row["wsId"] = meta["workstation_id"]
                elif "work_station_id" in meta:
                    row["wsId"] = meta["work_station_id"]

            # ===== SENSOR VALUES - NEW: Check equipment_type =====
            # ===== SENSOR VALUES - STRICT equipment_type filtering =====
            for s in sensors:
                if not isinstance(s, dict):
                    continue
                
                # CRITICAL: Check equipment_type FIRST
                equipment_type = s.get("equipment_type")
                
                # Get sensor name
                key = (
                    s.get("equipment_name")
                    or s.get("eqNm")
                    or s.get("parameter")
                    or s.get("eqNo")
                    or s.get("param")
                )
                if key is None:
                    continue
                
                # STRICT TYPE CHECK
                if equipment_type is True:
                    # INPUT ONLY
                    val = (
                        s.get("gen_read_val") 
                        or s.get("genReadVal") 
                        or s.get("counter_reading") 
                        or s.get("cntRead") 
                        or s.get("value")
                    )
                    if val is not None:  # Only add if value exists
                        row[f"in_{str(key)}"] = _num_text(val)
                
                elif equipment_type is False:
                    # OUTPUT ONLY
                    val = (
                        s.get("counter_reading") 
                        or s.get("cntRead") 
                        or s.get("value")
                    )
                    if val is not None:  # Only add if value exists
                        row[f"out_{str(key)}"] = _num_text(val)
                
                # else: equipment_type is None or other -> SKIP completely

            if row:
                rows.append(row)

        if rows:
            df_hist = pd.DataFrame(rows)
            if "ts" in df_hist.columns:
                df_hist = df_hist.sort_values("ts").reset_index(drop=True)

    # ---- 2) bu mesajdan tek satır çıkar (inVars + outVals + label + id) ----
    row_msg = {}
    
    # INPUTS (inVars)
    row_msg.update(_row_from_message_inputs(message))
    
    # OUTPUTS (outVals)
    row_msg.update(_row_from_message_outputs(message))

    # ts
    if "crDt" in message:
        try:
            row_msg.setdefault("ts", int(message["crDt"]))
        except Exception:
            pass

    # label alanları
    if "prSt" in message:
        row_msg.setdefault("prSt", message["prSt"])
    if "good" in message:
        row_msg.setdefault("good", message["good"])
    elif "goodCnt" in message:
        row_msg.setdefault("good", message["goodCnt"])

    # id alanları
    if "joOpId" in message:
        row_msg.setdefault("joOpId", message["joOpId"])
    if "wsId" in message:
        row_msg.setdefault("wsId", message["wsId"])

    df_msg = pd.DataFrame([row_msg]) if row_msg else pd.DataFrame()

    # ---- 3) history + mesajı birleştir ----
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

    # history + en son mesaj
    df = pd.concat([df_hist, df_msg], ignore_index=True)
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    return df




# ----------------- 6) orchestrator -----------------

def compute_and_save_feature_importance(
    sensor_values: Union[dict, list],
    message: dict,
    input_feature_names=None,
    output_feature_names=None,
    produce_col="prSt",
    good_col="good",
    drop_cols=("ts","joOpId","wsId"),
    algorithm="XGBOOST",   # <<< default'ı da XGBOOST yaptım
    p3_1_log=None,
    scope: str = "pid",
    scope_id=None
):
    """
    Compute feature importance for pid or ws scope and store in the right table.

    - scope="pid": per-process importance -> scada_feature_importance_values
    - scope="ws":  per-workstation/stock importance -> scada_feature_importance_values_summary

    algorithm:
      - "XGBOOST"    -> XGBClassifier feature_importances_
      - "RF_PERM"    -> RandomForest + permutation_importance
    """
    algo_upper = (algorithm or "").upper()

    # Build DF from bundles or (if empty) from Kafka message
    df = _bundles_to_wide_df(sensor_values, message)
    
    if p3_1_log:
        p3_1_log.info(
            f"[feature_importance] Built DataFrame with shape {df.shape} and columns: {list(df.columns)}"
        )
        # ===== DEBUG: Check 'good' column BEFORE coercion =====
        if good_col in df.columns:
            p3_1_log.info(
                f"[feature_importance] RAW 'good' values (first 10): "
                f"{df[good_col].head(10).tolist()}"
            )
            p3_1_log.info(
                f"[feature_importance] RAW 'good' dtypes: {df[good_col].dtype}"
            )

    if df.empty:
        if p3_1_log:
            p3_1_log.warning("[feature_importance] No feature rows (df empty after message fallback).")
        return

    # Inject target helpers if missing
    if produce_col not in df.columns:
        df[produce_col] = message.get("prSt")
    if good_col not in df.columns:
        if "good" in message:
            df[good_col] = message["good"]
        elif "goodCnt" in message:
            df[good_col] = message["goodCnt"]
        else:
            df[good_col] = pd.NA

    # >>> BURAYI EKLE <<<
    # DW/RAW tarafında 'good' çoğu zaman "true"/"false" string'e dönmüş durumda,
    # bunları tekrar bool'a çeviriyoruz ki derive_goodcnt yakalasın.
    if good_col in df.columns:
        df[good_col] = _coerce_to_bool(df[good_col])
        # ===== DEBUG: Check 'good' column AFTER coercion =====
        if p3_1_log:
            p3_1_log.info(
                f"[feature_importance] COERCED 'good' values (first 10): "
                f"{df[good_col].head(10).tolist()}"
            )

    # Derive target with your rule
    y_tri_series = pd.Series(derive_goodcnt(df[produce_col].values, df[good_col].values))
    mask = y_tri_series.notna()

    if p3_1_log:
        p3_1_log.info(f"[feature_importance] mask: {int(mask.sum())} labeled rows")

    if p3_1_log:
        p3_1_log.info(
            "[feature_importance] debug labels: "
            + str(df.loc[:, [produce_col, good_col]].value_counts().to_dict())
        )

    if int(mask.sum()) < 1:
        if p3_1_log:
            p3_1_log.warning(
                "[feature_importance] Not enough labeled rows after goodCnt rule."
            )
        return

    # filter df and make boolean target
    df = df.loc[mask].copy()
    y  = y_tri_series.loc[mask].astype(bool).to_numpy()

    if p3_1_log:
        p3_1_log.info(
            f"[feature_importance] after mask: {df.shape[0]} rows, {df.shape[1]} cols"
        )
        p3_1_log.info(
            f"[feature_importance] DF cols before drop: {list(df.columns)}"
        )

    # Drop helpers
    for c in (*drop_cols, produce_col, good_col):
        if c in df.columns:
            df.drop(columns=c, inplace=True, errors="ignore")

    if p3_1_log:
        p3_1_log.info(
            f"[feature_importance] DF cols after drop: {list(df.columns)}"
        )

    # Pick feature groups - IGNORE input_feature_names/output_feature_names from caller
    # We ONLY trust the prefix in the DataFrame columns built by _bundles_to_wide_df()

    # ===== INPUT features: ONLY columns starting with "in_" =====
    input_feature_names = [c for c in df.columns if c.startswith("in_")]

    # ===== OUTPUT features: ONLY columns starting with "out_" =====
    output_feature_names = [c for c in df.columns if c.startswith("out_")]

    if p3_1_log:
        p3_1_log.info(
            f"[feature_importance] Selected {len(input_feature_names)} INPUT features, "
            f"{len(output_feature_names)} OUTPUT features based on prefix"
        )
    
    # ===== VALIDATION: Check for contamination (SAFE - we know these are strings) =====
    if p3_1_log:
        sample_in = input_feature_names[:5] if input_feature_names else []
        sample_out = output_feature_names[:5] if output_feature_names else []
        
        p3_1_log.info(
            f"[feature_importance] INPUT sample: {sample_in}"
        )
        p3_1_log.info(
            f"[feature_importance] OUTPUT sample: {sample_out}"
        )
        
        # Strip prefixes and check overlap
        in_names_stripped = {c.replace("in_", "") for c in input_feature_names}
        out_names_stripped = {c.replace("out_", "") for c in output_feature_names}
        overlap = in_names_stripped & out_names_stripped
        
        if overlap:
            p3_1_log.warning(
                f"[feature_importance] CONTAMINATION: {len(overlap)} vars in BOTH: {list(overlap)[:10]}"
            )
        else:
            p3_1_log.info(
                f"[feature_importance] Clean separation: no overlap"
            )

    X_in  = df[input_feature_names]  if input_feature_names  else pd.DataFrame(index=df.index)
    X_out = df[output_feature_names] if output_feature_names else pd.DataFrame(index=df.index)

    # ---- burada algoritmaya göre seçim yapıyoruz ----
    if algo_upper == "XGBOOST":
        imp_in  = compute_xgb_importance(X_in,  y) if X_in.shape[1]  else {}
        imp_out = compute_xgb_importance(X_out, y) if X_out.shape[1] else {}
    else:
        # fallback: eski RF + permutation (istersen iptal edebilirsin)
        imp_in  = compute_perm_importance(X_in,  y) if X_in.shape[1]  else {}
        imp_out = compute_perm_importance(X_out, y) if X_out.shape[1] else {}

    if not imp_in and input_feature_names:
        # Hepsine eşit önem ver
        w = 1.0 / len(input_feature_names)
        imp_in = {f: w for f in input_feature_names}

    if not imp_out and output_feature_names:
        # Hepsine eşit önem ver
        w = 1.0 / len(output_feature_names)
        imp_out = {f: w for f in output_feature_names}
        
    
    # Tarihi mesajın crDt / createDate değerinden al
    try:
        now_ts = _to_utc_fi(message.get("crDt")) \
                 or _to_utc_fi(message.get("createDate")) \
                 or datetime.now(timezone.utc)
    except Exception:
        now_ts = datetime.now(timezone.utc)

    # Persist according to scope
    save_feature_importance(
        message=message,
        imp_input=imp_in,
        imp_output=imp_out,
        algorithm=algorithm,
        scope=scope,
        scope_id=scope_id,
        now_ts=now_ts,          # <<< buraya ekle
        p3_1_log=p3_1_log
    )

    if p3_1_log:
        top_in  = ", ".join(
            f"{k}:{v:.3f}" for k,v in sorted(imp_in.items(),  key=lambda kv: kv[1], reverse=True)[:5]
        ) or "(none)"
        top_out = ", ".join(
            f"{k}:{v:.3f}" for k,v in sorted(imp_out.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ) or "(none)"
        p3_1_log.info(
            f"[feature_importance] Saved (scope={scope}, algo={algorithm}) "
            f"INPUT top5: {top_in} | OUTPUT top5: {top_out}"
        )
