import time
import hashlib
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pytz

from utils.config_reader import ConfigReader
from utils.logger_2 import setup_logger
from utils.identity import get_workstation_uid, get_stock_key, NULL_TOKENS
from utils.keypoint_recorder import KP

from runtime.event_bus import enqueue_phase2, dequeue_phase2, task_done_phase2, qsize
from thread.phase_trigger_bus import request_anomaly

from modules.phase2.frame_builder import EventTimeBucketizer
from modules.phase2.state_store import Phase2StateStore
from modules.phase2.registry import Phase2Registry
from modules.phase2.models.lstm_ae import LSTMAEModel
from modules.batching.batch_assigner import extract_event_ts_ms

# safe_token MUST match offline artifacts
from modules.offline_outonly_trainer import safe_token

p2_rt_log = setup_logger("p2_runtime_log", "logs/p2_runtime.log")

cfg = ConfigReader()

# Lazy import: allow unit-level imports without requiring Cassandra driver in all environments.
_PERSIST_FN = None
def _persist_phase2_score(*args, **kwargs):
    global _PERSIST_FN
    if _PERSIST_FN is None:
        from modules.phase2.persist_dw_tbl_multiple_anomalies import persist_phase2_score as fn
        _PERSIST_FN = fn
    return _PERSIST_FN(*args, **kwargs)

# Back-compat alias (some local experiments referenced the double-underscore name).
__persist_phase2_score = _persist_phase2_score


PHASE2_ENABLED = bool(getattr(cfg, "phase2_enabled", True))

# Stub scorer fallback (kept for cold-start / no-model cases)
PHASE2_STUB_ENABLED = bool(getattr(cfg, "phase2_stub_enabled", True))
PHASE2_STUB_WHEN_MODEL_PRESENT = bool(getattr(cfg, "phase2_stub_when_model_present", False))
PHASE2_WINDOW_N = int(getattr(cfg, "phase2_stub_window_n", 200) or 200)
PHASE2_MIN_HIST = int(getattr(cfg, "phase2_stub_min_hist", 30) or 30)
PHASE2_Z_TH = float(getattr(cfg, "phase2_stub_z_thresh", 6.0) or 6.0)

# Importance / heatmap controls
PHASE2_IMPORTANCE_TOPK = int(getattr(cfg, "phase2_importance_topk", 5) or 5)
PHASE2_HEATMAP_TOPN = int(getattr(cfg, "phase2_heatmap_topn", 32) or 32)

# Trigger controls (D adds persistence/cooldown; keep simple here)
PHASE2_TRIGGER_TTL = int(getattr(cfg, "phase2_trigger_ttl_sec", 300) or 300)
PHASE2_TRIGGER_COOLDOWN_SEC = int(getattr(cfg, "phase2_trigger_cooldown_sec", 60) or 60)
PHASE2_TRIGGER_PERSIST_K = int(getattr(cfg, "phase2_trigger_persist_k", 2) or 2)
PHASE2_TRIGGER_PERSIST_M = int(getattr(cfg, "phase2_trigger_persist_m", 3) or 3)

PHASE2_TRIGGER_ENABLED = bool(getattr(cfg, "phase2_trigger_enabled", True))
PHASE2_TRIGGER_SCOPE = str(getattr(cfg, "phase2_trigger_scope", "ws") or "ws").strip().lower()

# Patch1: persist Phase2 scores (bucket-level) into Cassandra dw_tbl_multiple_anomalies
PHASE2_PERSIST_ENABLED = bool(getattr(cfg, "phase2_persist_enabled", True))
PHASE2_PERSIST_ONLY_ANOMALIES = bool(getattr(cfg, "phase2_persist_only_anomalies", False))

# M3.0-B: event-time resampling guards
RESAMPLE_SECONDS = int(getattr(cfg, "resample_seconds", 60) or 60)
PHASE2_RESAMPLE_SECONDS = int(getattr(cfg, "phase2_resample_seconds", RESAMPLE_SECONDS) or RESAMPLE_SECONDS)
PHASE2_MAX_GAP_SEC = int(
    getattr(cfg, "phase2_max_gap_sec", min(max(6 * PHASE2_RESAMPLE_SECONDS, 30), 900))
    or min(max(6 * PHASE2_RESAMPLE_SECONDS, 30), 900)
)
PHASE2_ALLOWED_LATENESS_SEC = int(
    getattr(cfg, "phase2_allowed_lateness_sec", min(max(2 * PHASE2_RESAMPLE_SECONDS, 10), 300))
    or min(max(2 * PHASE2_RESAMPLE_SECONDS, 10), 300)
)

# Watermark / epoch controls for mixed real-time + backfill streams
PHASE2_EPOCH_RESET_SEC = int(getattr(cfg, "phase2_epoch_reset_sec", 6 * 3600) or (6 * 3600))
PHASE2_MAX_OPEN_BUCKETS = int(getattr(cfg, "phase2_max_open_buckets", 0) or 0)
PHASE2_MAX_SENSORS = int(getattr(cfg, "phase2_max_sensors", 64) or 64)

# Bounded state controls
PHASE2_MAX_ACTIVE_KEYS = int(getattr(cfg, "phase2_max_active_keys", 5000) or 5000)
PHASE2_KEY_TTL_SEC = int(getattr(cfg, "phase2_key_ttl_sec", 3600) or 3600)

_STATE_STORE = Phase2StateStore(max_active_keys=PHASE2_MAX_ACTIVE_KEYS, key_ttl_sec=PHASE2_KEY_TTL_SEC)

# Legacy per-sensor history for stub z-score
_HIST: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=PHASE2_WINDOW_N)))

# Registry (M3.0-C): load LSTM-AE artifacts produced by offline_phase2_unsup_trainer
PHASE2_REGISTRY_ENABLED = bool(getattr(cfg, "phase2_registry_enabled", True))
PHASE2_MODELFAMILY = str(getattr(cfg, "phase2_model_family", "lstm_ae") or "lstm_ae").strip().lower()
PHASE2_CONTEXT_POLICY_ID = str(getattr(cfg, "phase2_context_policy_id", "ws_stock") or "ws_stock").strip().lower()
PHASE2_REGISTRY_REFRESH_EVERY_SEC = int(getattr(cfg, "phase2_registry_refresh_every_sec", 60) or 60)

DEFAULT_SCAN_DIRS = ["./models/phase2models/offline_unsup"]
_scan_dirs_raw = str(getattr(cfg, "phase2_models_dirs", "" ) or "").strip()
SCAN_DIRS = [x.strip() for x in _scan_dirs_raw.split(",") if x.strip()] or DEFAULT_SCAN_DIRS

_REGISTRY = Phase2Registry(scan_dirs=SCAN_DIRS)
_LAST_REGISTRY_REFRESH_WALL = 0.0


def _maybe_refresh_registry() -> None:
    global _LAST_REGISTRY_REFRESH_WALL
    if not PHASE2_REGISTRY_ENABLED:
        return
    now = time.time()
    if (now - _LAST_REGISTRY_REFRESH_WALL) < float(PHASE2_REGISTRY_REFRESH_EVERY_SEC):
        return
    try:
        n = _REGISTRY.refresh()
        KP.gauge("phase2.registry.artifacts", int(n))
    except Exception:
        KP.inc("phase2.registry.refresh_err", 1)
    _LAST_REGISTRY_REFRESH_WALL = now


def _event_ts_epoch(message: dict) -> float:
    """Event-time anchor for Phase2 (seconds since epoch).

    Project policy (v2.25+): use **crDt** as the canonical event time.
    Stage0 computes and attaches `_event_ts_ms` using the same policy.
    We therefore use:
      1) message['_event_ts_ms'] if present (preferred; already parsed/normalized)
      2) else batch_assigner.extract_event_ts_ms(message) (prefers crDt)
      3) else time.time()

    Note: Using message['measDt'] directly can introduce backwards jumps under mixed fields.
    """
    v = message.get("_event_ts_ms")
    if v is not None:
        try:
            return float(v) / 1000.0
        except Exception:
            pass

    try:
        ms = int(extract_event_ts_ms(message) or 0)
        if ms > 0:
            return float(ms) / 1000.0
    except Exception:
        pass

    return time.time()


def _measurement_dt_from_finalize(fin) -> datetime:
    """Measurement timestamp to persist for a finalized bucket.

    Preference:
      - fin.bucket_last_event_epoch_ms (Patch1 adds this)
    Fallback:
      - fin.bucket_start_epoch_sec
    """
    try:
        ms = int(getattr(fin, "bucket_last_event_epoch_ms"))
        return datetime.fromtimestamp(ms / 1000.0, tz=pytz.UTC)
    except Exception:
        try:
            s = int(getattr(fin, "bucket_start_epoch_sec"))
            return datetime.fromtimestamp(s, tz=pytz.UTC)
        except Exception:
            return datetime.now(pytz.UTC)


def _extract_outvals(message: dict) -> Dict[str, float]:
    outv = message.get("outVals") or []
    m: Dict[str, float] = {}
    for it in outv:
        sid = it.get("eqNo") or it.get("eqNm") or it.get("eqId")
        if sid is None:
            continue
        sid = str(sid).strip()
        if not sid or sid.lower() in NULL_TOKENS:
            continue
        v = it.get("cntRead")
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:  # NaN
            continue
        m[sid] = fv
    return m


def _extract_op_tc(message: dict) -> str:
    # best-effort; many sources may not populate it
    v = message.get("opTc") or message.get("op_tc")
    if v is not None:
        return str(v)
    outv = message.get("outVals") or []
    for it in outv:
        if it.get("opTc") is not None:
            return str(it.get("opTc"))
    return "ALL"


def _extract_bucket_meta(message: dict) -> Dict[str, object]:
    """Extract a compact, schema-compatible context payload to attach to finalized buckets.

    This prevents persisting 'newer message' metadata for an older finalized bucket.
    """
    meta: Dict[str, object] = {}
    # context identifiers
    for k in ("refNo","joRef","joOpId","opTc","prSt","mcSt","cust","customer","empId","good","currCycQty","chngCycQty",
              "plId","plNo","plNm","wcId","wcNo","wcNm","wsId","wsNo","wsNm"):
        if k in message:
            meta[k] = message.get(k)
    # product (first)
    prod = message.get("prodList") or []
    if isinstance(prod, dict):
        prod = [prod]
    if isinstance(prod, list) and prod and isinstance(prod[0], dict):
        p0 = prod[0]
        meta["prodList"] = [{
            "stId": p0.get("stId"),
            "stNo": p0.get("stNo"),
            "stNm": p0.get("stNm"),
        }]
    return meta



def _update_sensor_static(st_user: dict, message: dict) -> None:
    """Cache per-sensor static metadata (eqNo/eqNm/eqId, etc.) for payload compatibility.

    The backend often expects sensor_values/heatmap inner maps to include identifiers like eqNm.
    Since Phase2 persists bucket-level values (not raw outVals), we keep a small cache keyed by
    the extracted sensor id and reuse it when building persisted maps.
    """
    try:
        outv = message.get("outVals") or []
        if not isinstance(outv, list):
            return
        cache = st_user.setdefault("sensor_static", {})
        if not isinstance(cache, dict):
            cache = {}
            st_user["sensor_static"] = cache
        for it in outv:
            if not isinstance(it, dict):
                continue
            sid = it.get("eqNo") or it.get("eqNm") or it.get("eqId")
            if sid is None:
                continue
            sid = str(sid).strip()
            if not sid or sid.lower() in NULL_TOKENS:
                continue
            meta = {}
            for k in ("eqNo","eqNm","eqId","eqUnit","eqTp","eqType","equipment_no","equipment_name"):
                if k in it and it.get(k) is not None:
                    meta[k] = it.get(k)
            # Always include the key itself as a fallback for names.
            if "eqNo" not in meta:
                meta["eqNo"] = sid
            if "eqNm" not in meta:
                meta["eqNm"] = sid
            cache[sid] = meta
    except Exception:
        return

def _mean_topk(contrib: Dict[str, float], k: int) -> Optional[float]:
    if not contrib:
        return None
    items = []
    for _, v in contrib.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:
            continue
        items.append(abs(fv))
    if not items:
        return None
    items.sort(reverse=True)
    k = max(1, min(int(k), len(items)))
    return float(sum(items[:k]) / float(k))



def _robust_z(x: float, hist: deque) -> Optional[float]:
    if len(hist) < PHASE2_MIN_HIST:
        return None
    xs = sorted(hist)
    n = len(xs)
    med = xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    abs_dev = sorted([abs(v - med) for v in xs])
    mad = abs_dev[n // 2] if n % 2 == 1 else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
    if mad <= 0.0:
        return None
    return 0.6745 * (x - med) / mad


def phase2_enqueue(message: dict) -> None:
    if not PHASE2_ENABLED:
        return
    ok = enqueue_phase2(message)
    if ok:
        KP.inc("phase2.enqueue.ok", 1)
    else:
        KP.inc("phase2.enqueue.drop", 1)
        try:
            ws_uid = get_workstation_uid(message)
        except Exception:
            ws_uid = ""
        p2_rt_log.warning(f"[phase2] queue full -> drop ws={ws_uid} q={qsize()}")


def _bucketizer_factory():
    return EventTimeBucketizer(
        resample_sec=PHASE2_RESAMPLE_SECONDS,
        max_gap_sec=PHASE2_MAX_GAP_SEC,
        allowed_lateness_sec=PHASE2_ALLOWED_LATENESS_SEC,
        max_sensors=PHASE2_MAX_SENSORS,
        epoch_reset_sec=PHASE2_EPOCH_RESET_SEC,
        max_open_buckets=(PHASE2_MAX_OPEN_BUCKETS if int(PHASE2_MAX_OPEN_BUCKETS) > 0 else None),
    )



def _artifact_seq_key(art) -> str:
    # Stable key to prevent cross-opTc / cross-sensor-order contamination in seq buffers.
    cols = list(getattr(art, "sensor_cols", None) or [])
    h = hashlib.md5(("|".join(cols)).encode("utf-8")).hexdigest()[:12]
    return f"{art.model_family}|{art.stock_key}|{art.op_tc}|r{int(art.resample_sec)}|t{int(art.timesteps)}|c{h}"


def _get_seq_buffer(st_user: dict, *, seq_key: str, timesteps: int) -> deque:
    """Return the deque buffer for the given model/artifact signature."""
    store = st_user.get("seq_by_model")
    if not isinstance(store, dict):
        store = {}
        st_user["seq_by_model"] = store

    dq = store.get(seq_key)
    if dq is None or (not isinstance(dq, deque)) or dq.maxlen != int(timesteps):
        dq = deque(maxlen=int(timesteps))
        store[seq_key] = dq

    # Track LRU order so the dict doesn't grow unbounded across opTc/model switches.
    max_keep = int(getattr(cfg, "phase2_seq_keys_keep", 4) or 4)
    order = st_user.get("seq_key_order")
    if not isinstance(order, deque) or order.maxlen != max_keep:
        order = deque(maxlen=max_keep)
        st_user["seq_key_order"] = order

    # refresh LRU order
    try:
        if seq_key in order:
            order.remove(seq_key)
        order.append(seq_key)
    except Exception:
        pass

    # prune old keys if needed
    try:
        while len(store) > max_keep and len(order) > 0:
            old = order[0]
            if old == seq_key:
                break
            order.popleft()
            store.pop(old, None)
    except Exception:
        pass

    st_user["active_seq_key"] = seq_key
    return dq


def _trigger_guard(st_user: dict, *, is_anomaly: bool) -> bool:
    """K-of-M persistence + cooldown guard.

    - persistence: require at least K anomalies in last M scored buckets
    - cooldown: minimum time between triggers per key
    """
    k = max(1, int(PHASE2_TRIGGER_PERSIST_K))
    m = max(k, int(PHASE2_TRIGGER_PERSIST_M))

    dq = st_user.get("anom_hist")
    if dq is None or not isinstance(dq, deque) or dq.maxlen != m:
        dq = deque(maxlen=m)
        st_user["anom_hist"] = dq

    dq.append(1 if bool(is_anomaly) else 0)

    if sum(dq) < k:
        KP.inc("phase2.trigger.skip.persistence", 1)
        return False

    last = float(st_user.get("last_trigger_wall") or 0.0)
    now = time.time()
    if (now - last) < float(PHASE2_TRIGGER_COOLDOWN_SEC):
        KP.inc("phase2.trigger.skip.cooldown", 1)
        return False

    st_user["last_trigger_wall"] = now
    # clear history after emitting to avoid immediate re-triggers
    dq.clear()
    return True


def _try_score_with_registry(
    *,
    wsuid: str,
    stock: str,
    op_tc: str,
    fin_values: Dict[str, float],
    st_user: dict,
) -> Optional[Tuple[bool, float, float, str]]:
    """Returns (is_anom, score, threshold, model_family) or None if no model used."""
    if not PHASE2_REGISTRY_ENABLED:
        return None

    _maybe_refresh_registry()

    wsuid_token = safe_token(wsuid)
    art = _REGISTRY.find_best(
        wsuid_token=wsuid_token,
        resample_sec=int(PHASE2_RESAMPLE_SECONDS),
        context_policy_id=str(PHASE2_CONTEXT_POLICY_ID),
        stock_key=str(stock),
        op_tc=str(op_tc),
        model_family=PHASE2_MODELFAMILY if PHASE2_MODELFAMILY else None,
        prefer_accepted=True,
    )

    if art is None:
        KP.inc("phase2.model.missing", 1)
        return None

    # cache artifact in state for observability
    st_user["artifact"] = {
        "model_family": art.model_family,
        "timesteps": int(art.timesteps),
        "resample_sec": int(art.resample_sec),
        "stock_key": art.stock_key,
        "op_tc": art.op_tc,
        "trained_at_utc": art.trained_at_utc,
    }

    # per-call flags (used to decide stub fallback)
    st_user["artifact"]["load_ok"] = False
    st_user["artifact"]["scale_ok"] = False
    st_user["artifact"]["warmup_left"] = None

    try:
        lr = _REGISTRY.load_model(art)
        model = LSTMAEModel(keras_model=lr.model, threshold=float(art.threshold))
        scaler = _REGISTRY.load_scaler(art)
        st_user["artifact"]["load_ok"] = True
    except Exception:
        KP.inc("phase2.model.load_err", 1)
        return None

    timesteps = int(art.timesteps)
    seq_key = _artifact_seq_key(art)
    st_user["artifact"]["seq_key"] = seq_key
    seq = _get_seq_buffer(st_user, seq_key=seq_key, timesteps=timesteps)

    # Build feature vector in artifact sensor order
    cols = list(art.sensor_cols or [])
    if not cols:
        KP.inc("phase2.model.bad_artifact", 1)
        return None

    x = np.full((1, len(cols)), float(art.fillna_value), dtype=float)
    for i, sid in enumerate(cols):
        if sid in fin_values:
            x[0, i] = float(fin_values[sid])

    # Apply scaler if present
    try:
        if scaler is not None:
            x = scaler.transform(x)
        st_user["artifact"]["scale_ok"] = True
    except Exception:
        KP.inc("phase2.model.scale_err", 1)
        return None

    seq.append(x[0].astype(float))

    if len(seq) < timesteps:
        KP.inc("phase2.skip.insufficient_seq", 1)
        try:
            st_user["artifact"]["warmup_left"] = int(timesteps - len(seq))
        except Exception:
            pass
        return None

    X_seq = np.stack(list(seq), axis=0).reshape((1, timesteps, len(cols)))

    try:
        sc = model.score_sequence(X_seq)
    except Exception:
        KP.inc("phase2.model.score_err", 1)
        return None


    per_sensor_contrib: Dict[str, float] = {}
    try:
        det = getattr(sc, "detail", None) or {}
        per_feat = det.get("per_feature_mse")
        if isinstance(per_feat, list) and len(per_feat) == len(cols):
            per_sensor_contrib = {str(cols[i]): float(per_feat[i]) for i in range(len(cols))}
    except Exception:
        per_sensor_contrib = {}

    importance = _mean_topk(per_sensor_contrib, int(PHASE2_IMPORTANCE_TOPK))
    if importance is None:
        importance = float(sc.score)

    KP.inc("phase2.model.score_ok", 1)
    return (
        bool(sc.is_anomaly),
        float(sc.score),
        float(sc.threshold),
        str(art.model_family),
        per_sensor_contrib,
        float(importance),
    )



def execute_phase_two_worker() -> None:
    p2_rt_log.info("[phase2_runtime] started")
    while True:
        msg = dequeue_phase2()
        KP.inc("phase2.dequeue", 1)
        try:
            if not PHASE2_ENABLED:
                continue

            wsuid = get_workstation_uid(msg)
            stock = get_stock_key(msg, default="ALL")
            op_tc = _extract_op_tc(msg)
            key = f"{wsuid}|{stock}"

            st, expired, lru_removed = _STATE_STORE.get_or_create(key=key, bucketizer_factory=_bucketizer_factory)
            if expired:
                KP.inc("phase2.key_evicted.ttl", int(expired))
            if lru_removed:
                KP.inc("phase2.key_evicted.lru", int(lru_removed))

            xvals = _extract_outvals(msg)
            _update_sensor_static(st.user_state, msg)
            if not xvals:
                KP.inc("phase2.skip.no_outvals", 1)
                continue

            ts = _event_ts_epoch(msg)
            bmeta = _extract_bucket_meta(msg)
            fin, late_dropped = st.bucketizer.ingest(event_ts_epoch=ts, values=xvals, meta=bmeta)
            if late_dropped:
                KP.inc("phase2.late_drop", 1)
                continue

            if fin is None:
                KP.inc("phase2.bucket.pending", 1)
                continue

            # Gap reset: clear state-local buffers
            if fin.gap_reset:
                KP.inc("phase2.gap_reset", 1)
                try:
                    rk = str(getattr(fin, "reset_kind", "") or "").strip().lower()
                    if rk == "epoch":
                        KP.inc("phase2.reset.epoch", 1)
                    elif rk == "gap":
                        KP.inc("phase2.reset.gap", 1)
                    elif rk:
                        KP.inc("phase2.reset.other", 1)
                except Exception:
                    pass
                _HIST.pop(key, None)
                st.user_state.pop("seq_by_model", None)
                st.user_state.pop("seq_key_order", None)
                st.user_state.pop("active_seq_key", None)
                st.user_state.pop("artifact", None)

            # Registry-driven scoring (preferred when model exists)
            scored = _try_score_with_registry(
                wsuid=wsuid,
                stock=stock,
                op_tc=op_tc,
                fin_values=fin.values,
                st_user=st.user_state,
            )
            if scored is not None:
                is_anom, score, thr, fam, contrib, importance = scored

                # Patch1: persist bucket-level score to Cassandra
                if PHASE2_PERSIST_ENABLED and (bool(is_anom) or (not PHASE2_PERSIST_ONLY_ANOMALIES)):
                    algo = "LSTM" if ("lstm" in str(fam).lower()) else str(fam)
                    measurement_dt = _measurement_dt_from_finalize(fin)
                    # Build full per-sensor maps for backend compatibility (no truncation)
                    ss_meta = st.user_state.get("sensor_static", {}) if isinstance(st.user_state, dict) else {}
                    sensor_meta_by_sid = {}
                    if isinstance(ss_meta, dict):
                        for _sid in (getattr(fin, "values", {}) or {}).keys():
                            if _sid in ss_meta and isinstance(ss_meta.get(_sid), dict):
                                sensor_meta_by_sid[str(_sid)] = dict(ss_meta.get(_sid))
                    contrib_full = {}
                    try:
                        for _sid in (getattr(fin, "values", {}) or {}).keys():
                            contrib_full[str(_sid)] = float((contrib or {}).get(_sid, 0.0))
                    except Exception:
                        contrib_full = dict(contrib or {})
                    res = _persist_phase2_score(
                        message=(getattr(fin, "meta", None) or msg),
                        measurement_dt=measurement_dt,
                        algorithm=algo,
                        anomaly_score=float(score),
                        anomaly_importance=float(importance),
                        anomaly_threshold=float(thr) if thr is not None else None,
                        is_anomaly=bool(is_anom),
                        bucket_values=dict(getattr(fin, "values", {}) or {}),
                        heatmap_values=dict(contrib_full or {}),
                        sensor_meta_by_sid=sensor_meta_by_sid,
                    )
                    if not res.ok:
                        KP.inc("phase2.persist.err", 1)
                        try:
                            p2_rt_log.warning("[phase2_runtime] persist failed: %s", res.error)
                        except Exception:
                            pass

                if is_anom and PHASE2_TRIGGER_ENABLED:
                    if _trigger_guard(st.user_state, is_anomaly=True):
                        KP.inc("phase2.anomaly_trigger", 1)
                        KP.inc("phase2.trigger.emit", 1)
                        request_anomaly(
                            ws_uid=wsuid,
                            stock_key=stock,
                            scope=PHASE2_TRIGGER_SCOPE,
                            ttl_sec=PHASE2_TRIGGER_TTL,
                            reason=f"{fam} score={score:.6f} thr={thr:.6f} rsec={PHASE2_RESAMPLE_SECONDS}",
                        )
                        p2_rt_log.info(
                            f"[phase2_runtime] anomaly ws={wsuid} stock={stock} fam={fam} score={score:.6f} "
                            f"thr={thr:.6f} rsec={PHASE2_RESAMPLE_SECONDS}"
                        )
                else:
                    _trigger_guard(st.user_state, is_anomaly=False)
                    KP.inc("phase2.infer.ok", 1)

                continue

            # If a model artifact is present+usable but we're still warming up the sequence,
            # do NOT fall back to stub unless explicitly allowed.
            if not PHASE2_STUB_WHEN_MODEL_PRESENT:
                art_info = st.user_state.get("artifact") if isinstance(st.user_state, dict) else None
                if isinstance(art_info, dict) and (art_info.get("load_ok") is True) and (art_info.get("scale_ok") is True):
                    KP.inc("phase2.skip.stub_model_present", 1)
                    KP.inc("phase2.infer.warmup", 1)
                    _trigger_guard(st.user_state, is_anomaly=False)
                    continue

            # Cold-start fallback: stub robust z-score on finalized bucket values
            if not PHASE2_STUB_ENABLED:
                KP.inc("phase2.skip.no_model", 1)
                continue

            top = []  # (absz, sensor)
            for s, v in fin.values.items():
                hist = _HIST[key][s]
                z = _robust_z(float(v), hist)
                hist.append(float(v))
                if z is None:
                    continue
                az = abs(float(z))
                top.append((az, s))

            if not top:
                KP.inc("phase2.skip.no_score", 1)
                continue

            top.sort(reverse=True)
            per_sensor_contrib = {s: float(az) for az, s in top}
            importance = _mean_topk(per_sensor_contrib, int(PHASE2_IMPORTANCE_TOPK))
            if importance is None:
                importance = float(top[0][0])
            score = float(top[0][0])

            # Patch1: persist bucket-level score to Cassandra (stub)
            is_anom_stub = bool(score >= float(PHASE2_Z_TH))
            if PHASE2_PERSIST_ENABLED and (is_anom_stub or (not PHASE2_PERSIST_ONLY_ANOMALIES)):
                measurement_dt = _measurement_dt_from_finalize(fin)
                # Build full per-sensor maps for backend compatibility (no truncation)
                ss_meta = st.user_state.get("sensor_static", {}) if isinstance(st.user_state, dict) else {}
                sensor_meta_by_sid = {}
                if isinstance(ss_meta, dict):
                    for _sid in (getattr(fin, "values", {}) or {}).keys():
                        if _sid in ss_meta and isinstance(ss_meta.get(_sid), dict):
                            sensor_meta_by_sid[str(_sid)] = dict(ss_meta.get(_sid))
                per_sensor_contrib_full = {}
                try:
                    for _sid in (getattr(fin, "values", {}) or {}).keys():
                        per_sensor_contrib_full[str(_sid)] = float((per_sensor_contrib or {}).get(_sid, 0.0))
                except Exception:
                    per_sensor_contrib_full = dict(per_sensor_contrib or {})
                res = _persist_phase2_score(
                    message=(getattr(fin, "meta", None) or msg),
                    measurement_dt=measurement_dt,
                    algorithm="STUB_ZSCORE_BUCKET",
                    anomaly_score=float(score),
                    anomaly_importance=float(importance),
                    anomaly_threshold=float(PHASE2_Z_TH),
                    is_anomaly=bool(is_anom_stub),
                    bucket_values=dict(getattr(fin, "values", {}) or {}),
                    heatmap_values=dict(per_sensor_contrib_full or {}),
                    sensor_meta_by_sid=sensor_meta_by_sid,
                )
                if not res.ok:
                    KP.inc("phase2.persist.err", 1)
                    try:
                        p2_rt_log.warning("[phase2_runtime] persist failed: %s", res.error)
                    except Exception:
                        pass

            if is_anom_stub:
                sensors = ",".join([s for _, s in top[:3]])
                if _trigger_guard(st.user_state, is_anomaly=True):
                    KP.inc("phase2.anomaly_trigger", 1)
                    KP.inc("phase2.trigger.emit", 1)
                    request_anomaly(
                        ws_uid=wsuid,
                        stock_key=stock,
                        scope=PHASE2_TRIGGER_SCOPE,
                        ttl_sec=PHASE2_TRIGGER_TTL,
                        reason=f"stub_zscore_bucket score={score:.2f} top={sensors} rsec={PHASE2_RESAMPLE_SECONDS}",
                    )
                    p2_rt_log.info(
                        f"[phase2_runtime] anomaly ws={wsuid} stock={stock} score={score:.2f} "
                        f"top={sensors} rsec={PHASE2_RESAMPLE_SECONDS}"
                    )
            else:
                _trigger_guard(st.user_state, is_anomaly=False)
                KP.inc("phase2.infer.ok", 1)

        except Exception as e:
            KP.inc("phase2.error", 1)
            p2_rt_log.error(f"[phase2_runtime] error: {e}", exc_info=True)
        finally:
            task_done_phase2()
