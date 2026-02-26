import json
import threading
import time
import uuid
import re
import hashlib
from datetime import datetime, timezone, timedelta
from collections import defaultdict, OrderedDict


# Cassandra driver is optional in compute-only mode.
try:
    from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy, TokenAwarePolicy
    from cassandra import ConsistencyLevel
    _CASSANDRA_DRIVER_OK = True
    _CASSANDRA_DRIVER_ERR = None
except Exception as _e:  # pragma: no cover
    Cluster = None
    ExecutionProfile = None
    EXEC_PROFILE_DEFAULT = None
    PlainTextAuthProvider = None
    DCAwareRoundRobinPolicy = None
    RetryPolicy = None
    TokenAwarePolicy = None
    ConsistencyLevel = None
    _CASSANDRA_DRIVER_OK = False
    _CASSANDRA_DRIVER_ERR = _e

from utils.config_reader import ConfigReader
from runtime.event_bus import dequeue_phase3, task_done_phase3
from utils.identity import get_workstation_uid, get_stock_key
from utils.logger_2 import setup_logger
from utils.keypoint_recorder import KP

# Helper functions for correlation matrix
from thread.phase_3_correlation._3_1_helper_functions import map_to_text, compute_correlation, _is_input_type, aggregate_correlation_data

# Predictions / Feature importance imports are done lazily to avoid hard dependency in compute-only mode.

from thread.phase_2_multivariate_lstm_pipeline.phase2_runtime import phase2_enqueue
from thread.phase_trigger_bus import peek_anomaly, consume_anomaly, baseline_due, mark_baseline_run


# Batching and planning
from modules.batching.batch_assigner import BatchAssigner
from modules.planning.training_planner import TrainingPlanner



consumer3 = None
NONE_LIMIT = 100   # 30 defa üst üste None → ~30 saniye

cfg = ConfigReader()
cassandra_config = cfg["cassandra"]

PHASE3_DW_SNAPSHOT_ENABLED = bool(getattr(cfg, "phase3_dw_snapshot_enabled", False))
CASSANDRA_HOST = cassandra_config["host"]
USERNAME = cassandra_config["username"]
PASSWORD = cassandra_config["password"]
KEYSPACE = cassandra_config["keyspace"]


# Lazily-initialized Cassandra session (used only for RAW-table fallback reads).
# In compute-only mode, the driver may be absent; we degrade to buffer-only paths.
cluster = None
session = None

def _get_cassandra_session(p3_1_log=None):
    global cluster, session
    if session is not None:
        return session
    if not _CASSANDRA_DRIVER_OK:
        if p3_1_log:
            p3_1_log.warning(
                f"[phase3] Cassandra driver unavailable; RAW-table fallback disabled: {_CASSANDRA_DRIVER_ERR}"
            )
        return None

    try:
        auth_provider = (
            PlainTextAuthProvider(username=USERNAME, password=PASSWORD)
            if USERNAME and PASSWORD
            else None
        )

        profile = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy(local_dc=None)),
            request_timeout=20.0,
            consistency_level=ConsistencyLevel.LOCAL_ONE,
            retry_policy=RetryPolicy(),
        )

        cluster = Cluster(
            [CASSANDRA_HOST],
            auth_provider=auth_provider,
            execution_profiles={EXEC_PROFILE_DEFAULT: profile},
        )
        session = cluster.connect(KEYSPACE)
        session.set_keyspace(KEYSPACE)
        return session
    except Exception as e:
        if p3_1_log:
            p3_1_log.warning(
                f"[phase3] Cassandra session init failed; RAW-table fallback disabled: {e}",
                exc_info=True,
            )
        return None

def _try_fetch_dw_snapshot(limit: int = 20000, p3_1_log=None):
    """DW snapshot fetch via cqlengine model (optional)."""
    try:
        from cassandra_utils.models.dw_single_data import dw_tbl_raw_data
        return dw_tbl_raw_data.fetchData(limit=limit)
    except Exception as e:
        if p3_1_log:
            p3_1_log.warning(
                f"[phase3] DW snapshot fetch unavailable (continuing without snapshot): {e}"
            )
        return [], [], [], []


# Batching Globals
M1_LOG_BATCH_CONTEXT = bool(getattr(cfg, "m1_log_batch_context", True))
batch_assigner = BatchAssigner(
    session_gap_sec=int(getattr(cfg, "m1_session_gap_sec", 900)),
    explicit_gap_split_sec=int(getattr(cfg, "m1_explicit_gap_split_sec", 3600)),
    enable_hybrid_split=bool(getattr(cfg, "m1_enable_hybrid_split", False)),
)

# Planning Globals
M1_ENABLE_TRAINING_PLANNER = bool(getattr(cfg, "m1_enable_training_planner", True))
training_planner = TrainingPlanner(
    session_gap_sec=int(getattr(cfg, "m1_session_gap_sec", 900)),
    min_segments=int(getattr(cfg, "m1_min_segments", 5)),
    min_points_per_segment=int(getattr(cfg, "m1_min_points_per_segment", 200)),
    log_every_sec=int(getattr(cfg, "m1_planner_log_every_sec", 60)),
    keep_last_n=int(getattr(cfg, "m1_planner_keep_last_n", 200)),
)

# ---- M1.6: gated batch-aware PID buffers ----
M1_ENABLE_BATCH_BUFFERS = bool(getattr(cfg, "m1_enable_batch_buffers", False))
M1_BATCH_MIN_CONFIDENCE = float(getattr(cfg, "m1_batch_min_confidence", 0.8))
M1_BATCH_ALLOW_SESSION  = bool(getattr(cfg, "m1_batch_allow_session", False))

M1_MAX_ACTIVE_BATCHES_PER_WS = int(getattr(cfg, "m1_max_active_batches_per_ws", 50))
M1_BATCH_TRACK_TTL_SEC = int(getattr(cfg, "m1_batch_track_ttl_sec", 3600))
M1_BATCH_GUARD_LOG_EVERY_SEC = int(getattr(cfg, "m1_batch_guard_log_every_sec", 60))

_ACTIVE_BATCHES_BY_WS = defaultdict(OrderedDict)  # ws_uid -> OrderedDict(batch_id -> last_seen_epoch)
_LAST_BATCH_GUARD_LOG = {}

PHASE3_RAW_PERSIST_ENABLED = bool(getattr(cfg, "phase3_raw_persist_enabled", True))

def _pid_buffer_key(message: dict, pid):
    """
    Legacy: pid
    Batch-aware: ws_uid|batch_id|pid (only if confidence high)
    Guarded: caps active batch_ids per ws_uid to avoid memory/key explosion
    """
    if not M1_ENABLE_BATCH_BUFFERS:
        return pid

    ws_uid = message.get("_workstation_uid")
    batch_id = message.get("_batch_id")
    strat = message.get("_batch_strategy")
    conf = float(message.get("_batch_confidence") or 0.0)

    if not ws_uid or not batch_id:
        return pid
    if conf < M1_BATCH_MIN_CONFIDENCE:
        return pid
    if (not M1_BATCH_ALLOW_SESSION) and strat == "SESSION":
        return pid

    # ---- Guard: cap active batches per workstation ----
    if M1_MAX_ACTIVE_BATCHES_PER_WS > 0:
        now = time.time()
        od = _ACTIVE_BATCHES_BY_WS[ws_uid]

        # TTL cleanup
        if M1_BATCH_TRACK_TTL_SEC > 0 and od:
            for b in list(od.keys()):
                if (now - float(od[b] or 0.0)) > M1_BATCH_TRACK_TTL_SEC:
                    od.pop(b, None)

        if batch_id in od:
            od[batch_id] = now
            od.move_to_end(batch_id)
        else:
            if len(od) >= M1_MAX_ACTIVE_BATCHES_PER_WS:
                last = _LAST_BATCH_GUARD_LOG.get(ws_uid, 0.0)
                if (now - last) >= M1_BATCH_GUARD_LOG_EVERY_SEC:
                    _LAST_BATCH_GUARD_LOG[ws_uid] = now
                    p3_1_log.warning(
                        f"[M1.6] batch_guard ws={ws_uid} active_batches={len(od)} "
                        f"limit={M1_MAX_ACTIVE_BATCHES_PER_WS} rejecting_new_batch={batch_id} "
                        f"(fallback to legacy pid key)"
                    )
                return pid

            od[batch_id] = now

    return f"{ws_uid}|{batch_id}|{pid}"

# ---- ID normalization for backend-friendly placeholders ----
_NULL_ID_TOKENS = {"", "0", "none", "null", "nan", "n/a", "unknown", "unk"}

def _norm_id_or_zero_str(v) -> str:
    """
    Persist identifiers as strings. Unknown/missing => "0".
    Also collapses composite placeholders like "null||null", "0||0", "null|null" => "0".
    """
    if v is None:
        return "0"
    s = str(v).strip()
    if not s:
        return "0"

    sl = s.lower().strip()
    if sl in _NULL_ID_TOKENS:
        return "0"

    # composite placeholders
    if "|" in sl:
        parts = [p.strip() for p in re.split(r"\|+", sl) if p.strip() != ""]
        if parts and all(p in _NULL_ID_TOKENS for p in parts):
            return "0"

    return s


# ---- M1.7: prediction seed cache (bounded TTL/LRU) ----
M1_SEED_CACHE_ENABLED      = bool(getattr(cfg, "m1_seed_cache_enabled", True))
M1_SEED_CACHE_TTL_SEC      = int(getattr(cfg, "m1_seed_cache_ttl_sec", 180))
M1_SEED_CACHE_MAX_ENTRIES  = int(getattr(cfg, "m1_seed_cache_max_entries", 32))
M1_SEED_CACHE_MAX_POINTS   = int(getattr(cfg, "m1_seed_cache_max_points", 600))

_SEED_CACHE = OrderedDict()   # key -> (ts_epoch, seed_list)
_SEED_CACHE_LOCK = threading.Lock()

def _seed_cache_key(message: dict, op_tc: str, stock_no: str) -> str:
    cust = message.get("customer") or message.get("cust") or "UNKNOWN_CUSTOMER"
    ws_uid = message.get("_workstation_uid") or f"WSID_{message.get('wsId')}"
    batch_id = message.get("_batch_id")
    strat = message.get("_batch_strategy")
    conf = float(message.get("_batch_confidence") or 0.0)

    # Prefer batch-aware cache scope when M1.6 enabled and confidence high
    if M1_ENABLE_BATCH_BUFFERS and batch_id and conf >= M1_BATCH_MIN_CONFIDENCE and (M1_BATCH_ALLOW_SESSION or strat != "SESSION"):
        return f"{cust}|{ws_uid}|{batch_id}|{op_tc}|{stock_no}"

    return f"{cust}|{ws_uid}|{op_tc}|{stock_no}"

def _seed_cache_get(k: str):
    if not M1_SEED_CACHE_ENABLED:
        return None
    now = time.time()
    with _SEED_CACHE_LOCK:
        ent = _SEED_CACHE.get(k)
        if not ent:
            return None
        ts0, seed = ent
        if (now - ts0) > M1_SEED_CACHE_TTL_SEC:
            _SEED_CACHE.pop(k, None)
            return None
        _SEED_CACHE.move_to_end(k)
        return seed

def _seed_cache_put(k: str, seed_list):
    if not M1_SEED_CACHE_ENABLED:
        return
    if not seed_list:   # empty list / None
        return          # don't cache empties
    
    now = time.time()
    # keep only last N rows (prediction needs min_train_points, not 20k)
    if isinstance(seed_list, list) and len(seed_list) > M1_SEED_CACHE_MAX_POINTS:
        seed_list = seed_list[-M1_SEED_CACHE_MAX_POINTS:]

    with _SEED_CACHE_LOCK:
        _SEED_CACHE[k] = (now, seed_list)
        _SEED_CACHE.move_to_end(k)
        while len(_SEED_CACHE) > M1_SEED_CACHE_MAX_ENTRIES:
            _SEED_CACHE.popitem(last=False)

# ---- M1.8: Phase-3 gating + periodic metrics ----
def _phase3_parse_allowlist(v):
    """
    Accepts:
      - list[str]
      - comma-separated string
      - None/empty
    Returns: set[str] or None
    """
    if v is None:
        return None
    if isinstance(v, list):
        s = {str(x).strip() for x in v if str(x).strip()}
        return s or None
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        return set(parts) or None
    s = str(v).strip()
    return {s} if s else None


PHASE3_MODE = str(getattr(cfg, "phase3_mode", "full") or "full").strip().lower()
PHASE3_HEAVY_EVERY_SEC = int(getattr(cfg, "phase3_heavy_every_sec", 0) or 0)
PHASE3_MODEB_BASELINE_EVERY_SEC = int(getattr(cfg, "phase3_modeb_baseline_every_sec", PHASE3_HEAVY_EVERY_SEC) or 0)
PHASE3_MODEB_BASELINE_JITTER_SEC = int(getattr(cfg, "phase3_modeb_baseline_jitter_sec", 0) or 0)
PHASE3_MODEB_BASELINE_MAX_RUNS_PER_HOUR = int(getattr(cfg, "phase3_modeb_baseline_max_runs_per_hour", 0) or 0)

PHASE3_HEAVY_KEY_SCOPE = str(getattr(cfg, "phase3_heavy_key_scope", "ws") or "ws").strip().lower()

PHASE3_MODE_B_WS_ONLY = bool(getattr(cfg, "phase3_mode_b_ws_only", True))

PHASE3_WORKSTATION_ALLOWLIST = _phase3_parse_allowlist(getattr(cfg, "phase3_workstation_allowlist", None))
PHASE3_SKIP_LOG_EVERY_SEC = int(getattr(cfg, "phase3_skip_log_every_sec", 60) or 60)

_PHASE3_LAST_HEAVY_RUN = {}   # key -> last_epoch
_PHASE3_LAST_SKIP_LOG = {}    # ws_uid -> last_epoch


PHASE3_METRICS_ENABLED = bool(getattr(cfg, "phase3_metrics_enabled", True))
PHASE3_METRICS_LOG_INTERVAL_SEC = int(getattr(cfg, "phase3_metrics_log_interval_sec", 60) or 60)

_METRICS = defaultdict(int)
_METRICS_LAST_TS = time.time()
_METRICS_LAST_COUNTS = {}


def _m_inc(k: str, n: int = 1):
    _METRICS[k] += n
    # Mirror into global keypoint recorder (coarse observability across stages)
    try:
        KP.inc(f"phase3.{k}", int(n))
    except Exception:
        pass


def _phase3_get_ws_uid(message: dict) -> str:
    # Canonical workstation identity (must match Stage0 + registry artifact naming)
    try:
        return get_workstation_uid(message)
    except Exception:
        return "WSUID_UNKNOWN"





# ---- Mode B baseline helpers (deterministic jitter + optional rate limiting) ----
_MODEB_BASELINE_RUNS = defaultdict(list)   # key -> list[timestamps]
_MODEB_BASELINE_LOCK = threading.Lock()

def _stable_mod(s: str, mod: int) -> int:
    if mod <= 0:
        return 0
    h = hashlib.md5(s.encode('utf-8')).hexdigest()[:8]
    return int(h, 16) % mod

def _modeb_effective_baseline_every(ws_uid: str, stock_key: str) -> int:
    every = int(PHASE3_MODEB_BASELINE_EVERY_SEC or 0)
    if every <= 0:
        return 0
    jitter = int(PHASE3_MODEB_BASELINE_JITTER_SEC or 0)
    if jitter > 0:
        off = _stable_mod(f"{ws_uid}|{stock_key}|ws", jitter + 1)
        every = every + off
    return every

def _modeb_baseline_rate_ok(ws_uid: str, stock_key: str, now_epoch: float) -> bool:
    maxh = int(PHASE3_MODEB_BASELINE_MAX_RUNS_PER_HOUR or 0)
    if maxh <= 0:
        return True
    key = f"{ws_uid}|{stock_key}|ws"
    with _MODEB_BASELINE_LOCK:
        lst = _MODEB_BASELINE_RUNS.get(key) or []
        # prune older than 1 hour
        cutoff = float(now_epoch) - 3600.0
        if lst:
            lst = [t for t in lst if float(t) >= cutoff]
        if len(lst) >= maxh:
            _MODEB_BASELINE_RUNS[key] = lst
            return False
        lst.append(float(now_epoch))
        _MODEB_BASELINE_RUNS[key] = lst
        return True
def _phase3_should_run_heavy(message: dict, *, scope: str, scope_id, now_epoch: float) -> bool:
    """
    Heavy work = Cassandra fetch + prediction/correlation/feature importance.

    Modes:
      - full: always
      - persist_only: never
      - sampled_full: periodic throttle (legacy)
      - mode_b: Phase2 always; Phase3 heavy runs only on (anomaly trigger) OR (periodic baseline)
              WS-scope only when PHASE3_MODE_B_WS_ONLY=True
    """
    mode = PHASE3_MODE

    if mode == "persist_only":
        _m_inc("heavy_skip_reason_persist_only", 1)
        return False

    if mode == "full":
        return True

    # --- Mode B (triggered + periodic baseline) ---
    if mode == "mode_b":
        if scope == "pid" and PHASE3_MODE_B_WS_ONLY:
            _m_inc("heavy_skip_reason_ws_only", 1)
            return False

        ws_uid = _phase3_get_ws_uid(message)
        stock_key = _clean_stock(get_stock_key(message, default="")) or "ALL"

        anom_ok = peek_anomaly(ws_uid, stock_key, scope="ws")
        if (not anom_ok) and stock_key != "ALL":
            anom_ok = peek_anomaly(ws_uid, "ALL", scope="ws")

        base_ok = False
        eff_every = _modeb_effective_baseline_every(ws_uid, stock_key)
        if eff_every > 0:
            base_ok = baseline_due(
                ws_uid,
                stock_key,
                scope="ws",
                every_sec=eff_every,
                now_epoch=now_epoch,
            )
            if base_ok and (not _modeb_baseline_rate_ok(ws_uid, stock_key, now_epoch)):
                base_ok = False
                _m_inc("modeb_skip_baseline_ratelimit", 1)
                _m_inc("heavy_skip_reason_baseline_ratelimit", 1)

        if anom_ok:
            _m_inc("modeb_allow_anom", 1)
            return True
        if base_ok:
            _m_inc("modeb_allow_baseline", 1)
            return True

        _m_inc("modeb_skip_no_trigger", 1)
        _m_inc("heavy_skip_reason_no_trigger", 1)
        return False

    # --- sampled_full (legacy periodic throttle) ---
    if mode != "sampled_full":
        # unknown -> be safe
        _m_inc("heavy_skip_reason_unknown_mode", 1)
        return False

    if PHASE3_HEAVY_EVERY_SEC <= 0:
        return True

    ws_uid = _phase3_get_ws_uid(message)

    if PHASE3_HEAVY_KEY_SCOPE == "scope_id":
        k = f"{ws_uid}|{scope}|{scope_id}"
    else:
        k = f"{ws_uid}|{scope}"

    last = _PHASE3_LAST_HEAVY_RUN.get(k, 0.0)
    if (now_epoch - last) >= PHASE3_HEAVY_EVERY_SEC:
        _PHASE3_LAST_HEAVY_RUN[k] = now_epoch
        return True

    _m_inc("heavy_skip_reason_cooldown", 1)
    return False



def _maybe_log_metrics(p3_1_log, now_epoch: float):
    global _METRICS_LAST_TS, _METRICS_LAST_COUNTS

    if not PHASE3_METRICS_ENABLED:
        return
    if PHASE3_METRICS_LOG_INTERVAL_SEC <= 0:
        return

    dt = now_epoch - _METRICS_LAST_TS
    if dt < PHASE3_METRICS_LOG_INTERVAL_SEC:
        return

    # delta snapshot
    delta = {}
    for k, v in _METRICS.items():
        prev = _METRICS_LAST_COUNTS.get(k, 0)
        delta[k] = v - prev

    # basic rates
    msgs = delta.get("kafka_msg_parsed", 0)
    raw_ok = delta.get("raw_write_ok", 0)
    heavy_pid = delta.get("heavy_pid_run", 0)
    heavy_ws = delta.get("heavy_ws_run", 0)

    p3_1_log.info(
        "[phase3_metrics] "
        f"dt={dt:.1f}s "
        f"msg={msgs} ({msgs/dt:.2f}/s) "
        f"raw_ok={raw_ok} ({raw_ok/dt:.2f}/s) "
        f"commit_ok={delta.get('kafka_commit_ok', 0)} "
        f"commit_fail={delta.get('kafka_commit_fail', 0)} "
        f"heavy_pid_run={heavy_pid} heavy_pid_skip={delta.get('heavy_pid_skip', 0)} "
        f"heavy_ws_run={heavy_ws} heavy_ws_skip={delta.get('heavy_ws_skip', 0)} "
        f"raw_fail={delta.get('raw_write_fail', 0)} "
        f"errors_lvl2={delta.get('err_lvl2', 0)} "
        f"persist_only={delta.get('persist_only_msg', 0)} "
        f"allowlist_skip={delta.get('allowlist_skip', 0)} "
        f"seed_hit={delta.get('seed_cache_hit', 0)} seed_miss={delta.get('seed_cache_miss', 0)} "
        f"pred_ok={delta.get('pred_ok', 0)} pred_fail={delta.get('pred_fail', 0)} "
        f"pred_skip_cd={delta.get('pred_skip_cooldown', 0)} pred_skip_no_outputs={delta.get('pred_skip_no_outputs', 0)} "
        f"corr_ok={delta.get('corr_ok', 0)} corr_fail={delta.get('corr_fail', 0)} corr_skip_cd={delta.get('corr_skip_cooldown', 0)} "
        f"corr_skip_buf_pid={delta.get('corr_skip_buf_pid', 0)} corr_skip_buf_ws={delta.get('corr_skip_buf_ws', 0)} "
        f"fi_ok={delta.get('fi_ok', 0)} fi_fail={delta.get('fi_fail', 0)} fi_skip_cd={delta.get('fi_skip_cooldown', 0)} "

    )

    _METRICS_LAST_TS = now_epoch
    _METRICS_LAST_COUNTS = dict(_METRICS)


# Lock mechanism to ensure one message is completely processed at a time
processing_lock = threading.Lock()

buffer_for_process = {}
buffer_for_ws = {}

PID_COUNT_THRESHOLD = int(getattr(cfg, "phase3_pid_count_threshold", 20) or 20)
PID_TIME_THRESHOLD  = int(getattr(cfg, "phase3_pid_time_threshold_sec", 900) or 900)

WS_COUNT_THRESHOLD  = int(getattr(cfg, "phase3_ws_count_threshold", 20) or 20)
WS_TIME_THRESHOLD   = int(getattr(cfg, "phase3_ws_time_threshold_sec", 900) or 900)

_LAST_RUN_TS = {
    "correlation": {},
    "feature_importance": {},
    "realtime_prediction": {},
}

def _cooldown_ok(task: str, key: str, cooldown_sec: int) -> bool:
    if cooldown_sec is None:
        return True
    cooldown_sec = int(cooldown_sec)
    if cooldown_sec <= 0:
        return True
    now = time.time()
    last = _LAST_RUN_TS[task].get(key, 0.0)
    if (now - last) >= cooldown_sec:
        _LAST_RUN_TS[task][key] = now
        return True
    return False


def _should_fire(buffer_entry, min_points, max_interval_sec, now_epoch=None):
    """
    Decide whether a buffer has enough points or time to trigger processing.

    Returns: (fire_bool, count, time_lapsed)
    """
    if now_epoch is None:
        now_epoch = time.time()

    cnt = int(buffer_entry.get("count") or 0)

    first_seen = buffer_entry.get("first_seen")
    if first_seen is None:
        first_seen = now_epoch
        buffer_entry["first_seen"] = first_seen

    time_lapsed = max(0.0, float(now_epoch) - float(first_seen))

    min_points = int(min_points or 0)
    max_interval_sec = float(max_interval_sec or 0)

    fire = (cnt >= min_points) or (max_interval_sec > 0 and time_lapsed >= max_interval_sec)
    return fire, cnt, time_lapsed


p3_1_log = setup_logger(
    "p3_1_logger", "logs/p3_1.log"
)

RAW_TABLE = cfg["cassandra_props"]["raw_data_table"]  # "dw_tbl_raw_data"


# -------- Utilities -------- #
def _to_dt(v):
    """Convert to aware UTC datetime.

    Destekler:
    - datetime objesi
    - ISO string (2025-11-28T13:00:00Z, 2025-11-28 13:00:00, vs.)
    - epoch saniye (int/float)
    - epoch milisaniye (int/float veya numeric string)
    """
    if v is None:
        return None

    # Zaten datetime ise
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    # Sayısal epoch (ms veya s)
    try:
        s = int(v)
        # çok kaba ama pratik: 10^12'den büyükse ms kabul et
        if s > 10**12:
            return datetime.fromtimestamp(s / 1000.0, tz=timezone.utc)
        else:
            return datetime.fromtimestamp(s, tz=timezone.utc)
    except Exception:
        pass

    # ISO string formatı
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _to_epoch_ms_safe(dt):
    if not dt:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _num_text(v):
    if v is None:
        return "0"
    s = str(v).strip().replace(",", ".")
    try:
        float(s)
        return s
    except Exception:
        return "0"


def _map_to_text(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        elif isinstance(v, (int, float)):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()
        else:
            out[k] = "" if v is None else str(v)
    return out

def _combine_input_output_lists(input_list, output_list):
    """
    Combine input_list and output_list into wrapped format for compatibility.
    
    Args:
        input_list: List of input variable dicts from fetchData()
        output_list: List of [output_dict] from fetchData()
    
    Returns:
        combined_list: List of [combined_dict] where each dict has:
            - equipment_type: True for inputs, False for outputs
            - All relevant fields (eqNo/varNm, cntRead/genReadVal, etc.)
    """
    combined = []
    
    # Add outputs (already in [[dict]] format)
    for ov_wrapped in output_list:
        if not ov_wrapped or not isinstance(ov_wrapped, list):
            continue
        ov = ov_wrapped[0] if len(ov_wrapped) > 0 else {}
        if not isinstance(ov, dict):
            continue
        
        # Mark as OUTPUT
        ov_copy = dict(ov)
        ov_copy["equipment_type"] = False
        combined.append([ov_copy])
    
    # Add inputs (need to wrap in [])
    for iv in input_list:
        if not isinstance(iv, dict):
            continue
        
        # Convert input format to match output format
        iv_wrapped = {
            "equipment_type": True,  # Mark as INPUT
            "eqNm": iv.get("varNm"),  # varNm → eqNm
            "eqNo": iv.get("varNm") or iv.get("varNo"),  # varNo → eqNo
            "gen_read_val": iv.get("genReadVal"),  # genReadVal
            "joOpId": iv.get("joOpId"),
            "wsId": iv.get("wsId"),
            "good": iv.get("good"),
        }
        combined.append([iv_wrapped])
    
    return combined
# -------- DW'den fetch (PID-level, snapshot üzerinden) -------- #
# ---------- STOCK HELPERS (NEW) ----------

def _extract_stock_from_prodlist_message(prod_list):
    """
    prod_list: message['prodList'] (list of dict)
    Returns: (output_stock_no, output_stock_name)
    """
    if not prod_list or not isinstance(prod_list, list):
        return None, None

    for item in prod_list:
        if not isinstance(item, dict):
            continue
        st_no = item.get("stNo") or item.get("stockNo") or item.get("st_no")
        st_nm = item.get("stNm") or item.get("stockName") or item.get("st_name")
        if st_no not in (None, ""):
            return str(st_no), (str(st_nm) if st_nm is not None else None)

    return None, None


# -------- DW'den fetch (PID-level, snapshot üzerinden) -------- #
def fetch_latest_for_pid_via_dw(pid: int, rows, input_list, output_list, _batch, max_rows: int = 20000):
    """
    DW snapshot içinden job_order_operation_id == pid olan son ölçümleri al.
    Her zaman dilimi (ts) için:
      - sensörler: parameter / counter_reading / equipment_name
      - label/meta: good (boolean), prSt (workstation_state), crDt, stock info
    """
    # ts -> list of sensor dicts
    bucket = defaultdict(list)
    # ts -> meta (good, prSt, refs, stock vs.)
    label_meta = {}

    # Combine input and output lists
    combined_list = _combine_input_output_lists(input_list, output_list)

    for r, wrapped_ov in zip(rows, combined_list):
        if getattr(r, "job_order_operation_id", None) != pid:
            continue

        ts = getattr(r, "measurement_date", None)
        if ts is None:
            p3_1_log.debug(f"[fetch_latest_for_pid_via_dw] pid={pid}: skipping row with null measurement_date")
            continue

        jo_ref_val = getattr(r, "job_order_reference_no", None)
        prod_ref_val = getattr(r, "prod_order_reference_no", None)

        # stok bilgisi DW'deki producelist kolonundan
        #produce_list = getattr(r, "producelist", None)
        #st_no, st_nm = _extract_stock_from_producelist(produce_list)
        st_no, st_nm = getattr(r, "produced_stock_no", None), getattr(r, "produced_stock_name", None)

        # ----------------- LABEL / META KISMI -----------------
        good_val = getattr(r, "good", None)
        prst_val = getattr(r, "workstation_state", None)

        # Aynı ts için meta bir kez set edilsin; sonra tekrar yazmasın
        if ts not in label_meta:
            label_meta[ts] = {
                "good": good_val,
                "prSt": prst_val,
                "job_order_reference_no": jo_ref_val,
                "prod_order_reference_no": prod_ref_val,
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                # ---- OP FIELDS (NEW) ----
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # ----------------- SENSÖR KISMI -----------------
        ov = wrapped_ov[0] if wrapped_ov else {}
        if ov.get("equipment_type", True):  # INPUT ise atla
            pname = ov.get("varNo")
            cval = ov.get("genReadVal")
            eq_name = ov.get("varNm")
            eq_type = ov.get("equipment_type", True)

            bucket[ts].append({
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })
        else:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            eq_type = ov.get("equipment_type", False)

            bucket[ts].append({
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })

    if not bucket:
        p3_1_log.info(f"[fetch_latest_for_pid_via_dw] pid={pid}: 0 rows after filtering")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        # Bu ts için daha önce doldurduğumuz label/meta
        lm = label_meta.get(ts, {}) or {}
        good_val = lm.get("good")
        prst_val = lm.get("PrSt") if "PrSt" in lm else lm.get("prSt")  # güvenlik
        prst_val = lm.get("prSt") if prst_val is None else prst_val

        meta = {
            "job_order_operation_id": pid,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            # ---- label alanları: feature importance için lazım ----
            "good": good_val,          # dw_tbl_raw_data.good
            "prSt": prst_val,          # dw_tbl_raw_data.workstation_state (PRODUCTION vs)
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            # stok meta
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            # ---- OP FIELDS (NEW) ----
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(pid)
        dates.append(ts)
        ids.append(uuid.uuid4())
        # meta + sensörler -> hepsi text’e çevrilerek sensor_values’a gidiyor
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    try:
        sample_meta = sensor_values[0][0] if sensor_values and len(sensor_values[0]) > 0 else None
        sample_sensor = sensor_values[0][1] if sensor_values and len(sensor_values[0]) > 1 else None
        p3_1_log.debug(
            f"[fetch_latest_for_pid_via_dw] sample meta keys="
            f"{list(sample_meta.keys()) if isinstance(sample_meta, dict) else type(sample_meta)}, "
            f"sample sensor keys="
            f"{list(sample_sensor.keys()) if isinstance(sample_sensor, dict) else type(sample_sensor)}"
        )
    except Exception:
        pass

    p3_1_log.info(f"[fetch_latest_for_pid_via_dw] pid={pid} bundles={len(sensor_values)} (timestamps)")
    return job_ids, dates, ids, sensor_values


# -------- DW'den fetch (WS-level, snapshot üzerinden) -------- #
def fetch_latest_for_ws_via_dw(ws_id: int, rows, input_list, output_list, _batch, max_rows: int = 20000):
    bucket = defaultdict(list)   # key = (ts, stock_no) -> sensors
    label_meta = {}              # key = (ts, stock_no) -> meta

    combined_list = _combine_input_output_lists(input_list, output_list)

    for r, wrapped_ov in zip(rows, combined_list):
        ws_attr = (
            getattr(r, "work_station_id", None)
            or getattr(r, "workstation_id", None)
            or getattr(r, "ws_id", None)
            or getattr(r, "wsid", None)
        )
        if ws_attr != ws_id:
            continue

        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        # IMPORTANT: stock for this row
        st_no = getattr(r, "produced_stock_no", None)
        st_nm = getattr(r, "produced_stock_name", None)

        stock_key = str(st_no) if st_no not in (None, "") else None
        key = (ts, stock_key)

        # label/meta per (ts, stock)
        if key not in label_meta:
            label_meta[key] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": stock_key,
                "output_stock_name": (str(st_nm) if st_nm not in (None, "") else None),
                # ---- OP FIELDS (NEW) ----
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # sensor part
        ov = wrapped_ov[0] if wrapped_ov else {}
        if ov.get("equipment_type", True):  # INPUT ise atla
            pname = ov.get("varNo")
            cval = ov.get("genReadVal")
            eq_name = ov.get("varNm")
            eq_type = ov.get("equipment_type", True)

            bucket[key].append({  # ← FIXED: Use key (ts, stock_key)
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })
        else:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            eq_type = ov.get("equipment_type", False)

            bucket[key].append({  # ← FIXED: Use key (ts, stock_key)
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })

    if not bucket:
        p3_1_log.info(f"[fetch_latest_for_ws_via_dw] ws_id={ws_id}: 0 rows after filtering")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    # sort by ts (and stock_key for deterministic)
    for (ts, stock_key) in sorted(bucket.keys(), key=lambda x: (x[0], str(x[1]))):
        sensors = bucket[(ts, stock_key)]
        if not sensors:
            continue

        lm = label_meta.get((ts, stock_key), {}) or {}
        meta = {
            "workstation_id": ws_id,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            # ---- OP FIELDS (NEW) ----
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(f"[fetch_latest_for_ws_via_dw] ws_id={ws_id} bundles={len(sensor_values)} (ts,stock)")
    return job_ids, dates, ids, sensor_values


# -------- RAW_TABLE'dan direkt fetch (PID-level, eski timestamp fallback) -------- #
def fetch_latest_for_pid_via_raw_table(pid: int, max_rows: int = 20000):
    """
    UPDATED: Handles BOTH inputs and outputs from RAW_TABLE.
    """
    p3_1_log.info(
        f"[fetch_latest_for_pid_via_raw_table] Fallback querying RAW_TABLE={RAW_TABLE} for pid={pid}"
    )

    query = (
        f"SELECT job_order_operation_id, work_station_id, measurement_date, "
        f"equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type, "
        f"good, workstation_state, job_order_reference_no, prod_order_reference_no, "
        f"produced_stock_no, produced_stock_name, operationname, operationno, operationtaskcode "
        f"FROM {RAW_TABLE} "
        f"WHERE job_order_operation_id = %s "
        f"LIMIT %s ALLOW FILTERING"
    )

    try:
        sess = _get_cassandra_session(p3_1_log=p3_1_log)
        if sess is None:
            return [], [], [], []
        rows = sess.execute(query, (pid, max_rows))
    except Exception as e:
        p3_1_log.warning(
            f"[fetch_latest_for_pid_via_raw_table] CQL query failed for pid={pid}: {e}"
        )
        return [], [], [], []

    bucket = defaultdict(list)
    label_meta = {}

    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        st_no, st_nm = getattr(r, "produced_stock_no", None), getattr(r, "produced_stock_name", None)

        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # ===== NEW: Check equipment_type =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type) #(equipment_type is True)
        
        if is_input:
            # INPUT
            var_name = getattr(r, "equipment_name", None) or getattr(r, "parameter", None)
            var_value = getattr(r, "gen_read_val", None)
            
            if not var_name or var_value is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(var_name)}",
                "counter_reading": _num_text(var_value),
                "equipment_name": str(var_name),
                "equipment_type": True,
            })
        else:
            # OUTPUT
            pname = getattr(r, "parameter", None) or getattr(r, "equipment_no", None)
            cval = getattr(r, "counter_reading", None)
            eq_name = getattr(r, "equipment_name", None)
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })

    if not bucket:
        p3_1_log.info(
            f"[fetch_latest_for_pid_via_raw_table] pid={pid}: 0 rows found in RAW_TABLE fallback"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "job_order_operation_id": pid,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(pid)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[fetch_latest_for_pid_via_raw_table] pid={pid}: built {len(sensor_values)} bundles from RAW_TABLE"
    )
    return job_ids, dates, ids, sensor_values


# -------- RAW_TABLE'dan direkt fetch (WS-level, eski timestamp fallback) -------- #
def fetch_latest_for_ws_via_raw_table(ws_id: int, max_rows: int = 20000):
    """
    UPDATED: Handles BOTH inputs and outputs from RAW_TABLE.
    """
    p3_1_log.info(
        f"[fetch_latest_for_ws_via_raw_table] Fallback querying RAW_TABLE={RAW_TABLE} for ws_id={ws_id}"
    )

    query = (
        f"SELECT job_order_operation_id, work_station_id, measurement_date, "
        f"equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type, "
        f"good, workstation_state, job_order_reference_no, prod_order_reference_no, "
        f"produced_stock_no, produced_stock_name, operationname, operationno, operationtaskcode "
        f"FROM {RAW_TABLE} "
        f"WHERE work_station_id = %s "
        f"LIMIT %s ALLOW FILTERING"
    )

    try:
        # IMPORTANT: materialize inside try so paging/network errors are caught here
        sess = _get_cassandra_session(p3_1_log=p3_1_log)
        if sess is None:
            return [], [], [], []
        rows = list(sess.execute(query, (ws_id, max_rows)))
    except Exception as e:
        p3_1_log.warning(
            f"[fetch_latest_for_ws_via_raw_table] CQL query failed for ws_id={ws_id}: {e}"
        )
        return [], [], [], []

    bucket = defaultdict(list)
    label_meta = {}

    for r in rows:
        ws_attr = getattr(r, "work_station_id", None)
        if ws_attr != ws_id:
            continue

        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        st_no, st_nm = getattr(r, "produced_stock_no", None), getattr(r, "produced_stock_name", None)

        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # ===== NEW: Check equipment_type =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type) #(equipment_type is True)
        
        if is_input:
            var_name = getattr(r, "equipment_name", None) or getattr(r, "parameter", None)
            var_value = getattr(r, "gen_read_val", None)
            
            if not var_name or var_value is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(var_name)}",
                "counter_reading": _num_text(var_value),
                "equipment_name": str(var_name),
                "equipment_type": True,
            })
        else:
            pname = getattr(r, "parameter", None) or getattr(r, "equipment_no", None)
            cval = getattr(r, "counter_reading", None)
            eq_name = getattr(r, "equipment_name", None)
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })

    if not bucket:
        p3_1_log.info(
            f"[fetch_latest_for_ws_via_raw_table] ws_id={ws_id}: 0 rows found in RAW_TABLE fallback"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "workstation_id": ws_id,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[fetch_latest_for_ws_via_raw_table] ws_id={ws_id}: built {len(sensor_values)} bundles from RAW_TABLE"
    )
    return job_ids, dates, ids, sensor_values

# -------- Buffer'dan sensor_values üret (PID-level) -------- #
# ADD THIS HELPER FIRST (after line ~120)
def _extract_invars_from_message(message: dict):
    """
    Extract input variables from message's inVars or inputVariableList.
    
    Returns list of dicts with:
        - var_name: variable name (priority: varNm > varId)
        - var_value: variable value (genReadVal)
    """
    invars = message.get("inVars") or message.get("inputVariableList") or []
    if not isinstance(invars, list):
        return []
    
    result = []
    for iv in invars:
        if not isinstance(iv, dict):
            continue
        
        var_name = iv.get("varNm") or iv.get("varId")
        if not var_name or var_name in (None, "", "None"):
            continue
        
        var_value = iv.get("genReadVal")
        if var_value is None:
            continue
        
        result.append({
            "var_name": str(var_name),
            "var_value": var_value,
        })
    
    return result

# THEN UPDATE THE FUNCTION:
def build_sensor_values_from_pid_buffer(pid: int, messages):
    """
    UPDATED: Builds sensor_values from Kafka buffer with BOTH inputs and outputs.
    """
    p3_1_log.info(
        f"[build_sensor_values_from_pid_buffer] pid={pid}: received {len(messages or [])} buffered messages"
    )

    bucket = defaultdict(list)
    label_meta = {}

    for msg in messages or []:
        # ===== OUTPUTS (original) =====
        out_vals = msg.get("outVals") or []
        
        # ===== INPUTS (NEW) =====
        in_vars = _extract_invars_from_message(msg)
        
        if not out_vals and not in_vars:
            continue

        # Get timestamp
        meas_ms = (out_vals[0].get("measDt") if out_vals else None) or msg.get("crDt")
        ts = _to_dt(meas_ms)
        if ts is None:
            p3_1_log.debug(
                f"[build_sensor_values_from_pid_buffer] pid={pid}: could not parse ts"
            )
            continue

        st_no, st_nm = _extract_stock_from_prodlist_message(msg.get("prodList"))

        if ts not in label_meta:
            label_meta[ts] = {
                "good": msg.get("goodCnt"),
                "prSt": msg.get("prSt"),
                "job_order_reference_no": msg.get("joRef"),
                "prod_order_reference_no": msg.get("refNo"),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": msg.get("opNm"),
                "operationno": msg.get("opNo"),
                "operationtaskcode": msg.get("opTc")
            }

        # Add OUTPUT sensors
        for ov in out_vals:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })
        
        # Add INPUT sensors (NEW)
        for iv in in_vars:
            bucket[ts].append({
                "parameter": f"{iv['var_name']}",
                "counter_reading": _num_text(iv['var_value']),
                "equipment_name": iv['var_name'],
                "equipment_type": True,
            })

    if not bucket:
        p3_1_log.info(f"[build_sensor_values_from_pid_buffer] pid={pid}: 0 bundles from buffer")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "job_order_operation_id": pid,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(pid)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[build_sensor_values_from_pid_buffer] pid={pid}: built {len(sensor_values)} bundles from buffer"
    )
    return job_ids, dates, ids, sensor_values

# -------- Buffer'dan sensor_values üret (WS-level) -------- #
def build_sensor_values_from_ws_buffer(ws_id: int, messages):
    """
    UPDATED: Builds sensor_values from Kafka buffer with BOTH inputs and outputs.
    """
    p3_1_log.info(
        f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: received {len(messages or [])} buffered messages"
    )

    bucket = defaultdict(list)
    label_meta = {}

    for msg in messages or []:
        out_vals = msg.get("outVals") or []
        in_vars = _extract_invars_from_message(msg)
        
        if not out_vals and not in_vars:
            continue

        meas_ms = (out_vals[0].get("measDt") if out_vals else None) or msg.get("crDt")
        ts = _to_dt(meas_ms)
        if ts is None:
            p3_1_log.debug(
                f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: could not parse ts"
            )
            continue

        st_no, st_nm = _extract_stock_from_prodlist_message(msg.get("prodList"))

        if ts not in label_meta:
            label_meta[ts] = {
                "good": msg.get("goodCnt"),
                "prSt": msg.get("prSt"),
                "job_order_reference_no": msg.get("joRef"),
                "prod_order_reference_no": msg.get("refNo"),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": msg.get("opNm"),
                "operationno": msg.get("opNo"),
                "operationtaskcode": msg.get("opTc"),
            }

        # Add OUTPUTS
        for ov in out_vals:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })
        
        # Add INPUTS (NEW)
        for iv in in_vars:
            bucket[ts].append({
                "parameter": f"{iv['var_name']}",
                "counter_reading": _num_text(iv['var_value']),
                "equipment_name": iv['var_name'],
                "equipment_type": True,
            })

    if not bucket:
        p3_1_log.info(f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: 0 bundles from buffer")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "workstation_id": ws_id,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: built {len(sensor_values)} bundles from buffer"
    )
    return job_ids, dates, ids, sensor_values

def _clean_stock(v):
    if v is None:
        return None
    s = str(v)
    if s in ("", "None", "nan", "NaN"):
        return None
    return s

def _sensor_values_has_stock(sensor_values, stock_no: str) -> bool:
    """
    sensor_values: list of bundles -> each bundle is [meta_dict, sensor1_dict, ...]
    We check meta["output_stock_no"].
    """
    stock_no = _clean_stock(stock_no)
    if not stock_no or not sensor_values:
        return False

    for bundle in sensor_values:
        if not bundle or not isinstance(bundle, list):
            p3_1_log.debug(f"[_sensor_values_has_stock] Skipping bundle with invalid format: {bundle}")
            continue
        meta = bundle[0] if len(bundle) > 0 else None
        if not isinstance(meta, dict):
            p3_1_log.debug(f"[_sensor_values_has_stock] Skipping bundle with invalid meta: {meta}")
            continue
        st = _clean_stock(meta.get("output_stock_no"))
        if st == stock_no:
            return True
    return False

def _count_points_for_stock(sensor_values, stock_no: str) -> int:
    """
    Count distinct time points for that stock. Uses meta["crDt"] if present, else counts bundles.
    """
    stock_no = _clean_stock(stock_no)
    if not stock_no or not sensor_values:
        return 0

    crdts = set()
    cnt = 0
    for bundle in sensor_values:
        if not bundle or not isinstance(bundle, list):
            continue
        meta = bundle[0] if len(bundle) > 0 else None
        if not isinstance(meta, dict):
            p3_1_log.debug(f"[_count_points_for_stock] Skipping bundle with invalid meta: {meta}")
            continue
        st = _clean_stock(meta.get("output_stock_no"))
        if st != stock_no:
            #p3_1_log.debug(f"[_count_points_for_stock] Skipping bundle with stock_no={st}")
            continue

        cr = meta.get("crDt")
        if cr is not None:
            crdts.add(str(cr))
        else:
            cnt += 1

    return len(crdts) if crdts else cnt


def fetch_for_optc_stock_via_dw(rows, input_list, output_list, _batch, 
                                 op_tc=None, stock_no=None, 
                                 ws_id=None, pid=None, max_rows: int = 20000):
    """
    Fetch data by operation task code + stock (NOT by single PID).
    Optionally filter by ws_id or pid for additional constraints.
    
    This enables training across ALL PIDs with same op_tc + stock.
    """
    bucket = defaultdict(list)
    label_meta = {}
    pid_by_ts = {}
    ws_by_ts = {}
    
    # Clean inputs
    op_tc = None if op_tc in (None, "", "None") else str(op_tc)
    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)
    
    if not op_tc or not stock_no:
        p3_1_log.warning(
            f"[fetch_for_optc_stock_via_dw] Missing op_tc={op_tc} or stock_no={stock_no}, returning empty"
        )
        return [], [], [], []
    
    combined_list = _combine_input_output_lists(input_list, output_list)
    
    for r, wrapped_ov in zip(rows, combined_list):
        # ✅ PRIMARY FILTER: operation task code
        row_optc = getattr(r, "operationtaskcode", None)
        if row_optc != op_tc:
            continue
        
        # ✅ PRIMARY FILTER: stock
        row_stock = getattr(r, "produced_stock_no", None)
        row_stock = None if row_stock in (None, "", "None", "nan", "NaN") else str(row_stock)
        if row_stock != stock_no:
            continue
        
        # ✅ OPTIONAL FILTER: workstation (eğer belirtilmişse)
        if ws_id is not None:
            row_ws = getattr(r, "work_station_id", None) or getattr(r, "workstation_id", None)
            if row_ws != ws_id:
                continue
        
        # ✅ OPTIONAL FILTER: pid (eğer sadece o PID'i istiyorsak - genelde istemeyiz)
        if pid is not None:
            row_pid = getattr(r, "job_order_operation_id", None)
            if row_pid != pid:
                continue
        
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue
        
        # Track which PID/WS this timestamp belongs to
        if ts not in pid_by_ts:
            pid_by_ts[ts] = getattr(r, "job_order_operation_id", None)
        if ts not in ws_by_ts:
            ws_by_ts[ts] = getattr(r, "work_station_id", None)
        
        # Label/meta
        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": row_stock,
                "output_stock_name": getattr(r, "produced_stock_name", None),
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": row_optc,
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None) or getattr(r, "workstation_id", None),
            }
        
        # Sensor data
        ov = wrapped_ov[0] if wrapped_ov else {}
        if ov.get("equipment_type", True):  # INPUT
            var_name = ov.get("varNo")
            var_value = ov.get("genReadVal")
            
            if var_name and var_value is not None:
                bucket[ts].append({
                    "parameter": str(var_name),
                    "counter_reading": _num_text(var_value),
                    "equipment_name": ov.get("varNm", var_name),
                    "equipment_type": True,
                })
        else:  # OUTPUT
            param = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            
            if param and cval is not None:
                bucket[ts].append({
                    "parameter": str(param),
                    "counter_reading": _num_text(cval),
                    "equipment_name": str(eq_name),
                    "equipment_type": False,
                })
    
    if not bucket:
        p3_1_log.info(
            f"[fetch_for_optc_stock_via_dw] 0 rows for op_tc={op_tc}, stock={stock_no}"
        )
        return [], [], [], []
    
    job_ids, dates, ids, sensor_values = [], [], [], []
    
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue
        
        lm = label_meta.get(ts, {}) or {}
        meta = {
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
        }
        
        job_ids.append(pid_by_ts.get(ts) or 0)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])
    
    p3_1_log.info(
        f"[fetch_for_optc_stock_via_dw] Found {len(sensor_values)} bundles for "
        f"op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, pid={pid}"
    )
    return job_ids, dates, ids, sensor_values

def build_sensor_values_from_buffer_for_optc_stock(messages, op_tc=None, stock_no=None, 
                                                     ws_id=None, pid=None):
    """
    Build sensor_values from buffered messages filtered by op_tc + stock.
    Optionally filter by ws_id or pid.
    """
    op_tc = None if op_tc in (None, "", "None") else str(op_tc)
    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)
    
    if not op_tc or not stock_no:
        p3_1_log.warning(
            f"[build_sensor_values_from_buffer_for_optc_stock] Missing op_tc or stock_no"
        )
        return [], [], [], []
    
    bucket = defaultdict(list)
    label_meta = {}
    pid_by_ts = {}
    
    for msg in messages or []:
        # ✅ Filter by operation task code
        msg_optc = msg.get("operationtaskcode") or msg.get("opTc")
        if msg_optc != op_tc:
            continue
        
        # ✅ Filter by stock
        msg_stock, msg_stock_name = _extract_stock_from_prodlist_message(msg.get("prodList"))
        msg_stock = None if msg_stock in (None, "", "None", "nan", "NaN") else str(msg_stock)
        if msg_stock != stock_no:
            continue
        
        # ✅ Optional filter by workstation
        if ws_id is not None:
            msg_ws = msg.get("wsId")
            if msg_ws != ws_id:
                continue
        
        # ✅ Optional filter by pid
        if pid is not None:
            msg_pid = msg.get("joOpId")
            if msg_pid != pid:
                continue
        
        # Extract inputs and outputs
        out_vals = msg.get("outVals") or []
        in_vars = _extract_invars_from_message(msg)
        
        if not out_vals and not in_vars:
            continue
        
        meas_ms = (out_vals[0].get("measDt") if out_vals else None) or msg.get("crDt")
        ts = _to_dt(meas_ms)
        if ts is None:
            continue
        
        # Track PID for this timestamp
        if ts not in pid_by_ts:
            pid_by_ts[ts] = msg.get("joOpId")
        
        if ts not in label_meta:
            label_meta[ts] = {
                "good": msg.get("goodCnt"),
                "prSt": msg.get("prSt"),
                "job_order_reference_no": msg.get("joRef"),
                "prod_order_reference_no": msg.get("refNo"),
                "output_stock_no": msg_stock,
                "output_stock_name": msg_stock_name,
                "operationname": msg.get("opNm"),
                "operationno": msg.get("opNo"),
                "operationtaskcode": msg_optc,
                "job_order_operation_id": msg.get("joOpId"),
                "work_station_id": msg.get("wsId"),
            }
        
        # Add OUTPUTS
        for ov in out_vals:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            
            if pname and cval is not None:
                bucket[ts].append({
                    "parameter": str(pname),
                    "counter_reading": _num_text(cval),
                    "equipment_name": str(eq_name),
                    "equipment_type": False,
                })
        
        # Add INPUTS
        for iv in in_vars:
            bucket[ts].append({
                "parameter": iv['var_name'],
                "counter_reading": _num_text(iv['var_value']),
                "equipment_name": iv['var_name'],
                "equipment_type": True,
            })
    
    if not bucket:
        p3_1_log.info(
            f"[build_sensor_values_from_buffer_for_optc_stock] 0 bundles for "
            f"op_tc={op_tc}, stock={stock_no}"
        )
        return [], [], [], []
    
    job_ids, dates, ids, sensor_values = [], [], [], []
    
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue
        
        lm = label_meta.get(ts, {})
        meta = {
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
        }
        
        job_ids.append(pid_by_ts.get(ts) or 0)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])
    
    p3_1_log.info(
        f"[build_sensor_values_from_buffer_for_optc_stock] Built {len(sensor_values)} bundles "
        f"for op_tc={op_tc}, stock={stock_no}"
    )
    return job_ids, dates, ids, sensor_values

def fetch_latest_for_optc_stock_via_raw_table(
    op_tc,
    stock_no,
    *,
    max_rows=20000,
    ws_id=None,
    pid=None,
):
    """
    Fetch data from RAW_TABLE filtered by operation task code + stock.
    Optionally filter by ws_id or pid.
    
    Returns data from ALL PIDs with matching op_tc + stock (unless pid is specified).
    UPDATED: Handles BOTH inputs and outputs from RAW_TABLE.
    """
    op_tc = None if op_tc in (None, "", "None") else str(op_tc)
    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no).strip()

    if not op_tc or not stock_no:
        p3_1_log.warning(
            f"[fetch_latest_for_optc_stock_via_raw_table] Missing op_tc={op_tc} or stock_no={stock_no}"
        )
        return [], [], [], []

    # Build WHERE clause dynamically
    where = "WHERE operationtaskcode = %s AND produced_stock_no = %s"
    params = [op_tc, stock_no]

    if ws_id is not None:
        where += " AND work_station_id = %s"
        params.append(ws_id)
    
    #if pid is not None:
        #where += " AND job_order_operation_id = %s"
        #params.append(pid)

    query = (
        f"SELECT job_order_operation_id, work_station_id, measurement_date, "
        f"equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type, "
        f"good, workstation_state, job_order_reference_no, prod_order_reference_no, "
        f"produced_stock_no, produced_stock_name, operationname, operationno, operationtaskcode "
        f"FROM {RAW_TABLE} "
        f"{where} "
        f"LIMIT %s ALLOW FILTERING"
    )

    params.append(max_rows)

    try:
        sess = _get_cassandra_session(p3_1_log=p3_1_log)
        if sess is None:
            return [], [], [], []
        rows = list(sess.execute(query, tuple(params)))
    except Exception as e:
        p3_1_log.warning(
            f"[fetch_latest_for_optc_stock_via_raw_table] query failed for "
            f"op_tc={op_tc}, stock={stock_no}: {e}"
        )
        return [], [], [], []

    if not rows:
        p3_1_log.info(
            f"[fetch_latest_for_optc_stock_via_raw_table] 0 rows for "
            f"op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, pid={pid}"
        )
        return [], [], [], []

    bucket = defaultdict(list)
    label_meta = {}
    pid_by_ts = {}
    ws_by_ts = {}

    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        # Track which PID/WS this timestamp belongs to
        if ts not in pid_by_ts:
            pid_by_ts[ts] = getattr(r, "job_order_operation_id", None)
        if ts not in ws_by_ts:
            ws_by_ts[ts] = getattr(r, "work_station_id", None)

        # Label/meta info
        if ts not in label_meta:
            st_no = getattr(r, "produced_stock_no", None)
            st_nm = getattr(r, "produced_stock_name", None)

            st_no = None if st_no in (None, "", "None", "nan", "NaN") else str(st_no)
            st_nm = None if st_nm in (None, "", "None", "nan", "NaN") else str(st_nm)

            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None),
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None),
            }

        # ===== Check equipment_type for INPUT vs OUTPUT =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type)
        
        if is_input:
            # INPUT
            var_name = getattr(r, "equipment_name", None) or getattr(r, "parameter", None)
            var_value = getattr(r, "gen_read_val", None)
            
            if not var_name or var_value is None:
                continue
            
            bucket[ts].append({
                "parameter": str(var_name),
                "counter_reading": _num_text(var_value),
                "equipment_name": str(var_name),
                "equipment_type": True,
            })
        else:
            # OUTPUT
            pname = getattr(r, "parameter", None) or getattr(r, "equipment_no", None)
            cval = getattr(r, "counter_reading", None)
            eq_name = getattr(r, "equipment_name", None)
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })

    if not bucket:
        p3_1_log.info(
            f"[fetch_latest_for_optc_stock_via_raw_table] 0 bundles after processing for "
            f"op_tc={op_tc}, stock={stock_no}"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {}) or {}
        meta = {
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
        }

        job_ids.append(pid_by_ts.get(ts) or 0)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[fetch_latest_for_optc_stock_via_raw_table] rows={len(rows)}, "
        f"bundles={len(sensor_values)} for op_tc={op_tc}, stock={stock_no}, "
        f"ws_id={ws_id}, pid={pid}"
    )
    return job_ids, dates, ids, sensor_values

def _is_message_valid(message):
    """
    Validates required fields in message.
    Returns True if valid, False otherwise.
    """

    # Source of truth: Kafka key → message["customer"]
    if not (message.get("customer") or message.get("cust")):
        p3_1_log.warning("[execute_phase_three] Skipping message: customer missing (kafka key / message.cust)")
        return False

    # Must have at least one measurement list
    has_out = isinstance(message.get("outVals"), list) and len(message["outVals"]) > 0
    has_in  = isinstance(message.get("inVars"), list) and len(message["inVars"]) > 0
    if not (has_out or has_in):
        p3_1_log.warning("[execute_phase_three] Skipping message: no outVals/inVars present")
        return False

    # plId/wsId checks only if key exists
    if "plId" in message and message["plId"] is None:
        p3_1_log.info("[execute_phase_three] Skipping message: plId is None")
        return False

    # NOTE: wsId may be missing for some workstations; don't hard-fail here.
    # joOpId may also be missing; we can still run WS-level logic.

    return True


# -------- Phase 3 main loop -------- #
def execute_phase_three():
    p3_1_log.info("[execute_phase_three] Initializing Phase 3")

    # Legacy entrypoint: kept for backward compatibility only.
    # Default is queue-mode; enable explicitly by setting config.phase3_entrypoint=legacy_kafka
    entry = str(getattr(cfg, 'phase3_entrypoint', 'queue') or 'queue').strip().lower()
    if entry not in ('legacy_kafka', 'kafka'):
        raise RuntimeError(
            'execute_phase_three() is legacy Kafka mode and is disabled by default. '
            'Use execute_phase_three_worker() (queue mode) or set phase3_entrypoint=legacy_kafka.'
        )

    # Lazy import to avoid Kafka dependency in queue-mode
    from modules.kafka_modules import kafka_consumer3

    global consumer3

    if consumer3 is None:
        consumer3 = kafka_consumer3()

    # İlk başta DW snapshot'ı al
    rows, input_list, output_list, _batch = [], [], [], []

    if PHASE3_DW_SNAPSHOT_ENABLED:
        try:
            rows, input_list, output_list, _batch = _try_fetch_dw_snapshot(limit=20_000, p3_1_log=p3_1_log)
            p3_1_log.info(f"[execute_phase_three] DW snapshot loaded: rows={len(rows)}")
        except Exception as e:
            p3_1_log.error(f"[execute_phase_three] DW snapshot fetch failed (continuing without snapshot): {e}", exc_info=True)
    else:
        p3_1_log.info("[execute_phase_three] DW snapshot disabled (phase3_dw_snapshot_enabled=false)")

    # Ardışık None sayacı
    none_counter = 0
    NONE_LOG_EVERY = 30          # log every 30 consecutive Nones
    NO_ASSIGN_SLEEP_SEC = 1.0    # backoff when assignment=[]

    try:
        while True:        
            # 1) Kafka'dan mesaj çek
            msg = consumer3.poll(1.0)
            
            # ---- NONE HANDLING (soft restart) ----
            if msg is None:
                none_counter += 1

                if none_counter % NONE_LOG_EVERY == 0:
                    try:
                        a = consumer3.assignment()
                        p3_1_log.info(
                            f"[execute_phase_three] none_storm "
                            f"consecutive={none_counter} assignment={a}"
                        )

                        if not a:
                            p3_1_log.info(
                                "[execute_phase_three] none_storm: assignment=[] "
                                "(another consumer owns the partition or rebalance in progress)"
                            )
                        else:
                            p3_1_log.info(f"[execute_phase_three] position={consumer3.position(a)}")
                            p3_1_log.info(f"[execute_phase_three] committed={consumer3.committed(a, timeout=5)}")

                    except Exception as e:
                        p3_1_log.info(f"[execute_phase_three] none_storm debug failed: {e}")

                try:
                    if not consumer3.assignment():
                        time.sleep(NO_ASSIGN_SLEEP_SEC)
                except Exception:
                    pass

                continue


            # Buraya geldiysek gerçek bir mesaj var, sayaç sıfırla
            none_counter = 0

            # ---- ERROR HANDLING (opsiyonel: burada da restart edebilirsin) ----
            if msg.error():
                p3_1_log.error(
                    f"[execute_phase_three] Errornous Message: {msg.error()}"
                )
                # İstersen burada da aynı soft-restart mantığını kullan:
                try:
                    consumer3.close()
                except Exception as e:
                    p3_1_log.warning(
                        f"[execute_phase_three] Error while closing consumer after error: {e}"
                    )
                try:
                    consumer3 = kafka_consumer3()
                    p3_1_log.info(
                        "[execute_phase_three] Kafka consumer recreated after error"
                    )
                except Exception as e:
                    p3_1_log.error(
                        f"[execute_phase_three] Failed to recreate consumer after error: {e}",
                        exc_info=True,
                    )
                continue
            
            """try:
                rows, input_list, output_list, _batch = _try_fetch_dw_snapshot(limit=20_000, p3_1_log=p3_1_log)
            except Exception as e:
                p3_1_log.error(f"[dw_tbl_raw_data.fetchData] fetchData failed: {e}")
                return [], [], [], []"""
            
            try:
                raw_value = msg.value().decode("utf-8")
                message = json.loads(raw_value)

                # --- customer comes from Kafka message key (source of truth) ---
                k = msg.key()
                customer = (
                    k.decode("utf-8") if isinstance(k, (bytes, bytearray))
                    else (str(k) if k is not None else None)
                )

                # Fallbacks (if producer didn't set key)
                if not customer:
                    customer = message.get("customer") or message.get("cust")
                    if (not customer) and isinstance(message.get("outVals"), list) and message["outVals"]:
                        ov0 = message["outVals"][0]
                        if isinstance(ov0, dict):
                            customer = ov0.get("cust")

                message["customer"] = customer
                message["cust"] = customer  # backward compat

                # IMPORTANT: do NOT mutate message["outVals"][*]["cust"]

                # # # # # # # # # # # # # # 
                # --- M1 (shadow): batching and planning: derive workstation_uid + batch context ---
                try:
                    bctx = batch_assigner.assign(message)
                    message["_workstation_uid"] = bctx.workstation_uid
                    message["_batch_id"] = bctx.batch_id
                    message["_batch_strategy"] = bctx.strategy
                    message["_batch_confidence"] = bctx.confidence
                    message["_phase_id"] = bctx.phase_id
                    message["_event_ts_ms"] = bctx.event_ts_ms

                    if M1_LOG_BATCH_CONTEXT:
                        p3_1_log.info(
                            f"[M1] batch_ctx ws={bctx.workstation_uid} "
                            f"strat={bctx.strategy} batch={bctx.batch_id} "
                            f"conf={bctx.confidence:.2f} phase={bctx.phase_id} "
                            f"event_ts_ms={bctx.event_ts_ms} reason={bctx.reason}"
                        )
                except Exception as e:
                    p3_1_log.error(f"[M1] batch_assigner failed: {e}", exc_info=True)

                if M1_ENABLE_TRAINING_PLANNER:
                    try:
                        training_planner.observe(message)
                        training_planner.maybe_log(message["_workstation_uid"], p3_1_log)
                    except Exception as e:
                        p3_1_log.error(f"[M1] training_planner failed: {e}", exc_info=True)

                # # # # # # # # # # # # # # 
                
                if not _is_message_valid(message):
                    p3_1_log.info("[execute_phase_three] Skipping invalid message")
                    continue

                # Map joRef/refNo into raw table column names (do NOT overwrite if one is missing)
                if "joRef" in message:
                    message["job_order_reference_no"] = message.get("joRef")
                if "refNo" in message:
                    message["prod_order_reference_no"] = message.get("refNo")

                # Normalize all four fields to backend-friendly strings.
                # Unknown/missing/composite-null => "0"
                message["job_order_reference_no"] = _norm_id_or_zero_str(message.get("job_order_reference_no"))
                message["prod_order_reference_no"] = _norm_id_or_zero_str(message.get("prod_order_reference_no"))
                message["joRef"] = _norm_id_or_zero_str(message.get("joRef"))
                message["refNo"] = _norm_id_or_zero_str(message.get("refNo"))


                # NEW: extract stock from prodList and put on message
                st_no, st_nm = _extract_stock_from_prodlist_message(
                    message.get("prodList")
                )
                message["output_stock_no"] = st_no
                message["output_stock_name"] = st_nm

                message["operationname"] = message.get("opNm")
                message["operationno"] = message.get("opNo")
                message["operationtaskcode"] = message.get("opTc")


                if message.get("plId") and message["plId"] == 76:
                    p3_1_log.info("[execute_phase_three] Skipping asasa customer")
                    continue

                _m_inc("kafka_msg_parsed", 1)

                # DW raw table'a tek satır kaydet (durability boundary)
                if PHASE3_RAW_PERSIST_ENABLED:
                    try:
                        from cassandra_utils.models.dw_single_data import dw_tbl_raw_data
                        _ = dw_tbl_raw_data.saveData(message)
                        _m_inc("raw_write_ok", 1)
                    except Exception as e:
                        _m_inc("raw_write_fail", 1)
                        p3_1_log.error(f"[execute_phase_three] RAW write failed (will NOT commit): {e}", exc_info=True)
                        _maybe_log_metrics(p3_1_log, time.time())
                        continue
                else:
                    _m_inc("raw_write_skip", 1)

                # Commit AFTER successful raw write
                try:
                    consumer3.commit(message=msg, asynchronous=False)
                    _m_inc("kafka_commit_ok", 1)
                except Exception as e:
                    _m_inc("kafka_commit_fail", 1)
                    p3_1_log.error(f"[execute_phase_three] Kafka commit failed (will reprocess on restart): {e}", exc_info=True)

                # Phase2 always-on (Mode B): enqueue AFTER durability boundary (raw persist + commit)
                try:
                    phase2_enqueue(message)
                except Exception as _e:
                    # never fail ingestion because of Phase2
                    _m_inc("phase2_enqueue_fail", 1)

                # ---- M1.8: mode + allowlist gating (after durability boundary) ----
                ws_uid = _phase3_get_ws_uid(message)

                if PHASE3_MODE == "persist_only":
                    _m_inc("persist_only_msg", 1)
                    _m_inc("heavy_skip_reason_persist_only", 1)
                    _maybe_log_metrics(p3_1_log, time.time())
                    continue

                if PHASE3_WORKSTATION_ALLOWLIST and ws_uid not in PHASE3_WORKSTATION_ALLOWLIST:
                    _m_inc("allowlist_skip", 1)
                    _m_inc("heavy_skip_reason_allowlist", 1)
                    now_epoch = time.time()
                    last = _PHASE3_LAST_SKIP_LOG.get(ws_uid, 0.0)
                    if (now_epoch - last) >= PHASE3_SKIP_LOG_EVERY_SEC:
                        _PHASE3_LAST_SKIP_LOG[ws_uid] = now_epoch
                        p3_1_log.info(f"[phase3] persist_only_for_ws ws={ws_uid} (not in allowlist)")
                    _maybe_log_metrics(p3_1_log, now_epoch)
                    continue


                # Belirli müşteri skip
                """if message.get("outVals") and message["outVals"][0].get("cust") == "teknia_group":
                    p3_1_log.info("[execute_phase_three] Skipping teknia_group customer")
                    continue

                if message.get("plId") and message["plId"] == 20:
                    p3_1_log.info("[execute_phase_three] Skipping Savola customer")
                    continue

                if message.get("plId") and message["plId"] == 162:
                    p3_1_log.info("[execute_phase_three] Skipping Meriç customer")
                    continue
                
                if message.get("prodList") and message["prodList"][0].get("stNo") == "Antares PV":
                    p3_1_log.info("[execute_phase_three] Skipping Antares PV stock")
                    continue

                if message.get("prodList") and message["prodList"][0].get("stNo") == "Loperamide 2 mg granulate":
                    p3_1_log.info("[execute_phase_three] Skipping Loperamide 2 mg granulate stock")
                    continue"""
                
                plant = message.get("plId")
                wsSt = message.get("prSt")
                pid = message.get("joOpId") #
                ws_id = message.get("wsId")
                crDt_msg = _to_dt(message.get("crDt"))
                now_epoch = time.time()

                pid_key = _pid_buffer_key(message, pid)

                cnt = buffer_for_process.get(pid_key, {}).get("count", 0)
                p3_1_log.info(
                    f"[execute_phase_three] Initializing for -- plId={plant}, prod={message.get('refNo')}, "
                    f"pid={pid}, pid_key={pid_key}, wsId={ws_id}, wsSt={wsSt}, "
                    f"In Buffer={pid_key in buffer_for_process}, count={cnt}"
                )

                # --- PID buffer yönetimi ---
                if wsSt == "PRODUCTION" and pid is not None:
                    if pid_key not in buffer_for_process:
                        buffer_for_process[pid_key] = {
                            "count": 1,
                            "first_seen": now_epoch,
                            "last_time": now_epoch,
                            "first_crdt": crDt_msg,
                            "messages": [message],
                        }
                        p3_1_log.info(f"[execute_phase_three] Initialized PID buffer for pid_key={pid_key}")
                    else:
                        b = buffer_for_process[pid_key]
                        b["count"] = b.get("count", 0) + 1
                        b["last_time"] = now_epoch
                        if b.get("first_crdt") is None and crDt_msg:
                            b["first_crdt"] = crDt_msg
                        b.setdefault("messages", []).append(message)

                # --- WS buffer yönetimi ---
                if wsSt == "PRODUCTION" and ws_id is not None:
                    if ws_id not in buffer_for_ws:
                        buffer_for_ws[ws_id] = {
                            "count": 1,
                            "first_seen": now_epoch,
                            "last_time": now_epoch,
                            "first_crdt": crDt_msg,
                            "messages": [message],
                        }
                        p3_1_log.info(f"[execute_phase_three] Initialized WS buffer for wsId={ws_id}")
                    else:
                        w = buffer_for_ws[ws_id]
                        w["count"] = w.get("count", 0) + 1
                        w["last_time"] = now_epoch
                        if w.get("first_crdt") is None and crDt_msg:
                            w["first_crdt"] = crDt_msg
                        w.setdefault("messages", []).append(message)

                # --- PID / WS için threshold check ---
                if wsSt == "PRODUCTION" and pid is not None and pid_key in buffer_for_process:
                    
                    b = buffer_for_process[pid_key]
                    fire_pid, pid_cnt, pid_time_lapsed = _should_fire(b, PID_COUNT_THRESHOLD, PID_TIME_THRESHOLD, now_epoch)

                    p3_1_log.info(
                        f"[execute_phase_three] pid={pid}, pid_key={pid_key}, wsId={ws_id}, wsSt={wsSt} "
                        f"Updated PID buffer count={pid_cnt}, time_lapsed={pid_time_lapsed}"
                    )

                    if fire_pid:
                        # ---- M1.8 / ModeB gate ----
                        if not _phase3_should_run_heavy(message, scope="pid", scope_id=pid, now_epoch=now_epoch):
                            _m_inc("heavy_pid_skip", 1)
                            p3_1_log.info(
                                f"[phase3] heavy_skip scope=pid mode={PHASE3_MODE} ws={_phase3_get_ws_uid(message)} pid={pid} "
                                f"(every_sec={PHASE3_HEAVY_EVERY_SEC}, key_scope={PHASE3_HEAVY_KEY_SCOPE})"
                            )
                            # reset PID buffer to prevent repeated firing
                            b["count"] = 0
                            b["first_seen"] = b["last_time"]
                            b["first_crdt"] = None
                            b["messages"] = []
                        else:
                            _m_inc("heavy_pid_run", 1)

                            # Extract current message's op_tc and stock
                            op_tc = message.get("operationtaskcode") or message.get("opTc")
                            stock_no = message.get("output_stock_no")
                            ws_id_current = message.get("wsId")

                            # Clean them
                            op_tc = None if op_tc in (None, "", "None") else str(op_tc)
                            stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)

                            p3_1_log.info(
                                f"[execute_phase_three] PID={pid} FETCHING DATA for op_tc={op_tc}, "
                                f"stock={stock_no}, ws_id={ws_id_current}"
                            )

                            # === (1) DW SNAPSHOT - by op_tc + stock (NOT by PID!) ===
                            job_ids_pid, dates_pid, ids_pid, sensor_values_pid = fetch_for_optc_stock_via_dw(
                                rows, input_list, output_list, _batch,
                                op_tc=op_tc,
                                stock_no=stock_no,
                                ws_id=ws_id_current,
                                pid=None,
                                max_rows=20_000
                            )

                            # === (2) RAW_TABLE FALLBACK ===
                            if not sensor_values_pid:
                                p3_1_log.info(
                                    f"[execute_phase_three] pid={pid} — DW returned 0 bundles; trying RAW_TABLE fallback"
                                )
                                job_ids_pid, dates_pid, ids_pid, sensor_values_pid = fetch_latest_for_optc_stock_via_raw_table(
                                    op_tc=op_tc,
                                    stock_no=stock_no,
                                    ws_id=ws_id_current,
                                    pid=None,
                                    max_rows=20_000
                                )

                            # === (3) BUFFER FALLBACK (all PID buffers) ===
                            if not sensor_values_pid:
                                all_buffered_messages = []
                                for _, buf_entry in buffer_for_process.items():
                                    all_buffered_messages.extend(buf_entry.get("messages", []))

                                buffered_n = len(all_buffered_messages)
                                p3_1_log.info(
                                    f"[execute_phase_three] pid={pid} — RAW_TABLE returned 0 bundles; trying buffer fallback "
                                    f"(total_buffered_messages={buffered_n})"
                                )

                                job_ids_pid, dates_pid, ids_pid, sensor_values_pid = build_sensor_values_from_buffer_for_optc_stock(
                                    all_buffered_messages,
                                    op_tc=op_tc,
                                    stock_no=stock_no,
                                    ws_id=ws_id_current,
                                    pid=None
                                )

                            if not sensor_values_pid:
                                p3_1_log.info(
                                    f"[execute_phase_three] pid={pid} — still no data after DW+RAW+buffer "
                                    f"for op_tc={op_tc}, stock={stock_no}; skipping"
                                )
                            else:
                                p3_1_log.info(
                                    f"[execute_phase_three] Fetched {len(dates_pid)} time points "
                                    f"for op_tc={op_tc}, stock={stock_no} (from multiple PIDs)"
                                )

                                try:
                                    _run_3_tasks_and_wait(
                                        sensor_values=sensor_values_pid,
                                        message=message,
                                        dates=dates_pid,
                                        input_list=input_list,
                                        output_list=output_list,
                                        p3_1_log=p3_1_log,
                                        scope="pid",
                                        scope_id=pid,
                                        corr_group_by_output_stock=False,
                                        pred_group_by_stock=True,
                                        pred_algorithm="RANDOM_FOREST",
                                        pred_epochs=125,
                                        pred_min_train_points=250,
                                        fi_algorithm="XGBOOST",
                                    )
                                except Exception as e:
                                    p3_1_log.error(f"[execute_phase_three] PID parallel runner failed for pid={pid}: {e}", exc_info=True)

                            # ---- PID buffer reset policy ----
                            b["count"] = 0
                            b["first_seen"] = b["last_time"]
                            b["first_crdt"] = None
                            b["messages"] = []

                    else:
                        _m_inc("corr_skip_buf_pid", 1)
                        _m_inc("heavy_skip_reason_insufficient_window", 1)
                        _m_inc("heavy_skip_reason_insufficient_window_pid", 1)
                        p3_1_log.info(
                            f"[execute_phase_three] Pid={pid}, wsId={ws_id}, wsSt={wsSt} "
                            f"SKIPPING PID-level Correlation (cnt={pid_cnt}, elapsed={pid_time_lapsed})"
                        )

                    # ---------- WS-LEVEL CHECK (AGGREGATED) ----------
                    if ws_id is not None and ws_id in buffer_for_ws:
                        w = buffer_for_ws[ws_id]
                        fire_ws, ws_cnt, ws_time_lapsed = _should_fire(
                            w, WS_COUNT_THRESHOLD, WS_TIME_THRESHOLD, now_epoch
                        )

                        p3_1_log.info(
                            f"[execute_phase_three] wsId={ws_id}, wsSt={wsSt} "
                            f"Updated WS buffer count={ws_cnt}, time_lapsed={ws_time_lapsed}"
                        )

                        if fire_ws:
                            # ---- M1.8: sampled_full gate for WS scope ----
                            if not _phase3_should_run_heavy(message, scope="ws", scope_id=ws_id, now_epoch=now_epoch):
                                _m_inc("heavy_ws_skip", 1)
                                p3_1_log.info(
                                    f"[phase3] sampled_skip scope=ws ws={_phase3_get_ws_uid(message)} wsId={ws_id} "
                                    f"(every_sec={PHASE3_HEAVY_EVERY_SEC}, key_scope={PHASE3_HEAVY_KEY_SCOPE})"
                                )
                                # reset WS buffer to prevent repeated firing
                                w["count"] = 0
                                w["first_seen"] = w["last_time"]
                                w["first_crdt"] = None
                                w["messages"] = []
                                continue  # skip rest of WS heavy block
                            else:
                                _m_inc("heavy_ws_run", 1)

                            p3_1_log.info(
                                f"[execute_phase_three] Fetching WS-level raw data for wsId={ws_id} "
                                f"(cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                            )

                            # (1) DW SNAPSHOT
                            job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_dw(
                                ws_id, rows, input_list, output_list, _batch, max_rows=20_000
                            )

                            # (2) RAW_TABLE FALLBACK
                            if not sensor_values_ws:
                                p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — DW returned 0 bundles; trying RAW_TABLE fallback")
                                job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_raw_table(
                                    ws_id, max_rows=20_000
                                )

                            # (3) WS BUFFER FALLBACK
                            if not sensor_values_ws:
                                buffered_n = len(w.get("messages", []))
                                p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — RAW_TABLE returned 0 bundles; trying WS buffer fallback (buffered_messages={buffered_n})")
                                job_ids_ws, dates_ws, ids_ws, sensor_values_ws = build_sensor_values_from_ws_buffer(
                                    ws_id, w.get("messages", [])
                                )

                            if not sensor_values_ws:
                                p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — still no data after DW+RAW+buffer; skipping")
                                continue
                            else:
                                # ---- STOCK GATE (WS + group_by_output_stock) ----
                                skip_ws = False
                                current_stock = _clean_stock(get_stock_key(message, default=""))

                                if current_stock is None:
                                    p3_1_log.warning(
                                        f"[execute_phase_three] wsId={ws_id} group_by_output_stock=True but current message has no output_stock_no. "
                                        "Skipping WS correlation/LSTM/feature-importance to avoid mixing stocks."
                                    )
                                    skip_ws = True
                                else:
                                    has_stock = _sensor_values_has_stock(sensor_values_ws, current_stock)
                                    pts = _count_points_for_stock(sensor_values_ws, current_stock)

                                    if (not has_stock) or (pts < 2):
                                        p3_1_log.info(
                                            f"[execute_phase_three] wsId={ws_id} stock gate: current_stock={current_stock} "
                                            f"has_stock={has_stock} points={pts} (<2 or missing). "
                                            "Skipping WS correlation/LSTM/feature-importance; waiting for next message."
                                        )
                                        skip_ws = True

                                if not skip_ws:
                                    # --- WS: run correlation + prediction + feature-importance in parallel ---
                                    # Mode B: DO NOT consume triggers before the heavy run succeeds.
                                    ws_uid0 = _phase3_get_ws_uid(message)
                                    stk0 = current_stock or "ALL"

                                    try:
                                        _run_3_tasks_and_wait(
                                            sensor_values=sensor_values_ws,   # (NOTE) use what you already feed to correlation/fi
                                            message=message,
                                            dates=dates_ws,
                                            input_list=input_list,
                                            output_list=output_list,
                                            p3_1_log=p3_1_log,
                                            scope="ws",
                                            scope_id=ws_id,
                                            corr_group_by_output_stock=True,  # keep your WS behavior
                                            pred_group_by_stock=True,         # keep your WS behavior
                                            pred_algorithm="RANDOM_FOREST",
                                            pred_epochs=125,
                                            pred_min_train_points=250,
                                            fi_algorithm="XGBOOST",
                                        )
                                        # Heavy run succeeded: consume any active anomaly trigger + reset baseline timer.
                                        trig_reason = consume_anomaly(ws_uid0, stk0, scope="ws")
                                        if trig_reason is None and stk0 != "ALL":
                                            trig_reason = consume_anomaly(ws_uid0, "ALL", scope="ws")
                                        if trig_reason:
                                            p3_1_log.info(f"[phase3] mode_b_triggered ws={ws_uid0} stock={stk0} reason={trig_reason}")
                                        if PHASE3_MODEB_BASELINE_EVERY_SEC > 0:
                                            mark_baseline_run(ws_uid0, stk0, scope="ws", now_epoch=now_epoch)
                                    except Exception as e:
                                        p3_1_log.error(f"[execute_phase_three] WS parallel runner failed for wsId={ws_id}: {e}", exc_info=True)

                                # ---- WS buffer reset policy ----
                                if not skip_ws:
                                    w["count"] = 0
                                    w["first_seen"] = w["last_time"]
                                    w["first_crdt"] = None
                                    w["messages"] = []
                                else:
                                    # keep buffer so next message can re-trigger and fetch includes the stock
                                    pass

                        else:
                            _m_inc("corr_skip_buf_ws", 1)
                            _m_inc("heavy_skip_reason_insufficient_window", 1)
                            _m_inc("heavy_skip_reason_insufficient_window_ws", 1)
                            p3_1_log.info(
                                f"[execute_phase_three] wsId={ws_id}, wsSt={wsSt} "
                                f"SKIPPING WS-level Correlation (cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                            )

                # PRODUCTION değilse PID / WS bufferları temizle
                if wsSt != "PRODUCTION":
                    p3_1_log.info(
                        f"[execute_phase_three] pid={pid}, pid_key={pid_key}, wsId={ws_id}, ws not in PRODUCTION, "
                        f"clearing buffers, wsSt={wsSt}"
                    )
                    if pid is not None:
                        buffer_for_process.pop(pid_key, None)
                    if ws_id is not None:
                        buffer_for_ws.pop(ws_id, None)

                _maybe_log_metrics(p3_1_log, now_epoch)
            except Exception as e2:
                _m_inc("err_lvl2", 1)
                p3_1_log.error(f"[execute_phase_three] Error Level 2: {e2}", exc_info=True)

    except Exception as e1:
        p3_1_log.error(f"[execute_phase_three] Error Level 1: {e1}", exc_info=True)


import queue
import traceback

def _run_3_tasks_and_wait(
    *,
    sensor_values,
    message,
    dates,
    input_list,
    output_list,
    p3_1_log,
    scope: str,
    scope_id,
    # correlation
    corr_group_by_output_stock: bool,
    corr_algorithm: str = "SPEARMAN",
    # prediction
    pred_group_by_stock: bool,
    pred_algorithm: str = "LSTM",
    pred_lookback: int = 20,
    pred_epochs: int = 50,
    pred_min_train_points: int = 250,
    # feature importance
    fi_algorithm: str = "XGBOOST",
    fi_produce_col: str = "prSt",
    fi_good_col: str = "good",
    fi_drop_cols=("ts", "joOpId", "wsId"),
):
    """
    Runs (1) correlation, (2) realtime prediction, (3) feature importance in parallel threads.
    Waits until all complete. Exceptions are logged; main loop continues safely.
    """
    err_q = queue.Queue()

    def _wrap(name, fn):
        try:
            fn()
        except Exception as e:
            tb = traceback.format_exc()
            err_q.put((name, str(e), tb))

    # ---- task 1: correlation ----
    def _task_correlation():
        compute_correlation(
            sensor_values,
            message,
            p3_1_log=p3_1_log,
            algorithm=corr_algorithm,
            scope=scope,
            scope_id=scope_id,
            group_by_output_stock=corr_group_by_output_stock,
        )

    # ---- task 2: realtime prediction ----
    def _task_prediction():
        try:
            from thread.phase_3_correlation._3_3_predictions import (
                handle_realtime_prediction, history_from_fetch
            )
        except Exception as e:
            p3_1_log.warning(
                f"[prediction] import failed; skipping realtime prediction: {e}",
                exc_info=True,
            )
            _m_inc("pred_skip_import_error", 1)
            return

        if scope == "pid":
            op_tc = message.get("operationtaskcode") or message.get("opTc")
            stock_no = message.get("output_stock_no") or (
                (message.get("prodList") or [{}])[0].get("stNo")
                if isinstance(message.get("prodList"), list) else None
            )

            # DATASET: RAW_TABLE’da opTc + stock (+ ws) eşleşen tüm rowlar
            op_tc_s = str(op_tc) if op_tc not in (None, "", "None") else None
            st_s    = str(stock_no) if stock_no not in (None, "", "None", "nan", "NaN") else None

            seed = None
            if op_tc_s and st_s:
                ck = _seed_cache_key(message, op_tc_s, st_s)
                seed = _seed_cache_get(ck)
                if M1_SEED_CACHE_ENABLED:
                    if seed is not None:
                        _m_inc("seed_cache_hit", 1)
                        p3_1_log.info(f"[prediction] seed_cache HIT key={ck} points={len(seed)}")
                    else:
                        _m_inc("seed_cache_miss", 1)
                        p3_1_log.info(f"[prediction] seed_cache MISS key={ck}")

            if seed is None:
                _, dates_rt, _, sv_rt = fetch_latest_for_optc_stock_via_raw_table(
                    op_tc=op_tc_s,
                    stock_no=st_s,
                    max_rows=20000,
                    ws_id=message.get("wsId"),
                )

                if not sv_rt:
                    p3_1_log.warning(
                        f"[prediction] RAW_TABLE returned 0 rows for opTc={op_tc}, stock={stock_no}. "
                        f"Falling back to in-memory sensor_values (mix risk)."
                    )
                    _, hist_out = history_from_fetch(dates, sensor_values)
                else:
                    p3_1_log.info(
                        f"[prediction] Fetched {len(dates_rt)} time points from RAW_TABLE for "
                        f"opTc={op_tc}, stock={stock_no}."
                    )
                    _, hist_out = history_from_fetch(dates_rt, sv_rt)

                seed = hist_out or []
                # store cache (bounded)
                if op_tc_s and st_s:
                    _seed_cache_put(_seed_cache_key(message, op_tc_s, st_s), seed)

            # seed_history: OUTPUT-only (from cache or RAW_TABLE fallback)
            seed = seed or []


            res = handle_realtime_prediction(
                message=message,
                lookback=(pred_lookback // 2),
                epochs=pred_epochs,
                min_train_points=pred_min_train_points,
                p3_1_log=p3_1_log,
                algorithm=pred_algorithm,
                seed_history=seed,
                scope="pid",
                scope_id=scope_id,
                group_by_stock=True
            )

        else:
            # WS tarafı: aynı mantık, yine sadece OUTPUT history seed
            _, hist_out = history_from_fetch(dates, sensor_values)
            seed = hist_out or []

            res = handle_realtime_prediction(
                message=message,
                lookback=pred_lookback,
                epochs=pred_epochs,
                min_train_points=pred_min_train_points,
                p3_1_log=p3_1_log,
                algorithm=pred_algorithm,
                seed_history=seed,
                scope=scope,
                scope_id=scope_id,
                group_by_stock=pred_group_by_stock,
            )
        
        # res has been set by either PID or WS branch above
        try:
            if isinstance(res, dict) and res.get("ok"):
                _m_inc("pred_ok", 1)
            else:
                # Common benign skip: no outputs
                if isinstance(res, dict) and res.get("reason") == "no_outputs":
                    _m_inc("pred_skip_no_outputs", 1)
                else:
                    _m_inc("pred_fail", 1)
        except Exception:
            _m_inc("pred_fail", 1)

        p3_1_log.info(f"[execute_phase_three] {scope}-level realtime_prediction status={res}")


    # ---- task 3: feature importance ----
    def _task_feature_importance():
        try:
            from thread.phase_3_correlation._3_5_feature_importance import (
                compute_and_save_feature_importance
            )
        except Exception as e:
            p3_1_log.warning(
                f"[feature_importance] import failed; skipping: {e}",
                exc_info=True,
            )
            _m_inc("fi_skip_import_error", 1)
            return

        compute_and_save_feature_importance(
            sensor_values,
            message,
            input_feature_names=input_list,
            output_feature_names=output_list,
            produce_col=fi_produce_col,
            good_col=fi_good_col,
            drop_cols=fi_drop_cols,
            algorithm=fi_algorithm,
            p3_1_log=p3_1_log,
            scope=scope,
            scope_id=scope_id,
        )


    corr_cd = int(getattr(cfg, "correlation_cooldown_sec", 60))
    fi_cd   = int(getattr(cfg, "feature_importance_cooldown_sec", 300))
    pred_cd = int(getattr(cfg, "prediction_cooldown_sec", 10))

    cust = message.get("customer") or message.get("cust") or "UNKNOWN_CUSTOMER"
    sid  = scope_id if scope_id not in (None, "", "None") else "UNKNOWN_SCOPE"
    cool_key = f"{cust}|{scope}|{sid}"

    run_corr = _cooldown_ok("correlation", cool_key, corr_cd)
    run_fi   = _cooldown_ok("feature_importance", cool_key, fi_cd)
    run_pred = _cooldown_ok("realtime_prediction", cool_key, pred_cd)


    # Start 3 threads
    threads = []
    scheduled = set()

    if run_corr:
        scheduled.add("correlation")
        _m_inc("corr_run", 1)
        threads.append(threading.Thread(target=lambda: _wrap("correlation", _task_correlation), daemon=True))
    else:
        _m_inc("corr_skip_cooldown", 1)
        p3_1_log.info(f"[execute_phase_three] cooldown: skip correlation scope={scope} scope_id={scope_id}")

    if run_pred:
        scheduled.add("realtime_prediction")
        _m_inc("pred_run", 1)
        threads.append(threading.Thread(target=lambda: _wrap("realtime_prediction", _task_prediction), daemon=True))
    else:
        _m_inc("pred_skip_cooldown", 1)
        p3_1_log.info(f"[execute_phase_three] cooldown: skip realtime_prediction scope={scope} scope_id={scope_id}")

    if run_fi:
        scheduled.add("feature_importance")
        _m_inc("fi_run", 1)
        threads.append(threading.Thread(target=lambda: _wrap("feature_importance", _task_feature_importance), daemon=True))
    else:
        _m_inc("fi_skip_cooldown", 1)
        p3_1_log.info(f"[execute_phase_three] cooldown: skip feature_importance scope={scope} scope_id={scope_id}")

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    # Log any errors (don’t crash the main loop)
    had_err = False
    failed = set()
    while not err_q.empty():
        had_err = True
        name, msg, tb = err_q.get()
        failed.add(name)
        if name == "correlation":
            _m_inc("corr_fail", 1)
        elif name == "feature_importance":
            _m_inc("fi_fail", 1)
        elif name == "realtime_prediction":
            _m_inc("pred_fail", 1)
        p3_1_log.error(f"[parallel_task:{name}] failed: {msg}\n{tb}")

    if "correlation" in scheduled and "correlation" not in failed:
        _m_inc("corr_ok", 1)
    if "feature_importance" in scheduled and "feature_importance" not in failed:
        _m_inc("fi_ok", 1)

    p3_1_log.info(
        f"[execute_phase_three] Parallel tasks done for scope={scope} scope_id={scope_id} "
        f"(elapsed={elapsed:.3f}s, errors={had_err})"
    )



def execute_phase_three_worker() -> None:
    """
    Phase3 worker (queue mode):
    - consumes enriched messages from runtime.event_bus (fed by kafka_fanout)
    - runs the same Phase3 logic (buffers + Mode B gating + heavy workloads)
    - does NOT touch Kafka or RAW persistence
    """
    import queue as _queue

    p3_1_log.info("[execute_phase_three_worker] started (queue mode)")

    # Keep the same DW snapshot warmup used in execute_phase_three (used for seeding / compatibility paths)
    try:
        rows, input_list, output_list, _batch = _try_fetch_dw_snapshot(limit=20_000, p3_1_log=p3_1_log)
    except Exception:
        rows, input_list, output_list, _batch = [], [], [], []

    while True:
        try:
            message = dequeue_phase3(timeout=0.5)
        except _queue.Empty:
            continue

        try:
            # ---- M1.8: mode + allowlist gating (after durability boundary) ----
            ws_uid = _phase3_get_ws_uid(message)

            if PHASE3_MODE == "persist_only":
                _m_inc("persist_only_msg", 1)
                _m_inc("heavy_skip_reason_persist_only", 1)
                _maybe_log_metrics(p3_1_log, time.time())
                continue

            if PHASE3_WORKSTATION_ALLOWLIST and ws_uid not in PHASE3_WORKSTATION_ALLOWLIST:
                _m_inc("allowlist_skip", 1)
                _m_inc("heavy_skip_reason_allowlist", 1)
                now_epoch = time.time()
                last = _PHASE3_LAST_SKIP_LOG.get(ws_uid, 0.0)
                if (now_epoch - last) >= PHASE3_SKIP_LOG_EVERY_SEC:
                    _PHASE3_LAST_SKIP_LOG[ws_uid] = now_epoch
                    p3_1_log.info(f"[phase3] persist_only_for_ws ws={ws_uid} (not in allowlist)")
                _maybe_log_metrics(p3_1_log, now_epoch)
                continue


            # Belirli müşteri skip
            """if message.get("outVals") and message["outVals"][0].get("cust") == "teknia_group":
                p3_1_log.info("[execute_phase_three] Skipping teknia_group customer")
                continue

            if message.get("plId") and message["plId"] == 20:
                p3_1_log.info("[execute_phase_three] Skipping Savola customer")
                continue

            if message.get("plId") and message["plId"] == 162:
                p3_1_log.info("[execute_phase_three] Skipping Meriç customer")
                continue

            if message.get("prodList") and message["prodList"][0].get("stNo") == "Antares PV":
                p3_1_log.info("[execute_phase_three] Skipping Antares PV stock")
                continue

            if message.get("prodList") and message["prodList"][0].get("stNo") == "Loperamide 2 mg granulate":
                p3_1_log.info("[execute_phase_three] Skipping Loperamide 2 mg granulate stock")
                continue"""

            plant = message.get("plId")
            wsSt = message.get("prSt")
            pid = message.get("joOpId") #
            ws_id = message.get("wsId")
            crDt_msg = _to_dt(message.get("crDt"))
            now_epoch = time.time()

            pid_key = _pid_buffer_key(message, pid)

            cnt = buffer_for_process.get(pid_key, {}).get("count", 0)
            p3_1_log.info(
                f"[execute_phase_three] Initializing for -- plId={plant}, prod={message.get('refNo')}, "
                f"pid={pid}, pid_key={pid_key}, wsId={ws_id}, wsSt={wsSt}, "
                f"In Buffer={pid_key in buffer_for_process}, count={cnt}"
            )

            # --- PID buffer yönetimi ---
            if wsSt == "PRODUCTION" and pid is not None:
                if pid_key not in buffer_for_process:
                    buffer_for_process[pid_key] = {
                        "count": 1,
                        "first_seen": now_epoch,
                        "last_time": now_epoch,
                        "first_crdt": crDt_msg,
                        "messages": [message],
                    }
                    p3_1_log.info(f"[execute_phase_three] Initialized PID buffer for pid_key={pid_key}")
                else:
                    b = buffer_for_process[pid_key]
                    b["count"] = b.get("count", 0) + 1
                    b["last_time"] = now_epoch
                    if b.get("first_crdt") is None and crDt_msg:
                        b["first_crdt"] = crDt_msg
                    b.setdefault("messages", []).append(message)

            # --- WS buffer yönetimi ---
            if wsSt == "PRODUCTION" and ws_id is not None:
                if ws_id not in buffer_for_ws:
                    buffer_for_ws[ws_id] = {
                        "count": 1,
                        "first_seen": now_epoch,
                        "last_time": now_epoch,
                        "first_crdt": crDt_msg,
                        "messages": [message],
                    }
                    p3_1_log.info(f"[execute_phase_three] Initialized WS buffer for wsId={ws_id}")
                else:
                    w = buffer_for_ws[ws_id]
                    w["count"] = w.get("count", 0) + 1
                    w["last_time"] = now_epoch
                    if w.get("first_crdt") is None and crDt_msg:
                        w["first_crdt"] = crDt_msg
                    w.setdefault("messages", []).append(message)

            # --- PID / WS için threshold check ---
            if wsSt == "PRODUCTION" and pid is not None and pid_key in buffer_for_process:

                b = buffer_for_process[pid_key]
                fire_pid, pid_cnt, pid_time_lapsed = _should_fire(b, PID_COUNT_THRESHOLD, PID_TIME_THRESHOLD, now_epoch)

                p3_1_log.info(
                    f"[execute_phase_three] pid={pid}, pid_key={pid_key}, wsId={ws_id}, wsSt={wsSt} "
                    f"Updated PID buffer count={pid_cnt}, time_lapsed={pid_time_lapsed}"
                )

                if fire_pid:
            # ---- M1.8 / ModeB gate ----
                    if not _phase3_should_run_heavy(message, scope="pid", scope_id=pid, now_epoch=now_epoch):
                        _m_inc("heavy_pid_skip", 1)
                        p3_1_log.info(
                            f"[phase3] heavy_skip scope=pid mode={PHASE3_MODE} ws={_phase3_get_ws_uid(message)} pid={pid} "
                            f"(every_sec={PHASE3_HEAVY_EVERY_SEC}, key_scope={PHASE3_HEAVY_KEY_SCOPE})"
                        )
                        # reset PID buffer to prevent repeated firing
                        b["count"] = 0
                        b["first_seen"] = b["last_time"]
                        b["first_crdt"] = None
                        b["messages"] = []
                    else:
                        _m_inc("heavy_pid_run", 1)

                        # Extract current message's op_tc and stock
                        op_tc = message.get("operationtaskcode") or message.get("opTc")
                        stock_no = message.get("output_stock_no")
                        ws_id_current = message.get("wsId")

                        # Clean them
                        op_tc = None if op_tc in (None, "", "None") else str(op_tc)
                        stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)

                        p3_1_log.info(
                            f"[execute_phase_three] PID={pid} FETCHING DATA for op_tc={op_tc}, "
                            f"stock={stock_no}, ws_id={ws_id_current}"
                        )

                        # === (1) DW SNAPSHOT - by op_tc + stock (NOT by PID!) ===
                        job_ids_pid, dates_pid, ids_pid, sensor_values_pid = fetch_for_optc_stock_via_dw(
                            rows, input_list, output_list, _batch,
                            op_tc=op_tc,
                            stock_no=stock_no,
                            ws_id=ws_id_current,
                            pid=None,
                            max_rows=20_000
                        )

                        # === (2) RAW_TABLE FALLBACK ===
                        if not sensor_values_pid:
                            p3_1_log.info(
                                f"[execute_phase_three] pid={pid} — DW returned 0 bundles; trying RAW_TABLE fallback"
                            )
                            job_ids_pid, dates_pid, ids_pid, sensor_values_pid = fetch_latest_for_optc_stock_via_raw_table(
                                op_tc=op_tc,
                                stock_no=stock_no,
                                ws_id=ws_id_current,
                                pid=None,
                                max_rows=20_000
                            )

                        # === (3) BUFFER FALLBACK (all PID buffers) ===
                        if not sensor_values_pid:
                            all_buffered_messages = []
                            for _, buf_entry in buffer_for_process.items():
                                all_buffered_messages.extend(buf_entry.get("messages", []))

                            buffered_n = len(all_buffered_messages)
                            p3_1_log.info(
                                f"[execute_phase_three] pid={pid} — RAW_TABLE returned 0 bundles; trying buffer fallback "
                                f"(total_buffered_messages={buffered_n})"
                            )

                            job_ids_pid, dates_pid, ids_pid, sensor_values_pid = build_sensor_values_from_buffer_for_optc_stock(
                                all_buffered_messages,
                                op_tc=op_tc,
                                stock_no=stock_no,
                                ws_id=ws_id_current,
                                pid=None
                            )

                        if not sensor_values_pid:
                            p3_1_log.info(
                                f"[execute_phase_three] pid={pid} — still no data after DW+RAW+buffer "
                                f"for op_tc={op_tc}, stock={stock_no}; skipping"
                            )
                        else:
                            p3_1_log.info(
                                f"[execute_phase_three] Fetched {len(dates_pid)} time points "
                                f"for op_tc={op_tc}, stock={stock_no} (from multiple PIDs)"
                            )

                            try:
                                _run_3_tasks_and_wait(
                                    sensor_values=sensor_values_pid,
                                    message=message,
                                    dates=dates_pid,
                                    input_list=input_list,
                                    output_list=output_list,
                                    p3_1_log=p3_1_log,
                                    scope="pid",
                                    scope_id=pid,
                                    corr_group_by_output_stock=False,
                                    pred_group_by_stock=True,
                                    pred_algorithm="RANDOM_FOREST",
                                    pred_epochs=125,
                                    pred_min_train_points=250,
                                    fi_algorithm="XGBOOST",
                                )
                            except Exception as e:
                                p3_1_log.error(f"[execute_phase_three] PID parallel runner failed for pid={pid}: {e}", exc_info=True)

            # ---- PID buffer reset policy ----
                        b["count"] = 0
                        b["first_seen"] = b["last_time"]
                        b["first_crdt"] = None
                        b["messages"] = []

                else:
                    _m_inc("corr_skip_buf_pid", 1)
                    _m_inc("heavy_skip_reason_insufficient_window", 1)
                    _m_inc("heavy_skip_reason_insufficient_window_pid", 1)
                    p3_1_log.info(
                        f"[execute_phase_three] Pid={pid}, wsId={ws_id}, wsSt={wsSt} "
                        f"SKIPPING PID-level Correlation (cnt={pid_cnt}, elapsed={pid_time_lapsed})"
                    )

            # ---------- WS-LEVEL CHECK (AGGREGATED) ----------
                if ws_id is not None and ws_id in buffer_for_ws:
                    w = buffer_for_ws[ws_id]
                    fire_ws, ws_cnt, ws_time_lapsed = _should_fire(
                        w, WS_COUNT_THRESHOLD, WS_TIME_THRESHOLD, now_epoch
                    )

                    p3_1_log.info(
                        f"[execute_phase_three] wsId={ws_id}, wsSt={wsSt} "
                        f"Updated WS buffer count={ws_cnt}, time_lapsed={ws_time_lapsed}"
                    )

                    if fire_ws:
            # ---- M1.8: sampled_full gate for WS scope ----
                        if not _phase3_should_run_heavy(message, scope="ws", scope_id=ws_id, now_epoch=now_epoch):
                            _m_inc("heavy_ws_skip", 1)
                            p3_1_log.info(
                                f"[phase3] sampled_skip scope=ws ws={_phase3_get_ws_uid(message)} wsId={ws_id} "
                                f"(every_sec={PHASE3_HEAVY_EVERY_SEC}, key_scope={PHASE3_HEAVY_KEY_SCOPE})"
                            )
                            # reset WS buffer to prevent repeated firing
                            w["count"] = 0
                            w["first_seen"] = w["last_time"]
                            w["first_crdt"] = None
                            w["messages"] = []
                            continue  # skip rest of WS heavy block
                        else:
                            _m_inc("heavy_ws_run", 1)

                        p3_1_log.info(
                            f"[execute_phase_three] Fetching WS-level raw data for wsId={ws_id} "
                            f"(cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                        )

                        # (1) DW SNAPSHOT
                        job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_dw(
                            ws_id, rows, input_list, output_list, _batch, max_rows=20_000
                        )

                        # (2) RAW_TABLE FALLBACK
                        if not sensor_values_ws:
                            p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — DW returned 0 bundles; trying RAW_TABLE fallback")
                            job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_raw_table(
                                ws_id, max_rows=20_000
                            )

                        # (3) WS BUFFER FALLBACK
                        if not sensor_values_ws:
                            buffered_n = len(w.get("messages", []))
                            p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — RAW_TABLE returned 0 bundles; trying WS buffer fallback (buffered_messages={buffered_n})")
                            job_ids_ws, dates_ws, ids_ws, sensor_values_ws = build_sensor_values_from_ws_buffer(
                                ws_id, w.get("messages", [])
                            )

                        if not sensor_values_ws:
                            p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — still no data after DW+RAW+buffer; skipping")
                            continue
                        else:
            # ---- STOCK GATE (WS + group_by_output_stock) ----
                            skip_ws = False
                            current_stock = _clean_stock(get_stock_key(message, default=""))

                            if current_stock is None:
                                p3_1_log.warning(
                                    f"[execute_phase_three] wsId={ws_id} group_by_output_stock=True but current message has no output_stock_no. "
                                    "Skipping WS correlation/LSTM/feature-importance to avoid mixing stocks."
                                )
                                skip_ws = True
                            else:
                                has_stock = _sensor_values_has_stock(sensor_values_ws, current_stock)
                                pts = _count_points_for_stock(sensor_values_ws, current_stock)

                                if (not has_stock) or (pts < 2):
                                    p3_1_log.info(
                                        f"[execute_phase_three] wsId={ws_id} stock gate: current_stock={current_stock} "
                                        f"has_stock={has_stock} points={pts} (<2 or missing). "
                                        "Skipping WS correlation/LSTM/feature-importance; waiting for next message."
                                    )
                                    skip_ws = True

                            if not skip_ws:
                                # --- WS: run correlation + prediction + feature-importance in parallel ---
                                # Mode B: DO NOT consume triggers before the heavy run succeeds.
                                ws_uid0 = _phase3_get_ws_uid(message)
                                stk0 = current_stock or "ALL"

                                try:
                                    _run_3_tasks_and_wait(
                                        sensor_values=sensor_values_ws,   # (NOTE) use what you already feed to correlation/fi
                                        message=message,
                                        dates=dates_ws,
                                        input_list=input_list,
                                        output_list=output_list,
                                        p3_1_log=p3_1_log,
                                        scope="ws",
                                        scope_id=ws_id,
                                        corr_group_by_output_stock=True,  # keep your WS behavior
                                        pred_group_by_stock=True,         # keep your WS behavior
                                        pred_algorithm="RANDOM_FOREST",
                                        pred_epochs=125,
                                        pred_min_train_points=250,
                                        fi_algorithm="XGBOOST",
                                    )
                                    # Heavy run succeeded: consume any active anomaly trigger + reset baseline timer.
                                    trig_reason = consume_anomaly(ws_uid0, stk0, scope="ws")
                                    if trig_reason is None and stk0 != "ALL":
                                        trig_reason = consume_anomaly(ws_uid0, "ALL", scope="ws")
                                    if trig_reason:
                                        p3_1_log.info(f"[phase3] mode_b_triggered ws={ws_uid0} stock={stk0} reason={trig_reason}")
                                    if PHASE3_MODEB_BASELINE_EVERY_SEC > 0:
                                        mark_baseline_run(ws_uid0, stk0, scope="ws", now_epoch=now_epoch)
                                except Exception as e:
                                    p3_1_log.error(f"[execute_phase_three] WS parallel runner failed for wsId={ws_id}: {e}", exc_info=True)

            # ---- WS buffer reset policy ----
                            if not skip_ws:
                                w["count"] = 0
                                w["first_seen"] = w["last_time"]
                                w["first_crdt"] = None
                                w["messages"] = []
                            else:
                                # keep buffer so next message can re-trigger and fetch includes the stock
                                pass

                    else:
                        _m_inc("corr_skip_buf_ws", 1)
                        _m_inc("heavy_skip_reason_insufficient_window", 1)
                        _m_inc("heavy_skip_reason_insufficient_window_ws", 1)
                        p3_1_log.info(
                            f"[execute_phase_three] wsId={ws_id}, wsSt={wsSt} "
                            f"SKIPPING WS-level Correlation (cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                        )

            # PRODUCTION değilse PID / WS bufferları temizle
            if wsSt != "PRODUCTION":
                p3_1_log.info(
                    f"[execute_phase_three] pid={pid}, pid_key={pid_key}, wsId={ws_id}, ws not in PRODUCTION, "
                    f"clearing buffers, wsSt={wsSt}"
                )
                if pid is not None:
                    buffer_for_process.pop(pid_key, None)
                if ws_id is not None:
                    buffer_for_ws.pop(ws_id, None)

            _maybe_log_metrics(p3_1_log, now_epoch)

        except Exception as e:
            p3_1_log.error(f"[execute_phase_three_worker] error: {e}", exc_info=True)
        finally:
            try:
                task_done_phase3()
            except Exception:
                pass
