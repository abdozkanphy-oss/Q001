# thread/phase_3_correlation/_3_2_correlations.py
import json
import time
import uuid
import threading
import queue
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy, TokenAwarePolicy
from cassandra import ConsistencyLevel
from cassandra.auth import PlainTextAuthProvider

from utils.config_reader import ConfigReader
from utils.logger_2 import setup_logger
from modules.kafka_modules import kafka_consumer3

from cassandra_utils.models.dw_single_data import dw_tbl_raw_data
from cassandra_utils.models.scada_correlation_matrix import ScadaCorrelationMatrix
from cassandra_utils.models.scada_correlation_matrix_summary import ScadaCorrelationMatrixSummary

from thread.phase_3_correlation._3_1_helper_functions import (
    compute_correlation, aggregate_correlation_data
)
from thread.phase_3_correlation._3_3_predictions import (
    handle_realtime_prediction, history_from_fetch
)
from thread.phase_3_correlation._3_5_feature_importance import (
    compute_and_save_feature_importance
)

# ---------------- config & logger ----------------
cfg = ConfigReader()
p3_1_log = setup_logger("p3_1_logger", "logs/p3_1.log")

# How many worker threads inside the orchestrator (defaults to 4 if missing)
PHASE3_WORKERS = 4

# ---------------- cassandra session ---------------
cass_cfg = cfg["cassandra"]
auth_provider = PlainTextAuthProvider(
    username=cass_cfg["username"], password=cass_cfg["password"]
)
profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy(local_dc=None)),
    request_timeout=20.0,
    consistency_level=ConsistencyLevel.LOCAL_ONE,
    retry_policy=RetryPolicy(),
)
cluster = Cluster(
    [cass_cfg["host"]],
    auth_provider=auth_provider,
    execution_profiles={EXEC_PROFILE_DEFAULT: profile},
)
session = cluster.connect(cass_cfg["keyspace"])
session.set_keyspace(cass_cfg["keyspace"])

# ---------------- kafka consumer -----------------
_consumer3 = kafka_consumer3()

# ---------------- orchestrator state -------------
_JOBS_Q: "queue.Queue[dict]" = queue.Queue(maxsize=1)
_STOP = object()
_pid_locks = defaultdict(threading.Lock)

buffer_for_process = {}   # per-PID counters for correlation windowing
COUNT_THRESHOLD = 20
TIME_THRESHOLD  = 900  # seconds

# Singleton guard so only one orchestrator is active
_orchestrator_started = threading.Event()

# ---------------- DW cache (preload) -------------
DW_CACHE = {
    "rows": [],
    "input_cols": [],
    "output_list": [],
    "batch": [],
    "last_refresh": None,
}

def _preload_dw(limit: int = 20_000):
    p3_1_log.info("[phase3] Preloading DW data...")
    rows, _input, output_list, _batch = dw_tbl_raw_data.fetchData(limit=limit)
    DW_CACHE.update({
        "rows": rows,
        "input_cols": _input,
        "output_list": output_list,
        "batch": _batch,
        "last_refresh": datetime.now(timezone.utc),
    })
    p3_1_log.info(
        f"[phase3] DW preload ok: rows={len(rows)}, inputs={len(_input)}, outputs={len(output_list)}"
    )

# ---------------- small utils --------------------
def _to_dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    s = str(v)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        try:
            iv = int(float(s))  # epoch ms
            return datetime.fromtimestamp(iv/1000.0, tz=timezone.utc)
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

# ------------- DW slice per PID ------------------
def _dw_slice_for_pid(pid: int):
    rows = DW_CACHE["rows"] or []
    output_list = DW_CACHE["output_list"] or []

    bucket = defaultdict(list)
    for r, wrapped_ov in zip(rows, output_list):
        if getattr(r, "job_order_operation_id", None) != pid:
            continue
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        ov = wrapped_ov[0] if wrapped_ov else {}
        pname = ov.get("eqNo")
        cval  = ov.get("cntRead")

        bucket[ts].append({
            "parameter": str(pname),
            "counter_reading": _num_text(cval),
        })

    if not bucket:
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue
        meta = {
            "job_order_operation_id": pid,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
        }
        job_ids.append(pid)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    return job_ids, dates, ids, sensor_values

# ------------- correlation aggregation ----------
def _row_to_meta_message(row: "ScadaCorrelationMatrix") -> dict:
    prod_list = [{
        "stNm": getattr(row, "output_stock_name", None),
        "stNo": getattr(row, "output_stock_no",   None),
        "stId": getattr(row, "output_stock_no",   None),
        "cust": getattr(row, "customer",          None),
    }]
    return {
        "crDt": int(datetime.now(timezone.utc).timestamp() * 1000),
        "cust":  getattr(row, "customer",        None),
        "plId":  getattr(row, "plant_id",        None),
        "wcNm":  getattr(row, "workcenter_name", None),
        "wcNo":  getattr(row, "workcenter_no",   None),
        "wsNm":  getattr(row, "workstation_name",None),
        "wsNo":  getattr(row, "workstation_no",  None),
        "opNm":  getattr(row, "operator_name",   None),
        "opNo":  getattr(row, "operator_no",     None),
        "prodList": prod_list,
    }

def aggregate_correlation(message, stid):
    rows = ScadaCorrelationMatrix.fetchData(str(stid)) or []
    corr_list = [getattr(r, "correlation_data", None) for r in rows if getattr(r, "correlation_data", None)]
    if not corr_list:
        p3_1_log.warning("[aggregate_correlation] No correlation_data found; skipping")
        return
    try:
        base_row = max(
            rows,
            key=lambda x: getattr(x, "partition_date", None) or datetime.min.replace(tzinfo=timezone.utc)
        )
    except ValueError:
        base_row = rows[-1]
    aggregated = aggregate_correlation_data(corr_list, p3_1_log=p3_1_log)
    meta_msg = _row_to_meta_message(base_row)
    ScadaCorrelationMatrixSummary.saveData(meta_msg, aggregated, p3_1_log=p3_1_log)

# ------------- threshold state per PID ----------
def _advance_pid_window(pid: int, crDt_msg: datetime):
    b = buffer_for_process.setdefault(pid, {
        "count": 0,
        "first_seen": time.time(),
        "last_time":  time.time(),
        "first_crdt": crDt_msg
    })
    now = time.time()
    b["count"] += 1
    b["last_time"] = now
    if b.get("first_crdt") is None and crDt_msg:
        b["first_crdt"] = crDt_msg
    do_compute = (b["count"] >= COUNT_THRESHOLD) or (now - b["first_seen"] >= TIME_THRESHOLD)
    if do_compute:
        start_dt, end_dt = b.get("first_crdt"), crDt_msg
        b["count"] = 0
        b["first_seen"] = b["last_time"]
        b["first_crdt"] = None
        return True, start_dt, end_dt
    return False, None, None

# ------------- per-message processing ----------
def _process_message(msg: dict):
    wsSt = msg.get("prSt")
    pid  = msg.get("joOpId")
    crDt = _to_dt(msg.get("crDt"))

    if pid is None:
        p3_1_log.warning("[worker] Skipping message with no joOpId")
        return

    with _pid_locks[pid]:
        # correlation/aggregation windowing
        if wsSt == "PRODUCTION" and pid not in buffer_for_process:
            buffer_for_process[pid] = {
                "count": 1, "first_seen": time.time(), "last_time": time.time(), "first_crdt": crDt
            }
            p3_1_log.info(f"[phase3] Initialized buffer for pid={pid}")
        elif wsSt == "PRODUCTION":
            do_compute, start_dt, end_dt = _advance_pid_window(pid, crDt)
            if do_compute:
                job_ids, dates, ids, sensor_values = _dw_slice_for_pid(pid)
                if sensor_values:
                    try:
                        compute_correlation(sensor_values, msg, p3_1_log=p3_1_log)
                    except Exception as e:
                        p3_1_log.error(f"[phase3] Correlation failed for pid={pid}: {e}", exc_info=True)
                    prod_stk_id = next((it.get("stId") for it in (msg.get("prodList") or []) if it.get("stId")), None)
                    if prod_stk_id:
                        try:
                            aggregate_correlation(msg, prod_stk_id)
                        except Exception as e:
                            p3_1_log.error(f"[phase3] Aggregation failed for pid={pid}: {e}", exc_info=True)
        else:
            buffer_for_process.pop(pid, None)

        # realtime LSTM
        try:
            job_ids, dates, ids, sensor_values = _dw_slice_for_pid(pid)
            seed = history_from_fetch(dates, sensor_values) if sensor_values else None
            handle_realtime_prediction(
                message=msg, lookback=20, epochs=3, min_train_points=120,
                p3_1_log=p3_1_log, seed_history=seed
            )
        except Exception as e:
            p3_1_log.warning(f"[rt_pred] skipped: {e}", exc_info=True)

        # feature importance
        try:
            _, _, _, slice_values = _dw_slice_for_pid(pid)
            compute_and_save_feature_importance(
                slice_values, msg,
                input_feature_names=DW_CACHE["input_cols"],
                output_feature_names=DW_CACHE["output_list"],
                produce_col="prSt", good_col="good",
                drop_cols=("ts","joOpId","wsId"),
                algorithm="rf_perm",
                p3_1_log=p3_1_log
            )
        except Exception as e:
            p3_1_log.warning(f"[feature_importance] skipped: {e}", exc_info=True)

# ------------- producer / workers --------------
# globals (once)
_STOP_EVENT = threading.Event()
_JOBS_Q = queue.Queue(maxsize=1)  # strict backpressure

def _producer_loop():
    p3_1_log.info("[kafka] producer loop started")
    try:
        while not _STOP_EVENT.is_set():
            # fetch exactly one message; block the queue until workers free it
            msg = _consumer3.poll(timeout=2.0)
            if msg is None:
                continue
            if getattr(msg, "error", None) and msg.error():
                p3_1_log.error(f"[kafka] message error: {msg.error()}")
                continue
            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except Exception as e:
                p3_1_log.error(f"[kafka] bad message decode: {e}", exc_info=True)
                continue
            p3_1_log.info(f"[kafka] received message: {payload}")
            _JOBS_Q.put(payload)  # BLOCKS; no timeout → no queue.Full
    except Exception as e:
        p3_1_log.error(f"[kafka] producer loop error: {e}", exc_info=True)
    finally:
        p3_1_log.info("[kafka] producer loop exiting")

def _worker_loop(name):
    p3_1_log.info(f"[worker-{name}] started")
    while not _STOP_EVENT.is_set():
        try:
            job = _JOBS_Q.get()  # BLOCKS until there is work
            try:
                # your existing per-message processing here
                _process_message(job)  # wraps correlation, aggregate, prediction, feature-importance
            except Exception as e:
                p3_1_log.error(f"[worker-{name}] job failed: {e}", exc_info=True)
            finally:
                _JOBS_Q.task_done()
        except Exception as e:
            p3_1_log.error(f"[worker-{name}] loop error: {e}", exc_info=True)
    p3_1_log.info(f"[worker-{name}] exiting")

def execute_phase_three():  # <-- this is the single method your main submits
    p3_1_log.info("[phase3] Initializing Phase 3 (orchestrator)")

    # 1) Preload DW data once (kept in memory for workers to use)
    p3_1_log.info("[phase3] Preloading DW data...")
    rows, _input, output_list, _batch = dw_tbl_raw_data.fetchData(limit=20_000)
    p3_1_log.info(f"[phase3] DW preload ok: rows={len(rows)}, inputs={len(_input)}, outputs={len(output_list)}")

    # 2) Start producer + workers (non-daemon so they keep the process alive)
    p3_1_log.info(f"[phase3] starting {cfg['execute_phase_three_thread']} workers")
    prod = threading.Thread(target=_producer_loop, name="phase3-producer", daemon=False)
    prod.start()

    workers = []
    for i in range(int(cfg['execute_phase_three_thread'])):
        t = threading.Thread(target=_worker_loop, args=(i,), name=f"phase3-worker-{i}", daemon=False)
        t.start()
        workers.append(t)

    # 3) BLOCK here so this function does not return (prevents re-invocation)
    try:
        while True:
            time.sleep(3600)  # or _STOP_EVENT.wait(3600)
    except KeyboardInterrupt:
        p3_1_log.info("[phase3] shutdown requested")
    finally:
        _STOP_EVENT.set()
        try:
            _JOBS_Q.join()   # wait until all enqueued jobs are finished
        except Exception:
            pass
        prod.join(timeout=10)
        for t in workers:
            t.join(timeout=10)
        p3_1_log.info("[phase3] clean shutdown complete")
