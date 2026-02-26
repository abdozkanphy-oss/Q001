import json
import uuid
import time
from datetime import datetime, timezone
from collections import defaultdict

#from fastapi import params
from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore

from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy, TokenAwarePolicy
from cassandra import ConsistencyLevel

from utils.logger_2 import setup_logger
from utils.config_reader import ConfigReader
from thread.phase_3_correlation._3_6_retrain import handle_retrain_request


# -------------------- setup --------------------
cfg = ConfigReader()
cassandra_config = cfg["cassandra"]

CASSANDRA_HOST = cassandra_config["host"]
USERNAME = cassandra_config["username"]
PASSWORD = cassandra_config["password"]
KEYSPACE = cassandra_config["keyspace"]

RAW_TABLE = cfg["cassandra_props"]["raw_data_table"]  # "dw_tbl_raw_data"

p3_2_log = setup_logger("p3_2_retrain_logger", "logs/p3_2_retrain.log")

auth_provider = PlainTextAuthProvider(username=USERNAME, password=PASSWORD)
profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy(local_dc=None)),
    request_timeout=60.0,
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


# -------------------- helper functions --------------------
def _to_dt(v):
    """Convert to aware UTC datetime."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        s = int(v)
        if s > 10**12:
            return datetime.fromtimestamp(s / 1000.0, tz=timezone.utc)
        else:
            return datetime.fromtimestamp(s, tz=timezone.utc)
    except Exception:
        pass
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

def _clean_stock(v):
    if v is None:
        return None
    s = str(v)
    if s in ("", "None", "nan", "NaN"):
        return None
    return s


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

# -------------------- data fetch --------------------
def fetch_data_for_pid_optc(payload: dict, pid: str = None, max_rows: int = 20_000, include_all_pids: bool = True) -> dict:
    """
    PID-based: önce pid için operationtaskcode'u bul, sonra aynı opTc + stock_no olan tüm rowları çek.
    
    Args:
        payload: Request payload
        pid: İlgili PID (opTc bulmak için)
        max_rows: Maksimum row sayısı
        include_all_pids: True ise aynı opTc+stock'taki TÜM PID'lerin verisini çeker (retrain için)
                         False ise sadece belirtilen PID'in verisini çeker
    """
    # Hangi PID'den opTc çekeceğimizi belirle
    target_pid = pid if pid is not None else payload.get("joOpId")
    
    if target_pid is None:
        p3_2_log.warning(f"[fetch_data_for_pid_optc] No PID provided")
        return {}
    
    # 1) Bu PID'den operationtaskcode'u çek
    op_tc = payload.get("operationtaskcode") or None
    if op_tc is None:
        try:
            op_tc_query = f"""
                SELECT operationtaskcode 
                FROM {RAW_TABLE} 
                WHERE job_order_operation_id = %s 
                LIMIT 1 ALLOW FILTERING
            """
            op_tc_rows = list(session.execute(op_tc_query, (int(target_pid),)))
            
            if not op_tc_rows:
                p3_2_log.warning(f"[fetch_data_for_pid_optc] No operationtaskcode found for pid={target_pid}")
                return {}
            
            op_tc = getattr(op_tc_rows[0], "operationtaskcode", None)
            
            if not op_tc:
                p3_2_log.warning(f"[fetch_data_for_pid_optc] operationtaskcode is null for pid={target_pid}")
                return {}
                
            p3_2_log.info(f"[fetch_data_for_pid_optc] Found operationtaskcode={op_tc} for pid={target_pid}")
            
        except Exception as e:
            p3_2_log.error(f"[fetch_data_for_pid_optc] Failed to fetch opTc for pid={target_pid}: {e}", exc_info=True)
            return {}
    
    # 2) Payload'dan stock_no al
    stock_no = _clean_stock(payload.get("stNo"))
    
    if not stock_no:
        p3_2_log.warning(f"[fetch_data_for_pid_optc] Missing stNo in payload")
        return {}
    
    # 3) Şimdi aynı opTc + stock_no olan TÜM rowları çek
    where_clause = "WHERE operationtaskcode = %s AND produced_stock_no = %s"
    params = [str(op_tc), str(stock_no)]
    
    # Opsiyonel: sadece aynı workstation'dan çekmek istersen
    ws_id = payload.get("wsId")
    if ws_id is not None:
        where_clause += " AND work_station_id = %s"
        params.append(int(ws_id))
    
    # CRITICAL: Sadece retrain için değilse, belirli PID'e filtrele
    # Retrain için (include_all_pids=True) TÜM PID'lerin verisini çek
    if pid is not None and not include_all_pids:
        #where_clause += " AND job_order_operation_id = %s"
        #params.append(int(pid))
        p3_2_log.info(f"[fetch_data_for_pid_optc] Filtering for single PID={pid}")
    else:
        p3_2_log.info(f"[fetch_data_for_pid_optc] Fetching ALL PIDs with opTc={op_tc}, stock={stock_no}")
    
    query = f"""
        SELECT job_order_operation_id, work_station_id, measurement_date,
               equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type,
               good, workstation_state, job_order_reference_no, prod_order_reference_no,
               produced_stock_no, produced_stock_name, 
               operationname, operationno, operationtaskcode,
               work_center_id, work_center_name, work_center_no,
               work_station_name, work_station_no,
               plant_id, plant_name, customer
        FROM {RAW_TABLE}
        {where_clause}
        LIMIT %s ALLOW FILTERING
    """
    params.append(max_rows)
    
    try:
        rows = list(session.execute(query, tuple(params)))
        p3_2_log.info(
            f"[fetch_data_for_pid_optc] (1) Fetched {len(rows)} rows for opTc={op_tc}, stock={stock_no}, "
            f"pid={pid if not include_all_pids else 'ALL'}, include_all_pids={include_all_pids}"
        )
    except Exception as e:
        p3_2_log.error(f"[fetch_data_for_pid_optc] Query failed (1): {e}", exc_info=True)
        try:
            time.sleep(0.7)
            rows = list(session.execute(query, tuple(params)))
            p3_2_log.info(
                f"[fetch_data_for_pid_optc] (2) Fetched {len(rows)} rows for opTc={op_tc}, stock={stock_no}, "
                f"pid={pid if not include_all_pids else 'ALL'}, include_all_pids={include_all_pids}"
            )
        except Exception as e:
            p3_2_log.error(f"[fetch_data_for_pid_optc] Query failed (2): {e}", exc_info=True)
            return {}
    
    if not rows:
        return {}
    
    # Her timestamp için sensörleri grupla
    bucket = defaultdict(list)
    label_meta = {}
    pid_by_ts = {}
    
    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue
        
        if ts not in pid_by_ts:
            pid_by_ts[ts] = getattr(r, "job_order_operation_id", None)
        
        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": _clean_stock(getattr(r, "produced_stock_no", None)),
                "output_stock_name": _clean_stock(getattr(r, "produced_stock_name", None)),
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None),
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None),
                # Workcenter bilgileri
                "workcenter_name": getattr(r, "work_center_name", None),
                "workcenter_no": getattr(r, "work_center_no", None),
                "work_center_id": getattr(r, "work_center_id", None),
                # Workstation bilgileri
                "workstation_name": getattr(r, "work_station_name", None),
                "workstation_no": getattr(r, "work_station_no", None),
                # Plant bilgileri
                "plant_id": getattr(r, "plant_id", None),
                "plant_name": getattr(r, "plant_name", None),
                # Customer
                "customer": getattr(r, "customer", None)
            }
        
        # ===== NEW: Check equipment_type =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type)
        
        if is_input:
            # INPUT variable
            var_name = getattr(r, "equipment_name", None) or getattr(r, "equipment_no", None)
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
            # OUTPUT variable
            pname = getattr(r, "equipment_no", None) or getattr(r, "equipment_name", None)
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
            # Workcenter bilgileri
            "workcenter_name": lm.get("workcenter_name"),
            "workcenter_no": lm.get("workcenter_no"),
            "work_center_id": lm.get("work_center_id"),
            # Workstation bilgileri
            "workstation_name": lm.get("workstation_name"),
            "workstation_no": lm.get("workstation_no"),
            # Plant bilgileri
            "plant_id": lm.get("plant_id"),
            "plant_name": lm.get("plant_name"),
            # Customer
            "customer": lm.get("customer")
        }
        
        job_ids.append(pid_by_ts.get(ts) or 0)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])
    
    return {
        "sensor_values": sensor_values,
        "dates": dates,
        "job_ids": job_ids,
        "operationtaskcode": str(op_tc),
        "output_stock_no": str(stock_no),
    }


def extract_pids_for_prod(payload: dict, max_rows: int = 20_000) -> list:
    """
    Production order reference no'ya göre tüm distinct job_order_operation_id'leri çek
    """
    ref_no = payload.get("refNo")
    
    if not ref_no:
        p3_2_log.warning(f"[extract_pids_for_prod] Missing refNo in payload")
        return []
    
    query = f"""
        SELECT DISTINCT job_order_operation_id
        FROM {RAW_TABLE}
        WHERE prod_order_reference_no = %s
        LIMIT %s ALLOW FILTERING
    """
    
    try:
        rows = list(session.execute(query, (str(ref_no), max_rows)))
        pids = [getattr(r, "job_order_operation_id") for r in rows if getattr(r, "job_order_operation_id", None) is not None]
        p3_2_log.info(f"[extract_pids_for_prod] Found {len(pids)} distinct PIDs for refNo={ref_no}")
        return pids
    except Exception as e:
        p3_2_log.error(f"[extract_pids_for_prod] Query failed: {e}", exc_info=True)
        return []


def fetch_data_for_ws_stno(payload: dict, max_rows: int = 20_000) -> dict:
    """
    WS-based: aynı work_station_id + produced_stock_no olan TÜM rowları çek (refNo filtresi YOK)
    """
    ws_id = payload.get("wsId")
    stock_no = _clean_stock(payload.get("stNo"))
    
    if ws_id is None or not stock_no:
        p3_2_log.warning(f"[fetch_data_for_ws_stno] Missing wsId or stNo in payload")
        return {}
    
    query = f"""
        SELECT job_order_operation_id, work_station_id, measurement_date,
               equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type,
               good, workstation_state, job_order_reference_no, prod_order_reference_no,
               produced_stock_no, produced_stock_name,
               operationname, operationno, operationtaskcode,
               work_center_id, work_center_name, work_center_no,
               work_station_name, work_station_no,
               plant_id, plant_name, customer
        FROM {RAW_TABLE}
        WHERE work_station_id = %s AND produced_stock_no = %s
        LIMIT %s ALLOW FILTERING
    """
    
    try:
        rows = list(session.execute(query, (int(ws_id), str(stock_no), max_rows)))
        p3_2_log.info(f"[fetch_data_for_ws_stno] (1) Fetched {len(rows)} rows for wsId={ws_id}, stock={stock_no}")
    except Exception as e:
        p3_2_log.error(f"[fetch_data_for_ws_stno] (1) Query failed: {e}", exc_info=True)
        try:
            time.sleep(0.7)
            rows = list(session.execute(query, (int(ws_id), str(stock_no), max_rows)))
            p3_2_log.info(f"[fetch_data_for_ws_stno] (2) Fetched {len(rows)} rows for wsId={ws_id}, stock={stock_no}")
        except Exception as e:
            p3_2_log.error(f"[fetch_data_for_ws_stno] (2) Query failed (2): {e}", exc_info=True)
            return {}
    
    if not rows:
        return {}
    
    bucket = defaultdict(list)
    label_meta = {}
    
    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue
        
        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": _clean_stock(getattr(r, "produced_stock_no", None)),
                "output_stock_name": _clean_stock(getattr(r, "produced_stock_name", None)),
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None),
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None),
                # Workcenter bilgileri
                "workcenter_name": getattr(r, "work_center_name", None),
                "workcenter_no": getattr(r, "work_center_no", None),
                "work_center_id": getattr(r, "work_center_id", None),
                # Workstation bilgileri
                "workstation_name": getattr(r, "work_station_name", None),
                "workstation_no": getattr(r, "work_station_no", None),
                # Plant bilgileri
                "plant_id": getattr(r, "plant_id", None),
                "plant_name": getattr(r, "plant_name", None),
                # Customer
                "customer": getattr(r, "customer", None)
            }
        
        # ===== NEW: Check equipment_type =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type)
        
        if is_input:
            # INPUT variable
            var_name = getattr(r, "equipment_name", None) or getattr(r, "equipment_no", None)
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
            # OUTPUT variable
            pname = getattr(r, "equipment_no", None) or getattr(r, "equipment_name", None)
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
    
    job_ids, dates, ids, sensor_values = [], [], [], []
    
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue
        
        lm = label_meta.get(ts, {}) or {}
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
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
            # Workcenter bilgileri
            "workcenter_name": lm.get("workcenter_name"),
            "workcenter_no": lm.get("workcenter_no"),
            "work_center_id": lm.get("work_center_id"),
            # Workstation bilgileri
            "workstation_name": lm.get("workstation_name"),
            "workstation_no": lm.get("workstation_no"),
            # Plant bilgileri
            "plant_id": lm.get("plant_id"),
            "plant_name": lm.get("plant_name"),
            # Customer
            "customer": lm.get("customer")
        }
        
        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])
    
    return {
        "sensor_values": sensor_values,
        "dates": dates,
        "job_ids": job_ids,
        "output_stock_no": str(stock_no),
    }


def fetch_data_for_prodorder(payload: dict, max_rows: int = 20_000) -> dict:
    """
    ProdOrder-based: work_station_id + produced_stock_no + prod_order_reference_no ile filtrele
    SADECE belirtilen production order'a ait rowları çek
    """
    ws_id = payload.get("wsId")
    stock_no = _clean_stock(payload.get("stNo"))
    ref_no = payload.get("refNo")
    
    if ws_id is None or not stock_no or not ref_no:
        p3_2_log.warning(f"[fetch_data_for_prodorder] Missing wsId, stNo or refNo in payload")
        return {}
    
    query = f"""
        SELECT job_order_operation_id, work_station_id, measurement_date,
               equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type,
               good, workstation_state, job_order_reference_no, prod_order_reference_no,
               produced_stock_no, produced_stock_name,
               operationname, operationno, operationtaskcode,
               work_center_id, work_center_name, work_center_no,
               work_station_name, work_station_no,
               plant_id, plant_name, customer
        FROM {RAW_TABLE}
        WHERE work_station_id = %s AND produced_stock_no = %s AND prod_order_reference_no = %s
        LIMIT %s ALLOW FILTERING
    """
    
    try:
        rows = list(session.execute(query, (int(ws_id), str(stock_no), str(ref_no), max_rows)))
        p3_2_log.info(f"[fetch_data_for_prodorder] (1) Fetched {len(rows)} rows for wsId={ws_id}, stock={stock_no}, refNo={ref_no}")
    except Exception as e:
        p3_2_log.error(f"[fetch_data_for_prodorder] Query failed (1): {e}", exc_info=True)
        rows = None
        limit = int(max_rows)
        for attempt in range(5):
            try:
                time.sleep(0.7)
                rows = list(session.execute(query, (int(ws_id), str(stock_no), str(ref_no), (limit // 2))))
                p3_2_log.info(f"[fetch_data_for_prodorder] (1) Fetched {len(rows)} rows for wsId={ws_id}, stock={stock_no}, refNo={ref_no}")
            except Exception as e:
                p3_2_log.error(f"[fetch_data_for_prodorder] Query failed (2): {e}", exc_info=True)
                time.sleep(0.7 * (attempt + 1))  # backoff
                limit = max(200, limit // 2)
                #return {}
            except Exception as e:
                p3_2_log.error(f"[fetch_data_for_prodorder] Unexpected error: {e}", exc_info=True)
                return {}
            
    if not rows:
        return {}
    
    bucket = defaultdict(list)
    label_meta = {}
    
    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue
        
        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": _clean_stock(getattr(r, "produced_stock_no", None)),
                "output_stock_name": _clean_stock(getattr(r, "produced_stock_name", None)),
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None),
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None),
                # Workcenter bilgileri
                "workcenter_name": getattr(r, "work_center_name", None),
                "workcenter_no": getattr(r, "work_center_no", None),
                "work_center_id": getattr(r, "work_center_id", None),
                # Workstation bilgileri
                "workstation_name": getattr(r, "work_station_name", None),
                "workstation_no": getattr(r, "work_station_no", None),
                # Plant bilgileri
                "plant_id": getattr(r, "plant_id", None),
                "plant_name": getattr(r, "plant_name", None),
                # Customer
                "customer": getattr(r, "customer", None)
            }
        
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type)
        
        if is_input:
            # INPUT variable
            var_name = getattr(r, "equipment_name", None) or getattr(r, "equipment_no", None)
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
            # OUTPUT variable
            pname = getattr(r, "equipment_no", None) or getattr(r, "equipment_name", None)
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
    
    job_ids, dates, ids, sensor_values = [], [], [], []
    
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue
        
        lm = label_meta.get(ts, {}) or {}
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
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
            # Workcenter bilgileri
            "workcenter_name": lm.get("workcenter_name"),
            "workcenter_no": lm.get("workcenter_no"),
            "work_center_id": lm.get("work_center_id"),
            # Workstation bilgileri
            "workstation_name": lm.get("workstation_name"),
            "workstation_no": lm.get("workstation_no"),
            # Plant bilgileri
            "plant_id": lm.get("plant_id"),
            "plant_name": lm.get("plant_name"),
            # Customer
            "customer": lm.get("customer")
        }
        
        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])
    
    return {
        "sensor_values": sensor_values,
        "dates": dates,
        "job_ids": job_ids,
        "output_stock_no": str(stock_no),
    }



def fetch_dataset_for_retrain_request(payload: dict, max_rows: int = 20_000) -> dict:
    """
    Retrain request için dataset hazırla
    """
    # Validate payload
    if not payload:
        p3_2_log.error("[fetch_dataset_for_retrain_request] Empty payload")
        return {}
    
    required_fields = ["plId", "wsId", "cust", "stNo", "algorithm", "type"]
    missing = [f for f in required_fields if payload.get(f) is None]
    if missing:
        p3_2_log.error(f"[fetch_dataset_for_retrain_request] Missing fields: {missing}")
        return {}
    
    if payload.get("type") not in ["CORRELATION", "FEATURE_IMPORTANCE", "REALTIME_PREDICTION"]:
        p3_2_log.error(f"[fetch_dataset_for_retrain_request] Invalid type: {payload.get('type')}")
        return {}
    
    # Determine scope
    scope = None
    scope_id = None
    
    if payload.get("joOpId") is not None:
        # PID-based
        scope = "pid"
        scope_id = payload.get("joOpId")
    elif payload.get("refNo") is not None:
        # Production order based
        scope = "prodOrder"
        scope_id = payload.get("refNo")
    else:
        # WS-based
        scope = "ws"
        scope_id = payload.get("wsId")
    
    if not scope or scope_id is None:
        p3_2_log.error("[fetch_dataset_for_retrain_request] Could not determine scope")
        return {}
    
    # Determine type
    retrain_for = None
    if payload.get("type") == "CORRELATION":
        retrain_for = "correlation"
    elif payload.get("type") == "FEATURE_IMPORTANCE":
        retrain_for = "feature_importance"
    elif payload.get("type") == "REALTIME_PREDICTION":
        retrain_for = "realtime_prediction"
    
    if not retrain_for:
        p3_2_log.error(f"[fetch_dataset_for_retrain_request] Invalid retrain type")
        return {}
    
    # Fetch data based on scope
    dataset = {}
    
    if scope == "pid":
        # PID'e göre önce opTc'yi bul
        try:
            op_tc_query = f"""
                SELECT operationtaskcode 
                FROM {RAW_TABLE} 
                WHERE job_order_operation_id = %s 
                LIMIT 1 ALLOW FILTERING
            """
            op_tc_rows = list(session.execute(op_tc_query, (int(scope_id),)))
            if op_tc_rows:
                payload["operationtaskcode"] = getattr(op_tc_rows[0], "operationtaskcode", None)
        except Exception as e:
            p3_2_log.error(f"[fetch_dataset_for_retrain_request] Failed to fetch opTc for pid={scope_id}: {e}")
        
        # RETRAIN için TÜM PID'lerin verisini çek (include_all_pids=True)
        dataset = fetch_data_for_pid_optc(payload, pid=str(scope_id), max_rows=max_rows, include_all_pids=True)
        
    elif scope == "prodOrder":
        # *** YENİ: ProdOrder için özel fonksiyon kullan (refNo filtreli) ***
        p3_2_log.info(f"[fetch_dataset_for_retrain_request] ProdOrder scope: fetching with refNo filter for refNo={scope_id}")
        
        # Sadece bu production order'a ait veriyi çek
        dataset = fetch_data_for_prodorder(payload, max_rows=max_rows)  # <<<< YENİ FONKSIYON
        
        if not dataset or not dataset.get("sensor_values"):
            p3_2_log.warning(f"[fetch_dataset_for_retrain_request] No data found for prodOrder={scope_id}")
            return {}
        
        # Sensor_values'dan unique PID'leri çıkar (bilgi için)
        unique_pids = set()
        for bundle in dataset.get("sensor_values", []):
            if bundle and isinstance(bundle, list) and len(bundle) > 0:
                meta = bundle[0]
                if isinstance(meta, dict):
                    pid = meta.get("job_order_operation_id")
                    if pid is not None:
                        unique_pids.add(pid)
        
        p3_2_log.info(
            f"[fetch_dataset_for_retrain_request] Found {len(unique_pids)} unique PIDs "
            f"in {len(dataset.get('sensor_values', []))} bundles for prodOrder={scope_id}"
        )
        
        # operationtaskcode ve output_stock_no bilgilerini de çıkar
        if dataset.get("sensor_values"):
            first_meta = dataset["sensor_values"][0][0] if isinstance(dataset["sensor_values"][0], list) else {}
            dataset["operationtaskcode"] = first_meta.get("operationtaskcode")
            dataset["output_stock_no"] = first_meta.get("output_stock_no")
        
    elif scope == "ws":
        dataset = fetch_data_for_ws_stno(payload, max_rows=max_rows)
    
    if not dataset or not dataset.get("sensor_values"):
        p3_2_log.warning(f"[fetch_dataset_for_retrain_request] No data found for scope={scope}, scope_id={scope_id}")
        return {}
    
    return {
        "scope": scope,
        "scope_id": scope_id,
        "algorithm": payload.get("algorithm"),
        "type": retrain_for,
        "sensor_values": dataset.get("sensor_values", []),
        "dates": dataset.get("dates", []),
        "job_ids": dataset.get("job_ids", []),
        "operationtaskcode": dataset.get("operationtaskcode"),
        "output_stock_no": dataset.get("output_stock_no"),
    }


# -------------------- server --------------------
def execute_retrain():
    p3_2_log.info("[execute_retrain] Starting retrain Flask server...")

    # Test Cassandra connection before starting Flask
    try:
        p3_2_log.info("[execute_retrain] Testing Cassandra connection...")
        test_query = f"SELECT * FROM {RAW_TABLE} LIMIT 1"
        test_result = list(session.execute(test_query))
        p3_2_log.info(f"[execute_retrain] Cassandra connection OK (test query returned {len(test_result)} rows)")
    except Exception as e:
        p3_2_log.error(f"[execute_retrain] Cassandra connection FAILED: {e}", exc_info=True)
        p3_2_log.error("[execute_retrain] Cannot start server without database connection")
        return  # Exit early

    app = Flask(__name__)

    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        p3_2_log.info("[execute_retrain] /health endpoint called")
        return "OK"

    @app.route("/scada-optimization/retrain", methods=["POST"])
    def retrain_endpoint():
        try:
            p3_2_log.info("[execute_retrain] /scada-optimization/retrain endpoint called")
            
            payload = request.get_json(force=True) or {}
            p3_2_log.info(f"[execute_retrain] request payload={payload}")

            msg = {
                "plId": payload.get("plantId"),
                "cust": payload.get("customer"),
                "wsId": payload.get("workstationId"),
                "stNo": payload.get("outputStockNo"),
                "joOpId": payload.get("jobOrderOperationId"),
                "refNo": payload.get("prodOrderReferenceNo"),
                "algorithm": payload.get("algorithm"),
                "type": payload.get("type"),
            }

            p3_2_log.info(f"[execute_retrain] Fetching dataset for msg={msg}")

            ds = fetch_dataset_for_retrain_request(msg, max_rows=20_000)

            if not ds:
                p3_2_log.warning("[execute_retrain] Dataset fetch returned empty")
                return jsonify({
                    "status": "error",
                    "message": "Failed to fetch dataset"
                }), 400

            bundles = len(ds.get("sensor_values") or [])
            p3_2_log.info(
                f"[execute_retrain] dataset ready scope={ds.get('scope')} scope_id={ds.get('scope_id')} "
                f"bundles={bundles} algo={ds.get('algorithm')} type={ds.get('type')} "
                f"opTc={ds.get('operationtaskcode')} stock={ds.get('output_stock_no')}"
            )

            result = handle_retrain_request(
                payload=payload,
                dataset=ds.get("sensor_values") or [],
                dates=ds.get("dates") or [],
                job_ids=ds.get("job_ids") or [],
                scope=ds.get("scope"),
                scope_id=ds.get("scope_id"),
                operationtaskcode=ds.get("operationtaskcode"),
                output_stock_no=ds.get("output_stock_no"),
                input_feature_names=None,
                output_feature_names=None,
                logger=p3_2_log,
            )

            p3_2_log.info(f"[execute_retrain] Retrain completed, result={result}")

            return jsonify({
                "status": "ok",
                "scope": ds.get("scope"),
                "scope_id": ds.get("scope_id"),
                "bundles": bundles,
                "operationtaskcode": ds.get("operationtaskcode"),
                "output_stock_no": ds.get("output_stock_no"),
                "result": result,
            })

        except Exception as e:
            p3_2_log.error(f"[execute_retrain] endpoint error: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    try:
        port = int(cfg["retrain_port"] or 6196)
        p3_2_log.info(f"[execute_retrain] Flask app listening on 0.0.0.0:{port}")
        
        # Use debug=False in production, but add explicit startup message
        app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
        
    except Exception as e:
        p3_2_log.error(f"[execute_retrain] Flask app failed to start: {e}", exc_info=True)
        raise


"""
CORS(app)  # CORS'u aktif et

    # SWAGGER JSON ENDPOINT - Swagger UI için gerekli
    @app.route("/swagger.json", methods=["GET"])
    def swagger_spec():
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "META SMART FACTORY REST API",
                "version": "1.0",
                "description": "META SMART FACTORY REST API for ERP"
            },
            "servers": [
                {
                    "url": "http://209.250.235.243:6196",
                    "description": "Retrain Service"
                }
            ],
            "tags": [
                {
                    "name": "scada-optimization-controller",
                    "description": "SCADA Optimization Operations"
                }
            ],
            "paths": {
                "/health": {
                    "get": {
                        "tags": ["scada-optimization-controller"],
                        "summary": "Health check",
                        "responses": {
                            "200": {
                                "description": "Service is healthy",
                                "content": {
                                    "text/plain": {
                                        "schema": {
                                            "type": "string",
                                            "example": "OK"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/scada-optimization/retrain": {
                    "post": {
                        "tags": ["scada-optimization-controller"],
                        "summary": "Trigger model retraining",
                        "description": "Initiates the retraining process for SCADA optimization models",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["algorithm", "customer", "plantId", "workstationId", "type"],
                                        "properties": {
                                            "algorithm": {
                                                "type": "string",
                                                "description": "Algorithm type (required)",
                                                "example": "SPEARMAN"
                                            },
                                            "customer": {
                                                "type": "string",
                                                "description": "Customer name (required)",
                                                "example": "teknia_group"
                                            },
                                            "outputStockNo": {
                                                "type": "string",
                                                "description": "Output stock number (optional)",
                                                "example": "Citramon-Forte tablet mass"
                                            },
                                            "plantId": {
                                                "type": "integer",
                                                "description": "Plant identifier (required)",
                                                "example": 149
                                            },
                                            "workstationId": {
                                                "type": "integer",
                                                "description": "Workstation identifier (required)",
                                                "example": 441165
                                            },
                                            "type": {
                                                "type": "string",
                                                "description": "Operation type: CORRELATION, FEATURE_IMPORTANCE, or REALTIME_PREDICTION (required)",
                                                "example": "CORRELATION",
                                                "enum": ["CORRELATION", "FEATURE_IMPORTANCE", "REALTIME_PREDICTION"]
                                            },
                                            "prodOrderReferenceNo": {
                                                "type": "string",
                                                "description": "Production order reference number (optional)",
                                                "example": None
                                            },
                                            "jobOrderOperationId": {
                                                "type": "string",
                                                "description": "Job order operation identifier (optional)",
                                                "example": None
                                            },
                                            "startDate": {
                                                "type": "string",
                                                "format": "date-time",
                                                "description": "Start date (optional)",
                                                "example": "2025-09-08T09:03:11.000+00:00"
                                            },
                                            "endDate": {
                                                "type": "string",
                                                "format": "date-time",
                                                "description": "End date (optional)",
                                                "example": "2025-09-08T09:34:34.000+00:00"
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "OK",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string", "example": "ok"},
                                                "scope": {"type": "string"},
                                                "scope_id": {"type": "string"},
                                                "bundles": {"type": "integer"},
                                                "operationtaskcode": {"type": "string"},
                                                "output_stock_no": {"type": "string"},
                                                "result": {"type": "object"}
                                            }
                                        }
                                    }
                                }
                            },
                            "400": {
                                "description": "Bad request",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string", "example": "error"},
                                                "message": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            },
                            "500": {
                                "description": "Internal server error",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string", "example": "error"},
                                                "message": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return jsonify(spec)
"""