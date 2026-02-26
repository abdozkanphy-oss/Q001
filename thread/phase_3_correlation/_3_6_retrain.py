# ============================================================
# COMPLETE RETRAIN FIX - TRULY FIXED NOW
# ============================================================
# Critical fixes:
# 1. ✅ PID scope now uses scope="pid" (not "ws")
# 2. ✅ Uses LAST bundle's date (not current date)
# 3. ✅ Type-safe ID comparisons
# 4. ✅ Proper returns for all scopes
# ============================================================

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import os, gc, joblib
from collections import defaultdict

from utils.logger_2 import setup_logger
p3_2_log = setup_logger("p3_2_retrain_logger", "logs/p3_2_retrain.log")

from thread.phase_3_correlation._3_1_helper_functions import compute_correlation
from thread.phase_3_correlation._3_5_feature_importance import compute_and_save_feature_importance

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _normalize_id(value):
    """Normalize ID to string for comparison."""
    if value is None:
        return None
    try:
        s = str(value).strip()
        return s if s else None
    except:
        return None


def _build_message_from_bundle2(
    bundle: list,
    date,
    output_stock_no: str = None,
    operationtaskcode: str = None
) -> dict:
    
    """Build message dict from a single bundle's metadata."""
    meta = bundle[0] if (bundle and isinstance(bundle[0], dict)) else {}
    
    # Convert date to datetime
    if not isinstance(date, datetime):
        try:
            dt_ms = int(date)
            if dt_ms > 10**12:
                dt = datetime.fromtimestamp(dt_ms / 1000.0, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(dt_ms, tz=timezone.utc)
        except:
            dt = datetime.now(timezone.utc)
    else:
        dt = date if date.tzinfo else date.replace(tzinfo=timezone.utc)
    
    return {
        "crDt": int(dt.timestamp() * 1000),
        "plId": meta.get("plant_id"),
        "wsId": meta.get("work_station_id") or meta.get("workstation_id"),
        "wsNo": meta.get("workstation_no"),
        "wsNm": meta.get("workstation_name"),
        "wcNo": meta.get("workcenter_no"),
        "wcNm": meta.get("workcenter_name"),
        "opNm": meta.get("operationname") or meta.get("operation_name"),
        "opNo": meta.get("operationno") or meta.get("operation_no"),
        "joOpId": meta.get("job_order_operation_id"),
        "joRef": meta.get("job_order_reference_no"),
        "refNo": meta.get("prod_order_reference_no"),
        "output_stock_no": output_stock_no or meta.get("output_stock_no"),
        "output_stock_name": meta.get("output_stock_name"),
        "cust": meta.get("customer") or "teknia_group",
        "operationtaskcode": operationtaskcode or meta.get("operationtaskcode"),
        "opTc": operationtaskcode or meta.get("operationtaskcode"),
        "outVals": [{"cust": meta.get("customer") or "teknia_group"}],
        "prodList": [{
            "stNo": output_stock_no or meta.get("output_stock_no"),
            "stNm": meta.get("output_stock_name"),
        }],
        "prSt": meta.get("workstation_state") or "PRODUCTION"
    }

def _build_message_from_bundle(
    bundle: list,
    date,
    output_stock_no: str = None,
    operationtaskcode: str = None,
) -> dict:
    """Build message dict from bundle WITH sensor lists."""
    
    # Extract metadata
    meta = bundle[0] if (bundle and isinstance(bundle[0], dict)) else {}
    
    # Convert date
    if not isinstance(date, datetime):
        try:
            dt_ms = int(date)
            if dt_ms > 10**12:
                dt = datetime.fromtimestamp(dt_ms / 1000.0, tz=timezone.utc)
            else:
                dt = datetime.fromtimestamp(dt_ms, tz=timezone.utc)
        except:
            dt = datetime.now(timezone.utc)
    else:
        dt = date if date.tzinfo else date.replace(tzinfo=timezone.utc)
    
    # ===== BUILD SENSOR LISTS =====
    input_list = []
    output_list = []
    
    # Extract sensors from bundle (skip first element which is metadata)
    for sensor in bundle[1:]:
        if not isinstance(sensor, dict):
            continue
        
        equipment_type = sensor.get("equipment_type")
        
        # ===== STRICT CHECK: Determine if INPUT or OUTPUT =====
        is_input = False
        
        # Check for INPUT (boolean True or string "true")
        if equipment_type is True:
            is_input = True
        elif isinstance(equipment_type, str) and equipment_type.lower() == "true":
            is_input = True
        # else: equipment_type is False, "false", None, or other → OUTPUT
        
        # Add to appropriate list (MUTUALLY EXCLUSIVE)
        if is_input:
            input_list.append(sensor)
        else:
            output_list.append(sensor)
    
    # ===== BUILD MESSAGE =====
    return {
        "crDt": int(dt.timestamp() * 1000),
        "plId": meta.get("plant_id"),
        "wsId": meta.get("work_station_id") or meta.get("workstation_id"),
        "wsNo": meta.get("workstation_no"),
        "wsNm": meta.get("workstation_name"),
        "wcNo": meta.get("workcenter_no"),
        "wcNm": meta.get("workcenter_name"),
        "opNm": meta.get("operationname") or meta.get("operation_name"),
        "opNo": meta.get("operationno") or meta.get("operation_no"),
        "joOpId": meta.get("job_order_operation_id"),
        "joRef": meta.get("job_order_reference_no"),
        "refNo": meta.get("prod_order_reference_no"),
        "output_stock_no": output_stock_no or meta.get("output_stock_no"),
        "output_stock_name": meta.get("output_stock_name"),
        "cust": meta.get("customer") or "teknia_group",
        "operationtaskcode": operationtaskcode or meta.get("operationtaskcode"),
        "opTc": operationtaskcode or meta.get("operationtaskcode"),
        "prSt": meta.get("workstation_state") or "PRODUCTION",
        # ===== ADD SENSOR LISTS =====
        "inputVariableList": input_list,   # ✅ INPUT sensors
        "outputValueList": output_list,    # ✅ OUTPUT sensors
        
        "outVals": [{"cust":meta.get("customer") or "teknia_group"}],
        "prodList": [{
            "stNo": output_stock_no or meta.get("output_stock_no"),
            "stNm": meta.get("output_stock_name"),
        }],
        "good": meta.get("good")
    }

# ============================================================
# 1) CORRELATION RETRAIN - TRULY FIXED
# ============================================================

def recalculate_correlation(
    sensor_values: list,
    dates: list,
    job_ids: list,
    scope: str,
    scope_id,
    algorithm: str = "SPEARMAN",
    operationtaskcode: str = None,
    output_stock_no: str = None,
    logger=None,
) -> dict:
    """
    TRULY FIXED: Correct scope, correct date, correct filtering.
    """
    if logger is None:
        logger = p3_2_log
    
    logger.info(
        f"[recalculate_correlation] START scope={scope} scope_id={scope_id} "
        f"bundles={len(sensor_values)} algorithm={algorithm}"
    )
    
    if not sensor_values:
        return {"status": "error", "message": "Empty sensor_values"}
    
    try:
        if scope == "pid":
            # ===== Normalize scope_id =====
            scope_id_str = _normalize_id(scope_id)
            
            if not scope_id_str:
                logger.error(f"Invalid scope_id: {scope_id}")
                return {"status": "error", "message": "Invalid scope_id"}
            
            # ===== FILTER to requested PID =====
            filtered_bundles = []
            filtered_dates = []
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                pid = meta.get("job_order_operation_id")
                
                if _normalize_id(pid) == scope_id_str:
                    filtered_bundles.append(bundle)
                    filtered_dates.append(date)
            
            if not filtered_bundles:
                logger.error(
                    f"No data found for PID {scope_id} "
                    f"(searched as: '{scope_id_str}', total bundles: {len(sensor_values)})"
                )
                
                # DEBUG: Log sample PIDs
                sample_pids = set()
                for b in sensor_values[:10]:
                    if b and isinstance(b, list) and b[0]:
                        sample_pids.add(_normalize_id(b[0].get("job_order_operation_id")))
                
                logger.error(f"Sample PIDs in data: {sorted([p for p in sample_pids if p])}")
                
                return {"status": "error", "message": f"No data for PID {scope_id}"}

            # ===== Use LAST bundle's metadata =====
            last_bundle = filtered_bundles[-1]
            last_date = filtered_dates[-1]
            
            message = _build_message_from_bundle(
                last_bundle, last_date, output_stock_no, operationtaskcode
            )
            
            logger.info(
                f"[recalculate_correlation] PID={scope_id}: "
                f"filtered={len(filtered_bundles)} bundles, "
                f"last_date={datetime.fromtimestamp(message['crDt']/1000, tz=timezone.utc).isoformat()}, "
                f"joRef={message['joRef']}"
            )
            
            # ===== CRITICAL: Use scope="pid" for PID! =====
            compute_correlation(
                sensor_values=filtered_bundles,
                message=message,
                p3_1_log=logger,
                algorithm=algorithm,
                scope="pid",  # ✅ CORRECT: "pid" not "ws"!
                scope_id=scope_id,
                group_by_output_stock=False,  # ✅ CORRECT: False for PID
            )
            
            return {
                "status": "success",
                "scope": "pid",
                "scope_id": scope_id,
                "bundles_processed": len(filtered_bundles),
                "partition_date": datetime.fromtimestamp(message['crDt']/1000, tz=timezone.utc).isoformat(),
            }
        
        elif scope == "ws":
            # ===== Normalize =====
            scope_id_str = _normalize_id(scope_id)
            
            if not scope_id_str:
                logger.error(f"Invalid scope_id: {scope_id}")
                return {"status": "error", "message": "Invalid scope_id"}
            
            # ===== FILTER to requested WS + stock =====
            filtered_bundles = []
            filtered_dates = []
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                ws_id = meta.get("work_station_id") or meta.get("workstation_id")
                stock = meta.get("output_stock_no")
                
                if (_normalize_id(ws_id) == scope_id_str and 
                    (not output_stock_no or stock == output_stock_no)):
                    filtered_bundles.append(bundle)
                    filtered_dates.append(date)
            
            if not filtered_bundles:
                logger.error(f"No data found for WS {scope_id}, stock {output_stock_no}")
                return {"status": "error", "message": f"No data for WS {scope_id}"}
            
            # ===== Use LAST bundle's metadata =====
            last_bundle = filtered_bundles[-1]
            last_date = filtered_dates[-1]
            
            message = _build_message_from_bundle(
                last_bundle, last_date, output_stock_no, operationtaskcode
            )
            
            logger.info(
                f"[recalculate_correlation] WS={scope_id}: "
                f"filtered={len(filtered_bundles)} bundles, "
                f"stock={message['output_stock_no']}"
            )
            
            # ===== Compute with FILTERED data =====
            compute_correlation(
                sensor_values=filtered_bundles,
                message=message,
                p3_1_log=logger,
                algorithm=algorithm,
                scope="ws",  # ✅ CORRECT: "ws" for WS
                scope_id=scope_id,
                group_by_output_stock=True,  # ✅ CORRECT: True for WS
            )
            
            return {
                "status": "success",
                "scope": "ws",
                "scope_id": scope_id,
                "stock": message['output_stock_no'],
                "bundles_processed": len(filtered_bundles),
                "partition_date": datetime.fromtimestamp(message['crDt']/1000, tz=timezone.utc).isoformat(),
            }
        
        elif scope == "prodOrder":
            # ===== Normalize =====
            scope_id_str = _normalize_id(scope_id)
            
            if not scope_id_str:
                logger.error(f"Invalid scope_id: {scope_id}")
                return {"status": "error", "message": "Invalid scope_id"}
            
            # ===== Group by PID =====
            pid_bundles = defaultdict(list)
            pid_dates = defaultdict(list)
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                pid = meta.get("job_order_operation_id")
                ref_no = meta.get("prod_order_reference_no")
                
                pid_str = _normalize_id(pid)
                ref_no_str = _normalize_id(ref_no)
                
                if pid_str and ref_no_str == scope_id_str:
                    pid_bundles[pid_str].append(bundle)
                    pid_dates[pid_str].append(date)
            
            if not pid_bundles:
                logger.error(f"No PIDs found for prodOrder {scope_id}")
                return {"status": "error", "message": f"No PIDs for prodOrder {scope_id}"}
            
            results = []
            
            # ===== Process each PID separately =====
            for pid_str, bundles in pid_bundles.items():
                # Use LAST bundle for this PID
                last_bundle = bundles[-1]
                last_date = pid_dates[pid_str][-1]
                
                message = _build_message_from_bundle(
                    last_bundle, last_date, output_stock_no, operationtaskcode
                )
                
                logger.info(
                    f"[recalculate_correlation] PID={pid_str} (prodOrder={scope_id}): "
                    f"bundles={len(bundles)}"
                )
                
                # Compute with THIS PID's data only
                compute_correlation(
                    sensor_values=bundles,
                    message=message,
                    p3_1_log=logger,
                    algorithm=algorithm,
                    scope="pid",  # ✅ Each PID is "pid" scope
                    scope_id=pid_str,
                    group_by_output_stock=False,
                )
                
                results.append({
                    "pid": pid_str,
                    "bundles": len(bundles),
                    "partition_date": datetime.fromtimestamp(message['crDt']/1000, tz=timezone.utc).isoformat(),
                })
            
            return {
                "status": "success",
                "scope": "prodOrder",
                "scope_id": scope_id,
                "pids_processed": len(results),
                "results": results,
            }
        
        else:
            raise ValueError(f"Unknown scope: {scope}")
    
    except Exception as e:
        logger.error(f"[recalculate_correlation] Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# ============================================================
# 2) FEATURE IMPORTANCE RETRAIN - FIXED
# ============================================================

def retrain_feature_importance(
    sensor_values: list,
    dates: list,
    job_ids: list,
    scope: str,
    scope_id,
    algorithm: str = "XGBOOST",
    operationtaskcode: str = None,
    output_stock_no: str = None,
    input_feature_names=None,
    output_feature_names=None,
    logger=None,
) -> dict:
    """FIXED: Filters data, uses correct metadata, returns proper result."""
    if logger is None:
        logger = p3_2_log
    
    logger.info(
        f"[retrain_feature_importance] START scope={scope} scope_id={scope_id} "
        f"bundles={len(sensor_values)} algorithm={algorithm}"
    )
    
    if not sensor_values:
        return {"status": "error", "message": "Empty sensor_values"}
    
    try:
        if scope == "pid":
            scope_id_str = _normalize_id(scope_id)
            if not scope_id_str:
                return {"status": "error", "message": "Invalid scope_id"}
            
            filtered_bundles = []
            filtered_dates = []
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                pid = meta.get("job_order_operation_id")
                
                if _normalize_id(pid) == scope_id_str:
                    filtered_bundles.append(bundle)
                    filtered_dates.append(date)
            
            if not filtered_bundles:
                logger.error(f"No data found for PID {scope_id}")
                return {"status": "error", "message": f"No data for PID {scope_id}"}
            
            last_bundle = filtered_bundles[-1]
            last_date = filtered_dates[-1]
            
            message = _build_message_from_bundle(
                last_bundle, last_date, output_stock_no, operationtaskcode
            )
            
            logger.info(
                f"[retrain_feature_importance] PID={scope_id}: "
                f"filtered={len(filtered_bundles)} bundles"
            )

            logger.info(
                f"[retrain_feature_importance] IN={input_feature_names} OUT={output_feature_names}"
            )
            
            compute_and_save_feature_importance(
                sensor_values=filtered_bundles,
                message=message,
                input_feature_names=input_feature_names,
                output_feature_names=output_feature_names,
                produce_col="prSt",
                good_col="good",
                drop_cols=("ts", "joOpId", "wsId"),
                algorithm=algorithm,
                p3_1_log=logger,
                scope="pid",
                scope_id=scope_id,
            )
            
            return {
                "status": "success",
                "scope": "pid",
                "scope_id": scope_id,
                "bundles_processed": len(filtered_bundles),
                "partition_date": datetime.fromtimestamp(message['crDt']/1000, tz=timezone.utc).isoformat(),
            }
        
        elif scope == "ws":
            scope_id_str = _normalize_id(scope_id)
            if not scope_id_str:
                return {"status": "error", "message": "Invalid scope_id"}
            
            filtered_bundles = []
            filtered_dates = []
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                ws_id = meta.get("work_station_id") or meta.get("workstation_id")
                stock = meta.get("output_stock_no")
                
                if (_normalize_id(ws_id) == scope_id_str and 
                    (not output_stock_no or stock == output_stock_no)):
                    filtered_bundles.append(bundle)
                    filtered_dates.append(date)
            
            if not filtered_bundles:
                logger.error(f"No data found for WS {scope_id}")
                return {"status": "error", "message": f"No data for WS {scope_id}"}
            
            last_bundle = filtered_bundles[-1]
            last_date = filtered_dates[-1]
            
            message = _build_message_from_bundle(
                last_bundle, last_date, output_stock_no, operationtaskcode
            )
            
            compute_and_save_feature_importance(
                sensor_values=filtered_bundles,
                message=message,
                input_feature_names=input_feature_names,
                output_feature_names=output_feature_names,
                produce_col="prSt",
                good_col="good",
                drop_cols=("ts", "joOpId", "wsId"),
                algorithm=algorithm,
                p3_1_log=logger,
                scope="ws",
                scope_id=scope_id,
            )
            
            return {
                "status": "success",
                "scope": "ws",
                "scope_id": scope_id,
                "bundles_processed": len(filtered_bundles),
                "partition_date": datetime.fromtimestamp(message['crDt']/1000, tz=timezone.utc).isoformat(),
            }
        
        elif scope == "prodOrder":
            scope_id_str = _normalize_id(scope_id)
            if not scope_id_str:
                return {"status": "error", "message": "Invalid scope_id"}
            
            pid_bundles = defaultdict(list)
            pid_dates = defaultdict(list)
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                pid = meta.get("job_order_operation_id")
                ref_no = meta.get("prod_order_reference_no")
                
                pid_str = _normalize_id(pid)
                ref_no_str = _normalize_id(ref_no)
                
                if pid_str and ref_no_str == scope_id_str:
                    pid_bundles[pid_str].append(bundle)
                    pid_dates[pid_str].append(date)
            
            if not pid_bundles:
                logger.error(f"No PIDs found for prodOrder {scope_id}")
                return {"status": "error", "message": f"No PIDs for prodOrder {scope_id}"}
            
            results = []
            
            for pid_str, bundles in pid_bundles.items():
                last_bundle = bundles[-1]
                last_date = pid_dates[pid_str][-1]
                
                message = _build_message_from_bundle(
                    last_bundle, last_date, output_stock_no, operationtaskcode
                )
                
                compute_and_save_feature_importance(
                    sensor_values=bundles,
                    message=message,
                    input_feature_names=input_feature_names,
                    output_feature_names=output_feature_names,
                    produce_col="prSt",
                    good_col="good",
                    drop_cols=("ts", "joOpId", "wsId"),
                    algorithm=algorithm,
                    p3_1_log=logger,
                    scope="pid",
                    scope_id=pid_str,
                )
                
                results.append({
                    "pid": pid_str,
                    "bundles": len(bundles),
                    "partition_date": datetime.fromtimestamp(message['crDt']/1000, tz=timezone.utc).isoformat(),
                })
            
            return {
                "status": "success",
                "scope": "prodOrder",
                "scope_id": scope_id,
                "pids_processed": len(results),
                "results": results,
            }
        
        else:
            raise ValueError(f"Unknown scope: {scope}")
    
    except Exception as e:
        logger.error(f"[retrain_feature_importance] Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# ============================================================
# 3) REALTIME PREDICTION RETRAIN - WITH PROPER INPUT/OUTPUT
# ============================================================

from cassandra_utils.models.scada_real_time_predictions import ScadaRealTimePredictions
from cassandra_utils.models.scada_real_time_prediction_summary import ScadaRealTimePredictionSummary

from thread.phase_3_correlation._3_3_predictions import (
    history_from_fetch,
    handle_realtime_prediction,
    extract_prediction_metadata,
)

def retrain_realtime_prediction(
    sensor_values: list,
    dates: list,
    job_ids: list,
    scope: str,
    scope_id,
    algorithm: str = "LSTM",
    operationtaskcode: str = None,
    output_stock_no: str = None,
    customer: str = None,
    logger=None,
) -> dict:
    """FIXED: Uses proper INPUT/OUTPUT separation from _3_3_predictions.py"""
    if logger is None:
        logger = p3_2_log
    
    logger.info(
        f"[retrain_realtime_prediction] START scope={scope} scope_id={scope_id} "
        f"bundles={len(sensor_values)} algorithm={algorithm}"
    )
    
    if not sensor_values:
        return {"status": "error", "message": "Empty sensor_values"}
    
    try:
        if scope == "pid":
            scope_id_str = _normalize_id(scope_id)
            if not scope_id_str:
                return {"status": "error", "message": "Invalid scope_id"}
            
            filtered_bundles = []
            filtered_dates = []
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                pid = meta.get("job_order_operation_id")
                
                if _normalize_id(pid) == scope_id_str:
                    filtered_bundles.append(bundle)
                    filtered_dates.append(date)
            
            if not filtered_bundles:
                logger.error(f"No data for PID {scope_id}")
                return {"status": "error", "message": f"No data for PID {scope_id}"}
            
            # ===== USE REAL PREDICTION FUNCTION =====
            result = _retrain_and_predict_batch(
                sensor_values=filtered_bundles,
                dates=filtered_dates,
                scope="pid",
                scope_id=scope_id,
                algorithm=algorithm,
                operationtaskcode=operationtaskcode,
                output_stock_no=output_stock_no,
                customer=customer,
                logger=logger,
            )
            
            return result
        
        elif scope == "ws":
            scope_id_str = _normalize_id(scope_id)
            if not scope_id_str:
                return {"status": "error", "message": "Invalid scope_id"}
            
            filtered_bundles = []
            filtered_dates = []
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                ws_id = meta.get("work_station_id") or meta.get("workstation_id")
                
                if _normalize_id(ws_id) == scope_id_str:
                    filtered_bundles.append(bundle)
                    filtered_dates.append(date)
            
            if not filtered_bundles:
                logger.error(f"No data for WS {scope_id}")
                return {"status": "error", "message": f"No data for WS {scope_id}"}
            
            result = _retrain_and_predict_batch(
                sensor_values=filtered_bundles,
                dates=filtered_dates,
                scope="ws",
                scope_id=scope_id,
                algorithm=algorithm,
                operationtaskcode=operationtaskcode,
                output_stock_no=output_stock_no,
                customer=customer,
                logger=logger,
            )
            
            return result
        
        elif scope == "prodOrder":
            scope_id_str = _normalize_id(scope_id)
            if not scope_id_str:
                return {"status": "error", "message": "Invalid scope_id"}
            
            pid_bundles = defaultdict(list)
            pid_dates = defaultdict(list)
            
            for bundle, date in zip(sensor_values, dates):
                if not bundle or not isinstance(bundle, list):
                    continue
                meta = bundle[0] if isinstance(bundle[0], dict) else {}
                pid = meta.get("job_order_operation_id")
                ref_no = meta.get("prod_order_reference_no")
                
                pid_str = _normalize_id(pid)
                ref_no_str = _normalize_id(ref_no)
                
                if pid_str and ref_no_str == scope_id_str:
                    pid_bundles[pid_str].append(bundle)
                    pid_dates[pid_str].append(date)
            
            if not pid_bundles:
                logger.error(f"No PIDs for prodOrder {scope_id}")
                return {"status": "error", "message": f"No PIDs for prodOrder {scope_id}"}
            
            results = []
            
            for pid_str, bundles in pid_bundles.items():
                result = _retrain_and_predict_batch(
                    sensor_values=bundles,
                    dates=pid_dates[pid_str],
                    scope="pid",
                    scope_id=pid_str,
                    algorithm=algorithm,
                    operationtaskcode=operationtaskcode,
                    output_stock_no=output_stock_no,
                    customer=customer,
                    logger=logger,
                )
                
                results.append(result)
            
            return {
                "status": "success",
                "scope": "prodOrder",
                "scope_id": scope_id,
                "pids_processed": len(results),
                "results": results,
            }
        
        else:
            raise ValueError(f"Unknown scope: {scope}")
    
    except Exception as e:
        logger.error(f"[retrain_realtime_prediction] Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def _retrain_and_predict_batch(
    sensor_values: list,
    dates: list,
    scope: str,
    scope_id,
    algorithm: str,
    operationtaskcode: str,
    output_stock_no: str,
    customer: str,
    logger,
):
    """
    Train model on ALL historical data, then predict for EACH timestamp.
    Uses the REAL prediction function from _3_3_predictions.py.
    """
    
    logger.info(f"[_retrain_and_predict_batch] Training for {scope}={scope_id}")
    
    # ===== Prepare seed history (ALL data for training) =====
    hist_in, hist_out = history_from_fetch(dates, sensor_values)
    
    logger.info(
        f"[_retrain_and_predict_batch] Seed history: "
        f"inputs={len(hist_in)}, outputs={len(hist_out)}"
    )
    
    # ===== Predict for EACH timestamp =====
    predictions_written = 0
    
    for i, (current_bundle, current_date) in enumerate(zip(sensor_values, dates)):
        # Build message for THIS timestamp
        msg = _build_message_from_bundle(
            current_bundle, current_date, output_stock_no, operationtaskcode
        )
        
        # ===== USE REAL PREDICTION FUNCTION =====
        # This will:
        # 1. Train separate INPUT and OUTPUT models
        # 2. Predict both input and output variables
        # 3. Save to Cassandra with proper format
        
        try:
            status = handle_realtime_prediction(
                message=msg,
                p3_1_log=logger,
                lookback=10, #20,
                epochs=100,
                min_train_points= 60, #120,
                algorithm=algorithm,
                seed_history=hist_out,  # Use ALL historical outputs for training
                scope=scope,
                scope_id=scope_id,
                group_by_stock=False,  # PID doesn't group by stock
                retrain=True
            )
            
            if status.get("wrote"):
                predictions_written += 1
                
                if i % 10 == 0:  # Log every 10 predictions
                    logger.info(
                        f"[_retrain_and_predict_batch] Progress: {i+1}/{len(sensor_values)} "
                        f"(wrote={predictions_written})"
                    )
        
        except Exception as e:
            logger.error(
                f"[_retrain_and_predict_batch] Failed on timestamp {i}: {e}",
                exc_info=True
            )
            continue
    
    logger.info(f"[_retrain_and_predict_batch] Wrote {predictions_written} predictions")
    
    return {
        "status": "success",
        "scope": scope,
        "scope_id": scope_id,
        "predictions_written": predictions_written,
        "bundles_processed": len(sensor_values),
    }

# ============================================================
# 4) DISPATCHER
# ============================================================

def handle_retrain_request(
    payload: dict,
    dataset: list,
    dates: list,
    job_ids: list,
    scope: str,
    scope_id,
    operationtaskcode: str = None,
    output_stock_no: str = None,
    input_feature_names=None,
    output_feature_names=None,
    logger=None,
) -> dict:
    """Dispatch to correct retrain function."""
    
    if logger is None:
        logger = p3_2_log
    
    retrain_type = payload.get("type")
    algorithm = payload.get("algorithm")
    
    logger.info(f"[handle_retrain_request] type={retrain_type} scope={scope} scope_id={scope_id}")
    
    try:
        if retrain_type == "CORRELATION":
            result = recalculate_correlation(
                dataset, dates, job_ids, scope, scope_id, algorithm,
                operationtaskcode, output_stock_no, logger
            )
            
            return {
                "status": "success",
                "type": "CORRELATION",
                "result": result,
            }
        
        elif retrain_type == "FEATURE_IMPORTANCE":
            result = retrain_feature_importance(
                dataset, dates, job_ids, scope, scope_id, algorithm,
                operationtaskcode, output_stock_no,
                input_feature_names, output_feature_names, logger
            )
            
            return {
                "status": "success",
                "type": "FEATURE_IMPORTANCE",
                "result": result,
            }
        
        elif retrain_type == "REALTIME_PREDICTION":
            result = retrain_realtime_prediction(
                dataset, dates, job_ids, scope, scope_id, algorithm,
                operationtaskcode, output_stock_no, logger
            )
            
            return {
                "status": "success",
                "type": "REALTIME_PREDICTION",
                "result": result,
            }
        
        else:
            return {"status": "error", "message": f"Unknown type: {retrain_type}"}
    
    except Exception as e:
        logger.error(f"[handle_retrain_request] Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "type": retrain_type}