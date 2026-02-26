"""Phase2 Cassandra persistence for dw_tbl_multiple_anomalies.

Patch1 goal:
- Persist Phase2 scores (anomaly_score / anomaly_importance / anomaly_detected) to Cassandra
  in a schema-compatible manner.

Notes:
- Writes are bucket-level (FrameFinalizeResult) to avoid per-message write amplification.
- The table schema in customer deployments has historically varied; this writer is defensive
  about field presence and types, and will drop fields that cannot be safely coerced.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pytz

from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model

from cassandra_utils.cqlengine_init import ensure_cqlengine_setup
from utils.config_reader import ConfigReader
from utils.keypoint_recorder import KP

cfg = ConfigReader()
_LOCK = threading.Lock()
_MODEL: Optional[type[Model]] = None


def _get_first(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            v = d.get(k)
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            return v
    return default


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int,)):
            return int(x)
        s = str(x).strip()
        if not s:
            return None
        # common "PID_123" -> 123
        if s.lower().startswith("pid_"):
            s2 = s.split("_", 1)[-1]
            return int(s2)
        return int(float(s))
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    return None


def _dt_from_epoch_ms(ms: Any) -> datetime:
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=pytz.UTC)
    except Exception:
        return datetime.now(pytz.UTC)



def _build_sensor_maps_from_bucket_values(
    bucket_values: Dict[str, float],
    *,
    value_key: str = "cntRead",
    max_items: Optional[int] = None,
    sensor_meta_by_sid: Optional[Dict[str, Dict[str, object]]] = None,
    measurement_dt: Optional[datetime] = None,
) -> Optional[dict]:
    """Build sensor_values nested map from finalized bucket values.

    Shape: sensor_id -> {value_key: "<float>"}
    This keeps compatibility with the legacy nested-map schema without requiring full outVals payloads.
    """
    if not bucket_values:
        return None
    items = list(bucket_values.items())
    if max_items is not None and int(max_items) > 0 and len(items) > int(max_items):
        items = items[: int(max_items)]
    out: Dict[str, Dict[str, str]] = {}
    for sid, v in items:
        if sid is None:
            continue
        s = str(sid).strip()
        if not s:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:
            continue
        inner: Dict[str, str] = {str(value_key): f"{fv:.6f}"}
        if sensor_meta_by_sid and s in sensor_meta_by_sid and isinstance(sensor_meta_by_sid.get(s), dict):
            for k, vv in sensor_meta_by_sid.get(s, {}).items():
                if vv is None:
                    continue
                if isinstance(vv, (dict, list)):
                    continue
                inner[str(k)] = str(vv)
        if measurement_dt is not None:
            inner["measDt"] = measurement_dt.isoformat(sep=" ")
        inner.setdefault("eqNo", s)
        inner.setdefault("eqNm", s)
        out[s] = inner
    return out or None


def _build_heatmap_from_contrib(
    contrib: Dict[str, float],
    *,
    value_key: str = "cntRead",
    topn: int = 0,
    sensor_meta_by_sid: Optional[Dict[str, Dict[str, object]]] = None,
    measurement_dt: Optional[datetime] = None,
) -> Optional[dict]:
    """Build heatmap nested map from model/scorer per-sensor contributions.

    Shape: sensor_id -> {value_key: "<float>"}
    We store only top-N by magnitude to keep the payload bounded.
    """
    if not contrib:
        return None
    items = []
    for sid, v in contrib.items():
        if sid is None:
            continue
        s = str(sid).strip()
        if not s:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:
            continue
        items.append((abs(fv), s, fv))
    if not items:
        return None
    items.sort(reverse=True)
    if topn is not None and int(topn) > 0:
        items = items[: max(1, int(topn))]
    # else: keep all items (no truncation)

    out: Dict[str, Dict[str, str]] = {}
    for _, s, fv in items:
        inner: Dict[str, str] = {str(value_key): f"{float(fv):.6f}"}
        if sensor_meta_by_sid and s in sensor_meta_by_sid and isinstance(sensor_meta_by_sid.get(s), dict):
            for k, v in sensor_meta_by_sid.get(s, {}).items():
                if v is None:
                    continue
                if isinstance(v, (dict, list)):
                    continue
                inner[str(k)] = str(v)
        if measurement_dt is not None:
            inner["measDt"] = measurement_dt.isoformat(sep=" ")
        # Ensure name fields exist for backend display
        inner.setdefault("eqNo", s)
        inner.setdefault("eqNm", s)
        out[s] = inner
    return out or None



def _build_sensor_maps(message: Dict[str, Any]) -> Tuple[Optional[dict], Optional[dict]]:
    """Return (sensor_values, heatmap) as nested maps: sensor -> (k -> str(v)).

    sensor_values stores raw-ish payload (primarily cntRead + identifiers).
    heatmap stores a normalized magnitude per sensor (cntRead := abs(z)).
    """

    out_vals = message.get("outVals") or message.get("out_vals") or []
    if isinstance(out_vals, dict):
        out_vals = [out_vals]

    sensor_values: Dict[str, Dict[str, str]] = {}
    raw_val_by_sensor: Dict[str, float] = {}

    if isinstance(out_vals, list):
        for ov in out_vals:
            if not isinstance(ov, dict):
                continue
            name = _get_first(ov, ("eqNo", "eqNm", "equipment_no", "equipment_name"))
            if name is None:
                continue
            sname = str(name)

            inner: Dict[str, str] = {}
            for k, v in ov.items():
                if v is None:
                    continue
                # Nested structures are not supported in the inner map; stringify.
                try:
                    inner[str(k)] = str(v)
                except Exception:
                    continue

            sensor_values[sname] = inner

            v = _to_float(_get_first(ov, ("cntRead", "counter_reading", "val")))
            if v is not None:
                raw_val_by_sensor[sname] = v

    if not sensor_values:
        return None, None

    # Build a simple robust z-score heatmap.
    heatmap = _compute_heatmap(raw_val_by_sensor)

    heatmap_map: Dict[str, Dict[str, str]] = {}
    for sname, zabs in heatmap.items():
        # keep the same shape as sensor_values, but only include a compact set of keys
        heatmap_map[sname] = {"cntRead": f"{zabs:.6f}"}

    return sensor_values, heatmap_map


# ---- lightweight robust stats for heatmap ----

_HEAT_HIST_LOCK = threading.Lock()
_HEAT_HIST: Dict[str, list[float]] = {}
_HEAT_HIST_MAXLEN = 60


def _compute_heatmap(raw_val_by_sensor: Dict[str, float]) -> Dict[str, float]:
    """Compute abs(z) per sensor using a small rolling history."""
    import numpy as np

    zabs: Dict[str, float] = {}

    with _HEAT_HIST_LOCK:
        for s, v in raw_val_by_sensor.items():
            hist = _HEAT_HIST.get(s)
            if hist is None:
                hist = []
                _HEAT_HIST[s] = hist
            hist.append(float(v))
            if len(hist) > _HEAT_HIST_MAXLEN:
                del hist[: len(hist) - _HEAT_HIST_MAXLEN]

        for s, v in raw_val_by_sensor.items():
            hist = _HEAT_HIST.get(s) or []
            if len(hist) < 8:
                zabs[s] = 0.0
                continue
            arr = np.asarray(hist, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            scale = mad * 1.4826
            if scale <= 1e-9:
                zabs[s] = 0.0
                continue
            z = (float(v) - med) / scale
            zabs[s] = float(abs(z))

    return zabs


# ---- cqlengine model bootstrap ----


def _get_model() -> type[Model]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _LOCK:
        if _MODEL is not None:
            return _MODEL

        cfg = ConfigReader()
        cass = cfg.get("cassandra_props") or cfg.get("cassandra") or {}
        keyspace = cass.get("keyspace")
        table_name = (cass.get("multiple_anomalies") or "dw_tbl_multiple_anomalies").strip()

        ensure_cqlengine_setup(cfg)

        class DwTblMultipleAnomalies(Model):
            __keyspace__ = str(keyspace) if keyspace else None
            __table_name__ = table_name

            partition_date = columns.Text(partition_key=True)
            measurement_date = columns.DateTime(primary_key=True, clustering_order="DESC")
            # Must match table clustering order (ASC in deployments using dw_tbl_multiple_anomalies3.py)
            unique_code = columns.Text(primary_key=True, clustering_order="ASC")

            # Common payload columns (subset is OK; missing columns remain null in Cassandra)
            production_order_ref_id = columns.Text()
            algorithm = columns.Text()
            anomaly_detected = columns.Boolean()
            anomaly_importance = columns.Double()
            anomaly_score = columns.Double()
            heapmap_threshold = columns.Double()

            create_date = columns.DateTime()
            id = columns.UUID()

            # context
            customer = columns.Text()
            message_key = columns.Text()
            machine_state = columns.Text()

            plant_id = columns.Integer()
            plant_name = columns.Text()
            # Some deployments store plant_no as text (see dw_tbl_multiple_anomalies3.py)
            plant_no = columns.Text()

            workcenter_id = columns.Integer()
            workcenter_name = columns.Text()
            workcenter_no = columns.Text()

            workstation_id = columns.Integer()
            workstation_name = columns.Text()
            workstation_no = columns.Text()
            workstation_state = columns.Text()

            produced_stock_id = columns.Integer()
            produced_stock_name = columns.Text()
            produced_stock_no = columns.Text()

            employee_id = columns.Integer()
            job_order_operation_id = columns.Integer()
            job_order_operation_ref_id = columns.Text()

            good = columns.Boolean()
            current_quantity = columns.Double()
            # quantity_changed is historically a boolean column in dw_tbl_multiple_anomalies
            quantity_changed = columns.Boolean()

            # nested maps
            sensor_values = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))
            heatmap = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))
            input_variables = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))

            # optional shift fields
            shift_start_time = columns.DateTime()
            shift_finish_time = columns.DateTime()
            shift_start_text = columns.Text()
            shift_finish_text = columns.Text()

            active = columns.Boolean()

        _MODEL = DwTblMultipleAnomalies
        return _MODEL


# ---- public API ----


@dataclass
class Phase2PersistResult:
    ok: bool
    error: Optional[str] = None


def persist_phase2_score(
    *,
    message: Dict[str, Any],
    measurement_dt: datetime,
    algorithm: str,
    anomaly_score: float,
    anomaly_threshold: Optional[float],
    is_anomaly: bool,
    # Optional overrides to ensure bucket-level consistency
    anomaly_importance: Optional[float] = None,
    bucket_values: Optional[Dict[str, float]] = None,
    heatmap_values: Optional[Dict[str, float]] = None,
    sensor_meta_by_sid: Optional[Dict[str, Dict[str, object]]] = None,
) -> Phase2PersistResult:
    """Persist a Phase2 score row into dw_tbl_multiple_anomalies."""

    try:
        ModelCls = _get_model()

        # Cassandra primary key partition_date is stored as text (YYYY-MM-DD)
        partition_date = measurement_dt.strftime("%Y-%m-%d")

        # Context extraction (defensive)
        prod_order_ref = _get_first(
            message,
            (
                "refNo",
                "prod_order_reference_no",
                "production_order_ref_id",
                "productionOrderRefId",
                "prodOrderReferenceNo",
            ),
        )

        # IDs/names
        plant_id = _to_int(_get_first(message, ("plId", "plant_id", "plantId")))
        plant_no_raw = _get_first(message, ("plNo", "plant_no", "plantNo"), plant_id)
        plant_no = str(plant_no_raw) if plant_no_raw is not None else (str(plant_id) if plant_id is not None else None)
        plant_name = _get_first(message, ("plNm", "plant_name", "plantName"))

        wc_id = _to_int(_get_first(message, ("wcId", "work_center_id", "workcenter_id")))
        wc_no = _get_first(message, ("wcNo", "work_center_no", "workcenter_no"))
        wc_name = _get_first(message, ("wcNm", "work_center_name", "workcenter_name"))

        ws_id = _to_int(_get_first(message, ("wsId", "work_station_id", "workstation_id")))
        ws_no = _get_first(message, ("wsNo", "work_station_no", "workstation_no"))
        ws_name = _get_first(message, ("wsNm", "work_station_name", "workstation_name"))

        # stock
        prod = message.get("prodList") or []
        if isinstance(prod, dict):
            prod = [prod]
        p0 = prod[0] if isinstance(prod, list) and prod and isinstance(prod[0], dict) else {}
        st_id = _to_int(_get_first(message, ("stId", "produced_stock_id", "producedStockId"), _get_first(p0, ("stId",))))
        st_no = _get_first(message, ("stNo", "produced_stock_no", "producedStockNo"), _get_first(p0, ("stNo", "stNm")))
        st_name = _get_first(message, ("stNm", "produced_stock_name", "producedStockName"), _get_first(p0, ("stNm", "stNo")))

        # task
        jo_op_raw = _get_first(message, ("joOpId", "job_order_operation_id", "jobOrderOperationId"))
        jo_op_id = _to_int(jo_op_raw)
        jo_op_ref = _get_first(message, ("job_order_operation_ref_id", "jobOrderOperationRefId"))
        if jo_op_ref is None and jo_op_raw is not None:
            jo_op_ref = str(jo_op_raw)

        # misc
        customer = _get_first(message, ("cust", "customer"))
        mc_state = _get_first(message, ("mcSt", "machine_state", "machineState"))
        ws_state = _get_first(message, ("wsSt", "workstation_state", "workstationState", "prSt", "process_state", "processState"))
        active = _to_bool(_get_first(message, ("act", "active")))
        if active is None:
            active = True
        good = _to_bool(_get_first(message, ("good", "goodCnt")))

        curr_qty = _to_float(_get_first(message, ("currCycQty", "current_quantity")))
        # quantity_changed is a boolean in dw_tbl_multiple_anomalies; derive it if we only have a delta value.
        quantity_changed = _to_bool(_get_first(message, ("quantity_changed",)))
        if quantity_changed is None:
            chng_qty = _to_float(_get_first(message, ("chngCycQty",)))
            if chng_qty is not None:
                quantity_changed = bool(abs(float(chng_qty)) > 1e-12)
        emp_id = _to_int(_get_first(message, ("empId", "employee_id", "employeeId")))

        # message key: prefer provided; else synthesize
        msg_key = _get_first(message, ("message_key", "msgKey"))
        if msg_key is None:
            if ws_id is not None and st_id is not None:
                msg_key = f"{ws_id}_{st_id}"
            elif ws_id is not None and st_no is not None:
                msg_key = f"{ws_id}_{st_no}"

        sensor_values = None
        heatmap = None
        if bucket_values is not None:
            sensor_values = _build_sensor_maps_from_bucket_values(bucket_values, sensor_meta_by_sid=sensor_meta_by_sid, measurement_dt=measurement_dt)
        if heatmap_values is not None:
            topn = int(getattr(cfg, "phase2_heatmap_topn", 0) or 0)
            heatmap = _build_heatmap_from_contrib(heatmap_values, topn=topn, sensor_meta_by_sid=sensor_meta_by_sid, measurement_dt=measurement_dt)
        if sensor_values is None and heatmap is None:
            sensor_values, heatmap = _build_sensor_maps(message)

        values: Dict[str, Any] = {
            "partition_date": partition_date,
            "measurement_date": measurement_dt,
            "unique_code": str(uuid.uuid4()),
            "create_date": measurement_dt,
            "id": uuid.uuid4(),
            "production_order_ref_id": (str(prod_order_ref) if prod_order_ref is not None else None),
            "algorithm": str(algorithm),
            "anomaly_detected": bool(is_anomaly),
            "anomaly_importance": float(anomaly_importance) if anomaly_importance is not None else float(anomaly_score),
            "anomaly_score": float(anomaly_score),
            "heapmap_threshold": (float(anomaly_threshold) if anomaly_threshold is not None else None),
            "customer": (str(customer) if customer is not None else None),
            "message_key": (str(msg_key) if msg_key is not None else None),
            "machine_state": (str(mc_state) if mc_state is not None else None),
            "plant_id": plant_id,
            "plant_name": (str(plant_name) if plant_name is not None else None),
            "plant_no": plant_no,
            "workcenter_id": wc_id,
            "workcenter_name": (str(wc_name) if wc_name is not None else None),
            "workcenter_no": (str(wc_no) if wc_no is not None else None),
            "workstation_id": ws_id,
            "workstation_name": (str(ws_name) if ws_name is not None else None),
            "workstation_no": (str(ws_no) if ws_no is not None else None),
            "workstation_state": (str(ws_state) if ws_state is not None else None),
            "produced_stock_id": st_id,
            "produced_stock_name": (str(st_name) if st_name is not None else None),
            "produced_stock_no": (str(st_no) if st_no is not None else None),
            "employee_id": emp_id,
            "job_order_operation_id": jo_op_id,
            "job_order_operation_ref_id": (str(jo_op_ref) if jo_op_ref is not None else None),
            "good": good,
            "current_quantity": curr_qty,
            "quantity_changed": quantity_changed,
            "sensor_values": sensor_values,
            "heatmap": heatmap,
            "active": active,
        }

        # drop None to reduce query size
        clean_values = {k: v for k, v in values.items() if v is not None}

        ModelCls.create(**clean_values)

        KP.inc("phase2.persist.multiple_anomalies.ok", 1)
        if is_anomaly:
            KP.inc("phase2.persist.multiple_anomalies.anom", 1)
        return Phase2PersistResult(ok=True)

    except Exception as e:
        KP.inc("phase2.persist.multiple_anomalies.err", 1)
        return Phase2PersistResult(ok=False, error=str(e))
