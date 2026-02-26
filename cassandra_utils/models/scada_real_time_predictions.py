from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from datetime import datetime, timezone
import numpy as np

from utils.config_reader import ConfigReader
cfg = ConfigReader()
cass = cfg.get('cassandra_props') or cfg.get('cassandra') or {}
keyspace = cass.get('keyspace') or 'das_new_pm'


def _to_utc(dt_like):
    if dt_like is None:
        return None
    if isinstance(dt_like, datetime):
        return dt_like if dt_like.tzinfo else dt_like.replace(tzinfo=timezone.utc)
    s = str(dt_like)
    # epoch ms or seconds
    try:
        iv = int(float(s))
        return datetime.fromtimestamp(
            iv / 1000.0 if iv > 10 ** 11 else iv,
            tz=timezone.utc
        )
    except Exception:
        pass
    # ISO
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _sf(x: float) -> float:
    """safe float for Cassandra (no NaN/Inf/None)"""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


class ScadaRealTimePredictions(Model):
    __keyspace__ = keyspace
    __table_name__ = "scada_real_time_predictions"

    # PRIMARY KEY ((prod_order_reference_no), partition_date)
    #process_no      = columns.Text(partition_key=True)
    prod_order_reference_no = columns.Text(partition_key=True)
    partition_date  = columns.DateTime(primary_key=True)   # full precision (clustering)

    # regular columns
    algorithm          = columns.Text()
    current_ts         = columns.DateTime()
    start_date         = columns.DateTime()
    customer           = columns.Text()
    plant_id           = columns.Text()
    workcenter_name    = columns.Text()
    workcenter_no      = columns.Text()
    workstation_name   = columns.Text()
    workstation_no     = columns.Text()
    operator_name      = columns.Text()
    operator_no        = columns.Text()
    output_stock_name  = columns.Text()
    output_stock_no    = columns.Text()
    job_order_reference_no = columns.Text()
    #prod_order_reference_no = columns.Text()
    process_no      = columns.Text()

    # var -> {"actual": x, "predicted": y, "mean": z}
    input_predicted_values  = columns.Map(
        columns.Text,
        columns.Map(columns.Text, columns.Double)
    )
    output_predicted_values = columns.Map(
        columns.Text,
        columns.Map(columns.Text, columns.Double)
    )

    @classmethod
    def saveData(cls,
                 key: str,
                 now_ts: datetime,
                 algorithm: str,
                 input_payload: dict,
                 output_payload: dict,
                 meta: dict,
                 p3_1_log=None):
        """
        Save PID-level prediction:
        PK = (process_no=key, partition_date=now_ts).
        """
        if not key or not now_ts:
            raise ValueError(
                f"[ScadaRealTimePredictions] Missing PK parts: "
                f"process_no={key}, partition_date={now_ts}"
            )
        
        jo_ref_val = (
            meta.get("job_order_reference_no")
            or meta.get("joRef")
        )
        if jo_ref_val is not None:
            jo_ref_val = str(jo_ref_val)
        
        prod_ref_val = (
            meta.get("prod_order_reference_no")
            or meta.get("refNo")
        )
        if prod_ref_val is not None:
            prod_ref_val = str(prod_ref_val)
        else:
            prod_ref_val = str("0")
        
        process_no_val = "PID_" + str(meta.get("process_no") or "")

        try:
            row = cls.create(
                prod_order_reference_no = prod_ref_val,
                partition_date    = _to_utc(now_ts),
                
                algorithm         = algorithm,
                current_ts        = _to_utc(now_ts),
                start_date        = _to_utc(meta.get("start_date")),
                customer          = meta.get("customer") or "",
                plant_id          = str(meta.get("plant_id") or ""),
                workcenter_name   = meta.get("workcenter_name") or "",
                workcenter_no     = meta.get("workcenter_no") or "",
                workstation_name  = meta.get("workstation_name") or "",
                workstation_no    = meta.get("workstation_no") or "",
                operator_name     = meta.get("operator_name") or "",
                operator_no       = meta.get("operator_no") or "",
                output_stock_name = meta.get("output_stock_name") or "",
                output_stock_no   = meta.get("output_stock_no") or "",
                input_predicted_values  = input_payload,
                output_predicted_values = output_payload,
                job_order_reference_no = jo_ref_val,
                process_no        = process_no_val #str(key)
            )
            if p3_1_log:
                p3_1_log.info(
                    f"[ScadaRealTimePredictions] Inserted key={key}, process_no={process_no_val}, prod_order_reference_no={prod_ref_val}, "
                    f"ts={_to_utc(now_ts).isoformat()}, algo={algorithm}"
                )
            return row
        except Exception as e:
            if p3_1_log:
                p3_1_log.error(
                    f"[ScadaRealTimePredictions] Error saving process_no={key}: {e}",
                    exc_info=True
                )
            raise
