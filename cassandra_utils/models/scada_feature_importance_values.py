from datetime import datetime, timezone
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
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
        return datetime.fromtimestamp(iv/1000.0 if iv > 10**11 else iv, tz=timezone.utc)
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

class ScadaFeatureImportanceValues(Model):
    __keyspace__ = keyspace
    __table_name__ = "scada_feature_importance_values"

    # PRIMARY KEY ((proces_no), partition_date, algorithm)
    #proces_no      = columns.Text(partition_key=True)
    prod_order_reference_no = columns.Text(partition_key=True)
    partition_date = columns.DateTime(primary_key=True) #asc
    
    algorithm = columns.Text()
    feature_importance_input_value  = columns.Map(
        columns.Text,
        columns.Map(columns.Text, columns.Double)
    )
    feature_importance_output_value = columns.Map(
        columns.Text,
        columns.Map(columns.Text, columns.Double)
    )
    partition_key     = columns.Text()
    customer          = columns.Text()
    plant_id          = columns.Text()
    workcenter_no     = columns.Text()
    workcenter_name   = columns.Text()
    workstation_no    = columns.Text()
    workstation_name  = columns.Text()
    operator_no       = columns.Text()
    operator_name     = columns.Text()
    output_stock_no   = columns.Text()
    output_stock_name = columns.Text()
    job_order_reference_no = columns.Text()
    #prod_order_reference_no = columns.Text()
    proces_no      = columns.Text()

    @classmethod
    def saveData(cls,
                 message: dict,
                 imp_input: dict,
                 imp_output: dict,
                 algorithm: str = "rf_perm",
                 now_ts: datetime = None,          # <<< YENİ PARAM
                 p3_1_log=None):

        # Eğer yukarıdan now_ts geldiyse onu kullan, yoksa mevcut davranış
        if now_ts is None:
            now_ts = datetime.now(timezone.utc)

        proces_no = str(message.get("joOpId", ""))
        if not proces_no:
            raise ValueError("[ScadaFeatureImportanceValues] Missing proces_no (joOpId)")
        
        jo_ref_val = (
            message.get("job_order_reference_no")
            or message.get("joRef")
        )
        if jo_ref_val is not None:
            jo_ref_val = str(jo_ref_val)
        
        prod_ref_val = (
            message.get("prod_order_reference_no")
            or message.get("refNo")
        )
        if prod_ref_val is not None:
            prod_ref_val = str(prod_ref_val)
        else:
            prod_ref_val = str("0")

        out_vals  = message.get("outVals") or []
        prod_list = message.get("prodList") or []

        customer = (
            message.get("customer")
            or message.get("cust")
            or (
                out_vals[0].get("cust")
                if isinstance(out_vals, list) and out_vals and isinstance(out_vals[0], dict)
                else None
            )
        )

        stock_no = (
            prod_list[0].get("stNo")
            if isinstance(prod_list, list) and prod_list else
            message.get("stNo")
        )
        stock_name = (
            prod_list[0].get("stNm")
            if isinstance(prod_list, list) and prod_list else
            message.get("stNm")
        )

        payload_in  = {"goodCnt": {k: _sf(v) for k, v in (imp_input  or {}).items()}}
        payload_out = {"goodCnt": {k: _sf(v) for k, v in (imp_output or {}).items()}}

        try:
            row = cls.create(
                prod_order_reference_no = prod_ref_val,
                partition_date = now_ts,          # <<< artık burası now_ts
                
                algorithm      = algorithm,
                feature_importance_input_value  = payload_in,
                feature_importance_output_value = payload_out,
                proces_no      = proces_no,
                partition_key     = str(message.get("wsId", "")),
                customer          = customer or "",
                plant_id          = str(message.get("plId", "")),
                workcenter_no     = str(message.get("wcNo", "")),
                workcenter_name   = message.get("wcNm", ""),
                workstation_no    = str(message.get("wsNo", "")),
                workstation_name  = message.get("wsNm", ""),
                operator_no       = str(message.get("opNo", "")),
                operator_name     = message.get("opNm", ""),
                output_stock_no   = str(stock_no or ""),
                output_stock_name = stock_name or "",
                job_order_reference_no = jo_ref_val
            )
            if p3_1_log:
                p3_1_log.info(
                    f"[ScadaFeatureImportanceValues] Inserted proces_no={proces_no}, prod_order_reference_no={prod_ref_val}, "
                    f"algo={algorithm}, ts={now_ts.isoformat()}"
                )
            return row
        except Exception as e:
            if p3_1_log:
                p3_1_log.error(
                    f"[ScadaFeatureImportanceValues] Error saving proces_no={proces_no}: {e}",
                    exc_info=True
                )
            raise
