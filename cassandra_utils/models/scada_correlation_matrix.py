import os, json
from datetime import datetime, timezone
import numpy as np
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider
from utils.config_reader import ConfigReader


# ---------- Cassandra connection ----------
cfg = ConfigReader()
cassandra_config = cfg["cassandra_props"]
CASSANDRA_HOST = cassandra_config["host"]
USERNAME       = cassandra_config["username"]
PASSWORD       = cassandra_config["password"]
KEYSPACE       = cassandra_config["keyspace"]


# Backward-compatible alias
keyspace = KEYSPACE
connection.setup(
    hosts=[CASSANDRA_HOST],
    default_keyspace=KEYSPACE,
    auth_provider=PlainTextAuthProvider(username=USERNAME, password=PASSWORD),
    protocol_version=4
)


# ---------- Helpers ----------
def _safe_float(v):
    try:
        f = float(v)
    except Exception:
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):  # NaN/Inf
        return 0.0
    return f


def _normalize_corr_data(corr_data):
    """Ensure correlation_data follows frozen<list<frozen<map<text,frozen<map<text,double>>>>>>"""
    if corr_data is None:
        return []
    if isinstance(corr_data, list):
        out = []
        for item in corr_data:
            if not isinstance(item, dict):
                continue
            cleaned_item = {}
            for k, inner in item.items():
                if not isinstance(inner, dict):
                    cleaned_item[k] = {}
                    continue
                cleaned_item[k] = {ik: _safe_float(iv) for ik, iv in inner.items()}
            if cleaned_item:
                out.append(cleaned_item)
        return out
    if isinstance(corr_data, dict):
        return [{k: {ik: _safe_float(iv) for ik, iv in (inner or {}).items()}}
                for k, inner in corr_data.items()]
    return []


def _ts_from_ms(ms):
    """Convert epoch milliseconds to exact UTC datetime (full timestamp)."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc) if ms else None


# ---------- Model ----------
class ScadaCorrelationMatrix(Model):
    __keyspace__ = keyspace
    __table_name__ = 'scada_correlation_matrix'

    #proces_no      = columns.Text(partition_key=True)
    prod_order_reference_no = columns.Text(partition_key=True)
    partition_date = columns.DateTime(primary_key=True) #desc

    # MATCHES: frozen<list<frozen<map<text, frozen<map<text, double>>>>>>>
    correlation_data = columns.List(
        columns.Map(
            columns.Text,
            columns.Map(columns.Text, columns.Double)
        )
    )

    # Regular fields
    customer          = columns.Text()
    operator_name     = columns.Text()
    operator_no       = columns.Text()
    output_stock_name = columns.Text()
    output_stock_no   = columns.Text()
    plant_id          = columns.Integer()
    workcenter_name   = columns.Text()
    workcenter_no     = columns.Text()
    workstation_name  = columns.Text()
    workstation_no    = columns.Text()
    algorithm         = columns.Text()
    job_order_reference_no = columns.Text()
    #prod_order_reference_no = columns.Text()
    proces_no      = columns.Text()


    @classmethod
    def saveData(cls, message, corr_data, p3_1_log=None):
        """Save a new correlation record with full timestamp precision."""
        try:
            out_vals  = message.get("outVals", []) or []
            prod_list = message.get("prodList", []) or []

            customer = (
                message.get("customer")
                or message.get("cust")
                or next((it.get("cust") for it in out_vals if it.get("cust")), None)
            )

            pr_stk_id = next((it.get("stNo") for it in prod_list if it.get("stNo")), None)
            pr_stk_nm = next((it.get("stNm") for it in prod_list if it.get("stNm")), None)
            wsNo = message.get("wsNo")

            proces_no = message.get('joOpId')
            pdate     = _ts_from_ms(message.get('crDt'))   # exact timestamp

            if proces_no is None or pdate is None:
                raise ValueError(f"Missing required PK parts: proces_no={proces_no}, partition_date={pdate}")

            corr_clean = _normalize_corr_data(corr_data)

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

            if p3_1_log:
                p3_1_log.info(f"[ScadaCorrelationMatrix] Saving (proces_no={proces_no}, prod_order_reference_no={prod_ref_val}, partition_date={pdate.isoformat()})")

            fields = {
                "prod_order_reference_no": prod_ref_val,
                "partition_date":    pdate,                 # <-- full-precision timestamp
                
                "correlation_data":  corr_clean,
                "customer":          customer,
                "job_order_reference_no": jo_ref_val,
                "proces_no":         str(proces_no),
                "operator_name":     message.get('opNm'),
                "operator_no":       message.get('opNo'),
                "output_stock_name": pr_stk_nm,
                "output_stock_no":   str(pr_stk_id) if pr_stk_id is not None else None,
                "plant_id":          message.get('plId'),
                "workcenter_name":   message.get("wcNm"),
                "workcenter_no":     message.get("wcNo"),
                "workstation_name":  message.get("wsNm"),
                "workstation_no":    message.get("wsNo"),
                "algorithm":         "SPEARMAN"
            }
            
            if p3_1_log:
                p3_1_log.info(
                    f"[ScadaCorrelationMatrix] Inserting new row "
                    f"(ws={wsNo}, stock={pr_stk_id}, prod_order_reference_no={prod_ref_val}, date={pdate})"
                )

            clean_fields = {k: v for k, v in fields.items() if v is not None}
            return cls.create(**clean_fields)
        except Exception as ex:
            if p3_1_log:
                p3_1_log.error(f"[ScadaCorrelationMatrix] Error saving data: {ex}", exc_info=True)
            raise

    @classmethod
    def fetchData(cls, proces_no: str, limit: int = 60):
        """Fetch latest correlation records for given process_no, sorted by time."""
        rows = cls.objects(proces_no=proces_no).limit(limit)
        return sorted(rows, key=lambda x: x.partition_date, reverse=True)


# ---------- Sync ----------
# Make sure the table has `timestamp` type for partition_date before running this.
sync_table(ScadaCorrelationMatrix)
