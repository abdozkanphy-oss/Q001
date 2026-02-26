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


class ScadaFeatureImportanceValuesSummary(Model):
    __keyspace__ = keyspace
    __table_name__ = "scada_feature_importance_values_summary"

    # PRIMARY KEY ((output_stock_no, workstation_no), algorithm, partition_date)
    output_stock_no = columns.Text(partition_key=True)
    workstation_no  = columns.Text(partition_key=True)
    algorithm       = columns.Text(primary_key=True, clustering_order="ASC")
    partition_date  = columns.DateTime(primary_key=True, clustering_order="ASC")

    # maps:  "goodCnt" -> {feature: importance}
    feature_importance_input_value  = columns.Map(
        columns.Text,
        columns.Map(columns.Text, columns.Double)
    )
    feature_importance_output_value = columns.Map(
        columns.Text,
        columns.Map(columns.Text, columns.Double)
    )

    # metadata
    customer          = columns.Text()
    plant_id          = columns.Text()
    workcenter_name   = columns.Text()
    workcenter_no     = columns.Text()
    workstation_name  = columns.Text()
    operator_name     = columns.Text()
    operator_no       = columns.Text()
    output_stock_name = columns.Text()

    partition_key = columns.Text()  # optional context (e.g. wsId, plant-level key)

    @classmethod
    def saveData(cls,
                 message: dict,
                 imp_input: dict,
                 imp_output: dict,
                 algorithm: str = "rf_perm",
                 now_ts: datetime = None,   # <<< YENİ PARAM
                 p3_1_log=None):

        # Eğer yukarıdan now_ts verilmediyse, message.crDt / createDate'ten üret
        if now_ts is None:
            now_ts = _to_utc(message.get("crDt")) \
                     or _to_utc(message.get("createDate")) \
                     or datetime.now(timezone.utc)

        # stock/workstation from message
        prod_list = message.get("prodList") or []
        out_vals  = message.get("outVals") or []

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
        ws_no = str(message.get("wsNo") or "")

        if not stock_no or not ws_no:
            raise ValueError(
                f"[ScadaFeatureImportanceValuesSummary] Missing PK parts: "
                f"output_stock_no={stock_no}, workstation_no={ws_no}"
            )

        customer = (
            message.get("customer")
            or message.get("cust")
            or (
                out_vals[0].get("cust")
                if isinstance(out_vals, list) and out_vals and isinstance(out_vals[0], dict)
                else None
            )
        )


        payload_in  = {"goodCnt": {k: _sf(v) for k, v in (imp_input  or {}).items()}}
        payload_out = {"goodCnt": {k: _sf(v) for k, v in (imp_output or {}).items()}}

        try:
            row = cls.create(
                output_stock_no = str(stock_no),
                workstation_no  = ws_no,
                algorithm       = algorithm,
                partition_date  = now_ts,   # <<< artık bunu kullanıyoruz

                feature_importance_input_value  = payload_in,
                feature_importance_output_value = payload_out,

                customer          = customer or "",
                plant_id          = str(message.get("plId") or ""),
                workcenter_name   = message.get("wcNm") or "",
                workcenter_no     = str(message.get("wcNo") or ""),
                workstation_name  = message.get("wsNm") or "",
                operator_name     = message.get("opNm") or "",
                operator_no       = message.get("opNo") or "",
                output_stock_name = stock_name or "",
                partition_key     = str(message.get("wsId") or ""),
            )
            if p3_1_log:
                p3_1_log.info(
                    f"[ScadaFeatureImportanceValuesSummary] Inserted ws={ws_no}, "
                    f"stock={stock_no}, algo={algorithm}, ts={now_ts.isoformat()}"
                )
            return row
        except Exception as e:
            if p3_1_log:
                p3_1_log.error(
                    f"[ScadaFeatureImportanceValuesSummary] Error saving ws={ws_no}, "
                    f"stock={stock_no}: {e}",
                    exc_info=True
                )
            raise
