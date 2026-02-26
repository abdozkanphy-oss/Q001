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


class ScadaRealTimePredictionSummary(Model):
    __keyspace__ = keyspace
    __table_name__ = "scada_real_time_prediction_summary"

    # PRIMARY KEY ((output_stock_no, workstation_no), partition_date)
    output_stock_no = columns.Text(partition_key=True)
    workstation_no  = columns.Text(partition_key=True)
    partition_date  = columns.DateTime(primary_key=True, clustering_order="DESC")

    # payload
    algorithm               = columns.Text()
    current_ts              = columns.DateTime()
    start_date              = columns.DateTime()
    customer                = columns.Text()
    plant_id                = columns.Text()
    workcenter_name         = columns.Text()
    workcenter_no           = columns.Text()
    workstation_name        = columns.Text()
    operator_name           = columns.Text()
    operator_no             = columns.Text()
    output_stock_name       = columns.Text()

    # var -> {"actual": x, "predicted": y, "mean": z}
    input_predicted_values  = columns.Map(columns.Text, columns.Map(columns.Text, columns.Double))
    output_predicted_values = columns.Map(columns.Text, columns.Map(columns.Text, columns.Double))

    @classmethod
    def saveData(cls,
                 now_ts: datetime,
                 algorithm: str,
                 input_payload: dict,
                 output_payload: dict,
                 meta: dict,
                 p3_1_log=None):
        """
        Save WS-level (aggregated) prediction:
        PK = (output_stock_no, workstation_no, partition_date).
        """
        stock_no = (meta.get("output_stock_no") or "").strip()
        ws_no    = (meta.get("workstation_no") or "").strip()
        if not stock_no or not ws_no or not now_ts:
            raise ValueError(
                f"[ScadaRealTimePredictionsSummary] Missing PK parts: "
                f"output_stock_no={stock_no}, workstation_no={ws_no}, partition_date={now_ts}"
            )

        try:
            row = cls.create(
                output_stock_no       = stock_no,
                workstation_no        = ws_no,
                partition_date        = _to_utc(now_ts),
                algorithm             = algorithm,
                current_ts            = _to_utc(now_ts),
                start_date            = _to_utc(meta.get("start_date")),
                customer              = meta.get("customer") or "",
                plant_id              = meta.get("plant_id") or "",
                workcenter_name       = meta.get("workcenter_name") or "",
                workcenter_no         = meta.get("workcenter_no") or "",
                workstation_name      = meta.get("workstation_name") or "",
                operator_name         = meta.get("operator_name") or "",
                operator_no           = meta.get("operator_no") or "",
                output_stock_name     = meta.get("output_stock_name") or "",
                input_predicted_values  = input_payload,
                output_predicted_values = output_payload,
            )
            if p3_1_log:
                p3_1_log.info(
                    f"[ScadaRealTimePredictionsSummary] Inserted ws_no={ws_no}, "
                    f"stock_no={stock_no}, ts={now_ts.isoformat()}, algo={algorithm}"
                )
            return row
        except Exception as e:
            if p3_1_log:
                p3_1_log.error(
                    f"[ScadaRealTimePredictionsSummary] Error saving ws_no={ws_no}, "
                    f"stock_no={stock_no}: {e}",
                    exc_info=True
                )
            raise
