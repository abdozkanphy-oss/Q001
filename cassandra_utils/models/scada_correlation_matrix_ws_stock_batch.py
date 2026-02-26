"""V2 correlation writer: WS+STOCK+batch window.

Goal:
  - Store batch-window correlations separately from WS-global correlations.
  - Partition for fast reads: WS+STOCK+prod_order_reference_no (batch id).
  - Keep columns aligned with existing scada_correlation_matrix* tables (no backend breaks).

Table:
  scada_correlation_matrix_ws_stock_batch
  PK = ((workstation_no, output_stock_no, prod_order_reference_no), partition_date, algorithm)
"""

from __future__ import annotations

from datetime import datetime, timezone

from cassandra.auth import PlainTextAuthProvider
from cassandra.cqlengine import columns, connection
from cassandra.cqlengine.models import Model

from utils.config_reader import ConfigReader
from utils.identity import get_stock_key


cfg = ConfigReader()
props = cfg.get("cassandra_props", {}) or cfg.get("cassandra", {}) or {}


def _parse_hosts(h):
    if isinstance(h, (list, tuple)):
        return [str(x).strip() for x in h if str(x).strip()]
    if isinstance(h, str):
        return [x.strip() for x in h.split(",") if x.strip()]
    return ["localhost"]


def _ts_from_ms(ms):
    try:
        ms_i = int(ms)
    except Exception:
        return None
    if ms_i < 10**11:
        ms_i *= 1000
    return datetime.fromtimestamp(ms_i / 1000.0, tz=timezone.utc)


def _safe_float(v):
    try:
        f = float(v)
    except Exception:
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return float(f)


def _normalize_corr_data(corr_data):
    """Ensure correlation_data is list[dict[str, dict[str, float]]]."""
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
                    cleaned_item[str(k)] = {}
                    continue
                cleaned_item[str(k)] = {str(ik): _safe_float(iv) for ik, iv in inner.items()}
            if cleaned_item:
                out.append(cleaned_item)
        return out

    if isinstance(corr_data, dict):
        return [
            {str(k): {str(ik): _safe_float(iv) for ik, iv in (inner or {}).items()}}
            for k, inner in corr_data.items()
        ]

    return []


# -------- Cassandra connection (shared keyspace) --------
cassandra_host = props.get("host", "localhost")
username = props.get("username", "")
password = props.get("password", "")
keyspace = props.get("keyspace", "das_new_pm")

_auth = PlainTextAuthProvider(username=username, password=password) if (username or password) else None

connection.setup(
    hosts=_parse_hosts(cassandra_host),
    default_keyspace=keyspace,
    auth_provider=_auth,
    protocol_version=4,
)


class ScadaCorrelationMatrixWsStockBatch(Model):
    __keyspace__ = keyspace
    __table_name__ = "scada_correlation_matrix_ws_stock_batch"

    # PK: ((workstation_no, output_stock_no, prod_order_reference_no), partition_date, algorithm)
    workstation_no = columns.Text(partition_key=True)
    output_stock_no = columns.Text(partition_key=True)
    prod_order_reference_no = columns.Text(partition_key=True)
    partition_date = columns.DateTime(primary_key=True, clustering_order="DESC")
    algorithm = columns.Text(primary_key=True, clustering_order="ASC")

    correlation_data = columns.List(
        columns.Map(columns.Text, columns.Map(columns.Text, columns.Double))
    )

    # Metadata
    customer = columns.Text()
    operator_name = columns.Text()
    operator_no = columns.Text()
    output_stock_name = columns.Text()
    plant_id = columns.Integer()
    proces_no = columns.Text()
    job_order_reference_no = columns.Text()
    workcenter_name = columns.Text()
    workcenter_no = columns.Text()
    workstation_name = columns.Text()

    @classmethod
    def saveData(cls, message: dict, corr_data, algorithm: str = "SPEARMAN", p3_1_log=None):
        msg = message or {}

        ws_no = msg.get("wsNo") or msg.get("workstation_no") or msg.get("workstationNo") or msg.get("wsId")
        ws_no = str(ws_no).strip() if ws_no not in (None, "") else ""

        stock_no = get_stock_key(msg, default="ALL")

        _null_tokens = {"", "0", "none", "null", "nan", "n/a", "na", "unknown", "unk"}

        def _norm(v):
            if v is None:
                return None
            s = str(v).strip()
            if not s or s.lower() in _null_tokens:
                return None
            return s

        prod_ref = _norm(msg.get("prod_order_reference_no")) or _norm(msg.get("refNo")) or _norm(msg.get("_batch_id"))
        if prod_ref is None:
            # Fallback: deterministic session-like id to prevent hot partitions
            ts_ms = msg.get("measDt") or msg.get("crDt")
            try:
                ts_i = int(ts_ms)
            except Exception:
                ts_i = 0
            if ts_i > 0 and ts_i < 10**11:
                ts_i *= 1000
            hour_bucket = int(ts_i // (1000 * 60 * 60)) if ts_i > 0 else 0
            prod_ref = f"SESSION|{ws_no}|{stock_no}|{hour_bucket}"

        # For batch correlations, use exact event timestamp when available.
        pdate = _ts_from_ms(msg.get("crDt"))
        if pdate is None:
            pdate = datetime.now(timezone.utc)

        algo = str(algorithm or "SPEARMAN").upper()

        if not ws_no or not stock_no or not prod_ref or not pdate:
            raise ValueError(
                "[ScadaCorrelationMatrixWsStockBatch] Missing PK fields: "
                f"ws={ws_no}, stock={stock_no}, prod_order_reference_no={prod_ref}, partition_date={pdate}"
            )

        # Extract names (prefer prodList)
        prod_list = msg.get("prodList") or []
        if isinstance(prod_list, dict):
            prod_list = [prod_list]
        pr_stk_nm = None
        if isinstance(prod_list, list):
            pr_stk_nm = next((it.get("stNm") for it in prod_list if isinstance(it, dict) and it.get("stNm")), None)

        out_vals = msg.get("outVals") or []
        customer = (
            msg.get("customer")
            or msg.get("cust")
            or next((it.get("cust") for it in out_vals if isinstance(it, dict) and it.get("cust")), None)
        )

        row = {
            # PK
            "workstation_no": ws_no,
            "output_stock_no": str(stock_no),
            "prod_order_reference_no": prod_ref,
            "partition_date": pdate,
            "algorithm": algo,
            # data
            "correlation_data": _normalize_corr_data(corr_data),
            # meta
            "customer": customer,
            "operator_name": msg.get("opNm") or msg.get("operator_name"),
            "operator_no": msg.get("opNo") or msg.get("operator_no"),
            "output_stock_name": pr_stk_nm or msg.get("output_stock_name"),
            "plant_id": msg.get("plId") or msg.get("plant_id"),
            "proces_no": str(msg.get("joOpId") or msg.get("proces_no") or "") or None,
            "job_order_reference_no": (msg.get("job_order_reference_no") or msg.get("joRef")),
            "workcenter_name": msg.get("wcNm") or msg.get("workcenter_name"),
            "workcenter_no": msg.get("wcNo") or msg.get("workcenter_no"),
            "workstation_name": msg.get("wsNm") or msg.get("workstation_name"),
        }

        clean = {k: v for k, v in row.items() if v is not None}
        cls.create(**clean)

        if p3_1_log:
            p3_1_log.info(f"[V2 Corr Batch] wrote ws={ws_no} stock={stock_no} batch={prod_ref} pdate={pdate} algo={algo}")

        return clean
