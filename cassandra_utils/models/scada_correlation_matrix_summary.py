# Imports
import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime, timezone
import pytz
from utils.config_reader import ConfigReader


cfg = ConfigReader()
cassandra_config = cfg["cassandra"]

CASSANDRA_HOST = cassandra_config["host"]
USERNAME = cassandra_config["username"]
PASSWORD = cassandra_config["password"]
KEYSPACE = cassandra_config["keyspace"]


# Backward-compatible alias
keyspace = KEYSPACE
# Cassandra Connection Setup
connection.setup(
    hosts=[CASSANDRA_HOST],
    default_keyspace=KEYSPACE,
    auth_provider=PlainTextAuthProvider(username=USERNAME, password=PASSWORD),
    protocol_version=4
)

class ScadaCorrelationMatrixSummary(Model):
    __keyspace__ = keyspace
    __table_name__ = 'scada_correlation_matrix_summary'

    # Primary Key
    output_stock_no = columns.Text(primary_key=True, partition_key=True)
    partition_date  = columns.DateTime(primary_key=True, clustering_order="DESC")  # full timestamp

    # Data (frozen<list<frozen<map<text, frozen<map<text, double>>>>>>)
    correlation_data = columns.List(
        columns.Map(
            columns.Text,
            columns.Map(columns.Text, columns.Double)
        )
    )

    # Metadata
    customer = columns.Text()
    operator_name = columns.Text()
    operator_no = columns.Text()
    output_stock_name = columns.Text()
    plant_id = columns.Integer()
    workcenter_name = columns.Text()
    workcenter_no = columns.Text()
    workstation_name = columns.Text()
    workstation_no = columns.Text()
    algorithm = columns.Text()

    @classmethod
    def saveData(cls, message, corr_data, p3_1_log=None):
        # ---- ONLY CHANGE: use exact UTC timestamp from crDt (ms) ----
        def ts(ms):
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc) if ms else None

        # --- extract message bits ---
        output_value_list = message.get("outVals", []) or []
        prod_list         = message.get("prodList", []) or []

        customer = (
            message.get("customer")
            or message.get("cust")
            or next(
                (it.get("cust") for it in output_value_list if it.get("cust")), None
            )
        )

        pr_stk_id = next((it.get("stNo") for it in prod_list if it.get("stNo")), None) \
                    or next((it.get("stNo") for it in prod_list if it.get("stNo")), None)
        pr_stk_nm = next((it.get("stNm") for it in prod_list if it.get("stNm")), None)

        pdate = ts(message.get('crDt'))   # <-- exact timestamp, not date()
        ws_no = message.get("wsNo")
        algo  = "SPEARMAN"  # constant for this writer

        if not pr_stk_id or not pdate or not ws_no:
            raise ValueError(
                f"[ScadaCorrelationMatrixSummary] Missing required keys: "
                f"output_stock_no={pr_stk_id}, partition_date={pdate}, workstation_no={ws_no}"
            )

        # freeze correlation_data (list<map<text, map<text,double>>>)
        frozen_list = [{k: {ik: float(iv) for ik, iv in (inner or {}).items()}}
                       for k, inner in (corr_data or {}).items()]

        # Fields that are NOT primary-key columns
        non_pk_fields = {
            "correlation_data":  frozen_list,
            "customer":          customer,
            "operator_name":     message.get("opNm"),
            "operator_no":       message.get("opNo"),
            "output_stock_name": pr_stk_nm,
            "plant_id":          message.get("plId"),
            "workcenter_name":   message.get("wcNm"),
            "workcenter_no":     message.get("wcNo"),
            "workstation_name":  message.get("wsNm"),
        }
        non_pk_fields = {k: v for k, v in non_pk_fields.items() if v is not None}

        # Primary key parts (kept as-is)
        pk = {
            "workstation_no":  ws_no,
            "partition_date":  pdate,            # <-- exact timestamp saved
            "algorithm":       algo,
            "output_stock_no": str(pr_stk_id),
        }

        try:
            existing = list(cls.objects(**pk).limit(1))
            if existing:
                if p3_1_log:
                    p3_1_log.info(
                        f"[ScadaCorrelationMatrixSummary] Updating existing row "
                        f"(ws={ws_no}, stock={pr_stk_id}, date={pdate}, algo={algo})"
                    )
                cls.objects(**pk).update(**non_pk_fields)
                return

            if p3_1_log:
                p3_1_log.info(
                    f"[ScadaCorrelationMatrixSummary] Inserting new row "
                    f"(ws={ws_no}, stock={pr_stk_id}, date={pdate}, algo={algo})"
                )
            cls.create(**pk, **non_pk_fields)

        except Exception as e:
            if p3_1_log:
                p3_1_log.error(f"[ScadaCorrelationMatrixSummary] Upsert error for workstation_no={ws_no}: {e}")
            raise

    @classmethod
    def fetchData(cls, output_stock_no: str, limit: int = 60, p3_1_log=None):
        if p3_1_log:
            p3_1_log.info(f"[ScadaCorrelationMatrixSummary] Fetching summary data for stock: {output_stock_no} with limit: {limit}")
        try:
            rows = cls.objects(output_stock_no=output_stock_no).allow_filtering().order_by('-partition_date').limit(limit)
            result = list(rows)
            if p3_1_log:
                p3_1_log.info(f"[ScadaCorrelationMatrixSummary] Fetched {len(result)} summary records.")
            return result
        except Exception as e:
            if p3_1_log:
                p3_1_log.error(f"[ScadaCorrelationMatrixSummary] Error while fetching summary data: {e}")
            raise
