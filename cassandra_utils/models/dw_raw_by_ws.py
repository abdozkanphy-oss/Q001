# cassandra_utils/models/dw_raw_by_ws.py
from __future__ import annotations

from cassandra.cqlengine import columns, connection
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.auth import PlainTextAuthProvider

from utils.config_reader import ConfigReader

cfg = ConfigReader()
cass = cfg.get("cassandra_props") or cfg.get("cassandra") or {}

cassandra_host = cass.get("host")
username = cass.get("username")
password = cass.get("password")
keyspace = cass.get("keyspace")


def _parse_hosts(h):
    if isinstance(h, (list, tuple)):
        return [str(x).strip() for x in h if str(x).strip()]
    if isinstance(h, str):
        return [x.strip() for x in h.split(",") if x.strip()]
    return []


hosts = _parse_hosts(cassandra_host)
if not hosts or not keyspace:
    raise RuntimeError("Cassandra host/keyspace missing in config (cassandra_props).")

auth_provider = (
    PlainTextAuthProvider(username=username, password=password)
    if username and password
    else None
)

connection.setup(
    hosts=hosts,
    default_keyspace=keyspace,
    auth_provider=auth_provider,
    protocol_version=4,
)


class dw_tbl_raw_data_by_ws(Model):
    """
    Projection keyed by workstation/time:

      PRIMARY KEY ((plant_id, work_center_id, work_station_id),
                   measurement_date, unique_code)

    Goal: be training-friendly + preserve the full dw_tbl_raw_data export columns.
    """
    __keyspace__ = keyspace
    __table_name__ = "dw_tbl_raw_data_by_ws"

    # Partition
    plant_id = columns.Integer(partition_key=True)
    work_center_id = columns.Integer(partition_key=True)
    work_station_id = columns.Integer(partition_key=True)

    # Clustering
    measurement_date = columns.DateTime(primary_key=True, clustering_order="DESC")
    unique_code = columns.Text(primary_key=True, clustering_order="ASC")

    # ---- Value fields ----
    counter_reading = columns.Double()
    gen_read_val = columns.Double()
    equipment_type = columns.Boolean()  # historically unreliable but keep

    # ---- Identifiers / names ----
    customer = columns.Text()

    equipment_id = columns.Integer()
    equipment_name = columns.Text()
    equipment_no = columns.Text()

    operationtaskcode = columns.Text()
    operationname = columns.Text()
    operationno = columns.Text()

    produced_stock_id = columns.Integer()
    produced_stock_name = columns.Text()
    produced_stock_no = columns.Text()

    prod_order_reference_no = columns.Text()
    job_order_reference_no = columns.Text()
    job_order_operation_id = columns.Integer()

    plant_name = columns.Text()  # can be numeric in some exports; store as text
    work_center_name = columns.Text()
    work_center_no = columns.Text()
    work_station_name = columns.Text()
    work_station_no = columns.Text()

    # ---- Status / flags ----
    active = columns.Boolean()
    good = columns.Boolean()
    anomaly_detection_active = columns.Boolean()
    trend_calculation_active = columns.Boolean()

    machine_state = columns.Text()
    workstation_state = columns.Text()

    # ---- Time columns ----
    create_date = columns.DateTime()
    update_date = columns.DateTime()
    shift_start_time = columns.DateTime()
    shift_finish_time = columns.DateTime()

    # ---- Other ----
    stock_id = columns.Integer()
    valuation_code = columns.Text()
    year = columns.Text()


def ensure_dw_raw_by_ws():
    sync_table(dw_tbl_raw_data_by_ws)


if __name__ == "__main__":
    ensure_dw_raw_by_ws()
    print("OK: ensured/updated table dw_tbl_raw_data_by_ws")
