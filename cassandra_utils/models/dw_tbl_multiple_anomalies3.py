import uuid
from datetime import datetime
import pytz
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider
from utils.config_reader import ConfigReader

cfg = ConfigReader()
cassandra_config = cfg["cassandra"]

CASSANDRA_HOST = cassandra_config["host"]
USERNAME = cassandra_config["username"]
PASSWORD = cassandra_config["password"]
KEYSPACE = cassandra_config["keyspace"]

connection.setup(
    hosts=[CASSANDRA_HOST],
    default_keyspace=KEYSPACE,
    auth_provider=PlainTextAuthProvider(username=USERNAME, password=PASSWORD),
    protocol_version=4
)

# --- ORM Model ---
class DwTblMultipleAnomalies(Model):
    __keyspace__ = KEYSPACE
    __table_name__ = "dw_tbl_multiple_anomalies"

    # Primary Key
    partition_date = columns.Text(partition_key=True)
    measurement_date = columns.DateTime(primary_key=True, clustering_order="DESC")
    unique_code = columns.Text(primary_key=True, clustering_order="ASC")

    # Columns
    active = columns.Boolean()
    algorithm = columns.Text()
    anomaly_detected = columns.Boolean()
    anomaly_importance = columns.Double()
    anomaly_score = columns.Double()
    current_quantity = columns.Double()
    customer = columns.Text()
    employee_id = columns.Integer()
    good = columns.Boolean()
    heapmap_threshold = columns.Double()
    id = columns.UUID(default=uuid.uuid4)
    job_order_operation_id = columns.Integer()
    job_order_operation_ref_id = columns.Text()
    machine_state = columns.Text()
    message_key = columns.Text()
    plant_id = columns.Integer()
    plant_name = columns.Text()
    plant_no = columns.Text()
    produced_stock_id = columns.Integer()
    produced_stock_name = columns.Text()
    produced_stock_no = columns.Text()
    production_order_ref_id = columns.Text()
    quantity_changed = columns.Boolean()
    shift_finish_text = columns.Text()
    shift_finish_time = columns.DateTime()
    shift_start_text = columns.Text()
    shift_start_time = columns.DateTime()
    workcenter_id = columns.Integer()
    workcenter_name = columns.Text()
    workcenter_no = columns.Text()
    workstation_id = columns.Integer()
    workstation_name = columns.Text()
    workstation_no = columns.Text()
    workstation_state = columns.Text()
    heatmap = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))
    input_variables = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))
    sensor_values = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))

    @classmethod
    def saveData(cls, *, topic_data ,main_data, anomaly_score_, anomaly_importance_,
                 anomaly_detected_, sensor_values_, heatmap_, heatmap_threshold_):

        # Extract Meta Data from Main Message
        measurement_date = getattr(main_data, 'measurement_date', None)
        if not isinstance(measurement_date, datetime):
            raise ValueError("'measurement_date' must be a datetime.datetime object")

        partition_date = measurement_date.strftime("%Y-%m-%d")

        # Extract `active`
        component_batch_list = getattr(main_data, "componentbatchlist")
        active_flag = False
        if component_batch_list and isinstance(component_batch_list, list):
            active_flag = any(str(item.get("act")).lower() == "true" for item in component_batch_list)

        # Extract other fields
        output_value_list = getattr(main_data, "outputvaluelist")
        customer =  None
        if isinstance(output_value_list, list):
            customer = next((item.get("cust") for item in output_value_list if item.get("cust")), None)


        values = {
            "partition_date" : partition_date,
            "measurement_date" : measurement_date,
            "unique_code" : str(uuid.uuid4()),
            "active" : active_flag,
            "algorithm" : None,
            "anomaly_detected" : anomaly_detected_,
            "anomaly_importance" : anomaly_importance_,
            "anomaly_score" : anomaly_score_,
            "current_quantity" : getattr(main_data, "currentquantity"),
            "customer" : customer,
            "employee_id" : topic_data['empId'],
            "good" : getattr(main_data, "good"),
            "heapmap_threshold" : heatmap_threshold_,
            "job_order_operation_id" : getattr(main_data, "joborderoperationid"),
            "job_order_operation_ref_id" : getattr(main_data, "joborderoperationrefid"),
            "machine_state" : getattr(main_data, "machinestate"),
            "message_key" : None,
            "plant_id" : getattr(main_data, "plantid"),
            "plant_name" : getattr(main_data, "plantname"),
            "plant_no" : str(getattr(main_data, "plantid")),
            "produced_stock_id" : getattr(main_data, "outputstockid"),
            "produced_stock_name" : topic_data['prodList'][0]['stNm'] if topic_data.get('prodList') else None,
            "produced_stock_no" : topic_data['prodList'][0]['stNo'] if topic_data.get('prodList') else None,
            "production_order_ref_id" : getattr(main_data, "prodorderrefid"),  
            "quantity_changed" : getattr(main_data, "quantitychanged"),
            "shift_finish_text" : str(int(getattr(main_data, "shiftfinishtime").timestamp() * 1000)) if getattr(main_data, "shiftfinishtime") else None,
            "shift_finish_time" : getattr(main_data, "shiftfinishtime"),
            "shift_start_text" : str(int(getattr(main_data, "shiftstarttime").timestamp() * 1000)) if getattr(main_data, "shiftstarttime") else None,
            "shift_start_time" : getattr(main_data, "shiftstarttime"),
            "workcenter_id" : getattr(main_data, "workcenterid"),
            "workcenter_name" : getattr(main_data, "workcentername"),
            "workcenter_no" : getattr(main_data, "workcenterno"),
            "workstation_id" : getattr(main_data, "workstationid"),
            "workstation_name" : getattr(main_data, "workstationname"),
            "workstation_no" : getattr(main_data, "workstationno"),
            "workstation_state" : getattr(main_data, "workstationstate"),
            "heatmap" : heatmap_, 
            "input_variables" : None,
            "sensor_values" : sensor_values_, 
        }

        clean_values = {k: v for k, v in values.items() if v is not None}

        try:
            instance = cls.create(**clean_values)
            return instance
        except Exception as e:
            raise
