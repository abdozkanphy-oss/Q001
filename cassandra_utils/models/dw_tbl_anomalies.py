import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime

# Cassandra Connection Setup
from utils.config_reader import ConfigReader

cfg = ConfigReader()
cass = cfg.get("cassandra_props") or cfg.get("cassandra") or {}

cassandra_host = cass.get("host")
username = cass.get("username")
password = cass.get("password")
keyspace = cass.get("keyspace")

auth_provider = PlainTextAuthProvider(username=username, password=password) if username and password else None

connection.setup(
    hosts=[cassandra_host],
    default_keyspace=keyspace,
    auth_provider=auth_provider,
    protocol_version=4,
    lazy_connect=True,
    retry_connect=True,
    connect_timeout=20,
)

class DwTblAnomalies(Model):
    __keyspace__ = keyspace
    partition_date = columns.Text(primary_key=True)
    unique_code = columns.Text(primary_key=True)

    active = columns.Boolean()
    anomaly = columns.Integer()
    anomaly_detection_active = columns.Boolean()
    anomaly_importance = columns.Double()
    bias = columns.Double()
    coefficient = columns.Double()
    counter_reading = columns.Double()
    create_auto_maintenance_order_notification = columns.Boolean()
    create_date = columns.DateTime()
    customer = columns.Text()
    employee_id = columns.Integer()
    equipment_code_group_header_id = columns.Integer()
    equipment_id = columns.Integer()
    equipment_measuring_point_id = columns.Integer()
    equipment_name = columns.Text()
    equipment_no = columns.Text()
    job_order_operation_id = columns.Integer()
    machine_state = columns.Text()
    max_error_count_in_hour = columns.Integer()
    max_error_count_in_min = columns.Integer()
    max_error_count_in_shift = columns.Integer()
    mean_square_error = columns.Double()
    measurement_date = columns.DateTime()
    measurement_document_id = columns.Integer()
    message_key = columns.Text()
    month_year = columns.Text()
    organization_id = columns.Integer()
    organization_name = columns.Text()
    organization_no = columns.Text()
    parameter = columns.Text()
    plant_id = columns.Integer()
    plant_name = columns.Text()
    predicted_lower = columns.Double()
    predicted_upper = columns.Double()
    predicted_value = columns.Double()
    produced_stock_id = columns.Integer()
    produced_stock_name = columns.Text()
    produced_stock_no = columns.Text()
    regression_value = columns.Double()
    shift_finish_text = columns.Text()
    shift_finish_time = columns.DateTime()
    shift_start_text = columns.Text()
    shift_start_time = columns.DateTime()
    stock_id = columns.Integer()
    trend = columns.Double()
    trend_calculation_active = columns.Boolean()
    trend_lower = columns.Double()
    trend_upper = columns.Double()
    update_date = columns.DateTime()
    valuation_code = columns.Text()
    windowing_timestamp = columns.BigInt()
    work_center_id = columns.Integer()
    work_center_name = columns.Text()
    work_center_no = columns.Text()
    work_station_id = columns.Integer()
    work_station_name = columns.Text()
    work_station_no = columns.Text()
    workstation_state = columns.Text()
    year = columns.Text()

    # Add this inside the DwTblAnomalies class

    @classmethod
    def save_data(cls, message):
        # Convert timestamp fields from ms to datetime
        def to_datetime(ms):
            return datetime.fromtimestamp(ms / 1000) if ms else None

        values = {
            "partition_date": message.get("partition_date", datetime.today().strftime("%Y-%m-%d")),
            "unique_code": message.get("unique_code", str(uuid.uuid4())),

            "active": message.get("active"),
            "anomaly": message.get("anomaly"),
            "anomaly_detection_active": message.get("anomaly_detection_active"),
            "anomaly_importance": message.get("anomaly_importance"),
            "bias": message.get("bias"),
            "coefficient": message.get("coefficient"),
            "counter_reading": message.get("counter_reading"),
            "create_auto_maintenance_order_notification": message.get("create_auto_maintenance_order_notification"),
            "create_date": to_datetime(message.get("create_date")),
            "customer": str(message.get("cust")) if message.get("cust") else "turna",
            "employee_id": message.get("employee_id"),
            "equipment_code_group_header_id": message.get("equipment_code_group_header_id"),
            "equipment_id": message.get("equipment_id"),
            "equipment_measuring_point_id": message.get("equipment_measuring_point_id"),
            "equipment_name": message.get("equipment_name"),
            "equipment_no": message.get("equipment_no"),
            "job_order_operation_id": message.get("job_order_operation_id"),
            "machine_state": message.get("machine_state"),
            "max_error_count_in_hour": message.get("max_error_count_in_hour"),
            "max_error_count_in_min": message.get("max_error_count_in_min"),
            "max_error_count_in_shift": message.get("max_error_count_in_shift"),
            "mean_square_error": message.get("mean_square_error"),
            "measurement_date": to_datetime(message.get("measurement_date")),
            
            "measurement_document_id": message.get("measurement_document_id"),
            "message_key": message.get("message_key"),
            "month_year": message.get("month_year"),
            "organization_id": message.get("organization_id"),
            "organization_name": message.get("organization_name"),
            "organization_no": message.get("organization_no"),
            "parameter": message.get("parameter"),
            "plant_id": message.get("plant_id"),
            "plant_name": message.get("plant_name"),
            "predicted_lower": message.get("predicted_lower"),
            "predicted_upper": message.get("predicted_upper"),
            "predicted_value": message.get("predicted_value"),
            "produced_stock_id": message.get("produced_stock_id"),
            "produced_stock_name": message.get("produced_stock_name"),
            "produced_stock_no": message.get("produced_stock_no"),
            "regression_value": message.get("regression_value"),
            "shift_finish_text": message.get("shift_finish_text"),
            "shift_finish_time": to_datetime(message.get("shift_finish_time")),
            "shift_start_text": message.get("shift_start_text"),
            "shift_start_time": to_datetime(message.get("shift_start_time")),
            "stock_id": message.get("stock_id"),
            "trend": message.get("trend"),
            "trend_calculation_active": message.get("trend_calculation_active"),
            "trend_lower": message.get("trend_lower"),
            "trend_upper": message.get("trend_upper"),
            "update_date": to_datetime(message.get("update_date")),
            "valuation_code": message.get("valuation_code"),
            "windowing_timestamp": message.get("windowing_timestamp"),
            "work_center_id": message.get("work_center_id"),
            "work_center_name": message.get("work_center_name"),
            "work_center_no": message.get("work_center_no"),
            "work_station_id": message.get("work_station_id"),
            "work_station_name": message.get("work_station_name"),
            "work_station_no": message.get("work_station_no"),
            "workstation_state": message.get("workstation_state"),
            "year": message.get("year"),
        }

        # Filter out None values
        clean_values = {k: v for k, v in values.items() if v is not None}

        return cls.create(**clean_values)


# Sync the table if running directly
if __name__ == "__main__":
    sync_table(DwTblAnomalies)
    
