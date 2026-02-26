#  Imports
import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine.models import Model
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime
import pytz
from utils.config_reader import ConfigReader

cfg = ConfigReader()
cassandra_config = cfg["cassandra"]

CASSANDRA_HOST = cassandra_config["host"]
USERNAME = cassandra_config["username"]
PASSWORD = cassandra_config["password"]
KEYSPACE = cassandra_config["keyspace"]

# Cassandra Connection Setup

connection.setup(
    hosts=[CASSANDRA_HOST],
    default_keyspace=KEYSPACE,
    auth_provider=PlainTextAuthProvider(username=USERNAME, password=PASSWORD),
    protocol_version=4
)

# ORM Model Definition
class DwTblMultipleAnomalies2(Model):
    __keyspace__ = KEYSPACE
    __table_name__ = "dw_tbl_multiple_anomalies"

    # Primary Keys
    year_month = columns.Text(primary_key=True)
    partition_date = columns.Text(primary_key=True, clustering_order="DESC")
    measurement_date = columns.DateTime(primary_key=True, clustering_order="DESC")
    unique_code = columns.Text(primary_key=True, clustering_order="DESC")

    #new fields
    quantitychanged = columns.Integer
    good = columns.Boolean()

    # Regular Columns
    active = columns.Boolean()
    anomaly_importance = columns.Double()
    anomaly_score = columns.Double()
    anomaly_detected = columns.Boolean()
    create_date = columns.DateTime()
    customer = columns.Text()
    employee_id = columns.Integer()
    id = columns.UUID(default=uuid.uuid4)
    job_order_operation_id = columns.Integer()
    machine_state = columns.Text()
    message_key = columns.Text()
    month_year = columns.Text()
    organization_id = columns.Integer()
    organization_name = columns.Text()
    organization_no = columns.Text()
    plant_id = columns.Integer()
    plant_name = columns.Text()
    plant_no = columns.Text()
    produced_stock_id = columns.Integer()
    produced_stock_name = columns.Text()
    produced_stock_no = columns.Text()
    sensor_values = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))
    shift_start_time = columns.DateTime()
    shift_finish_time = columns.DateTime()
    shift_start_text = columns.Text()
    shift_finish_text = columns.Text()
    update_date = columns.DateTime()
    work_center_id = columns.Integer()
    work_center_name = columns.Text()
    work_center_no = columns.Text()
    work_station_id = columns.Integer()
    work_station_name = columns.Text()
    work_station_no = columns.Text()
    workstation_state = columns.Text()
    year = columns.Text()
    heatmap = columns.Map(columns.Text, columns.Map(columns.Text, columns.Text))
    heatmap_threshold = columns.Double()

    
#    anomaly_detection_active = columns.Boolean()    
#    stock_id = columns.Integer()

    @classmethod
    def saveData(cls, *, main_message, anomaly_score, anomaly_importance, status, sensor_dict, threshold_, heatmap_):


        #print("main message from dw table_________downstream sending")
        #print([getattr(main_message, col) for col in main_message._columns.keys()])
        # Extract timestamps
        measurement_date = getattr(main_message, 'createDate', None)
        if not isinstance(measurement_date, datetime):
            raise ValueError("'createDate' must be a datetime.datetime object")

        partition_date = measurement_date.strftime("%Y-%m-%d")
        year_month = measurement_date.strftime("%Y-%m")
        year = measurement_date.strftime("%Y")
        month_year = measurement_date.strftime("%Y-%m")


        def get_attr(obj, key):
            try:
                return getattr(obj, key)
            except:
                return None

        # Extract `active`
        component_batch_list = get_attr(main_message, "componentBatchList")
        active_flag = False
        if component_batch_list and isinstance(component_batch_list, list):
            active_flag = any(str(item.get("act")).lower() == "true" for item in component_batch_list)


        # Extract other fields
        output_value_list = get_attr(main_message, "outputValueList")
        customer =  None
        if isinstance(output_value_list, list):
            customer = next((item.get("cust") for item in output_value_list if item.get("cust")), None)
            #anom_det_act = next((str(item.get("anomDetAct")).lower() == "true" for item in output_value_list if "anomDetAct" in item), False)

              
        producted_stock_list = get_attr(main_message, "produceList")
        pr_stk_id = pr_stk_name = pr_stk_num = None
        if isinstance(producted_stock_list, list):
            pr_stk_id = next((item.get("stId") for item in producted_stock_list if item.get("stId")), None)
            pr_stk_name = next((item.get("stNm") for item in producted_stock_list if item.get("stNm")), None)
            pr_stk_num = next((item.get("stNo") for item in producted_stock_list if item.get("stNo")), None)

        values = {
            "year_month": year_month,
            "partition_date": partition_date,
            "measurement_date": measurement_date,
            "unique_code": str(uuid.uuid4()),

            "quantitychanged" : message.get("chngCycQty"),
            "good" : message.get("goodCnt"),


            "anomaly_score": float(anomaly_score),
            "anomaly_importance": float(anomaly_importance),
            "anomaly_detected": bool(status),
            "active": active_flag,
            "sensor_values": sensor_dict,
            "heatmap": heatmap_,
            "heatmap_threshold": threshold_,
            "create_date": measurement_date,
            "update_date": datetime.now(pytz.UTC),

            "employee_id": get_attr(main_message, "employeeId"),
            "job_order_operation_id": get_attr(main_message, "jobOrderOperationId"),
            "machine_state": get_attr(main_message, "machineState"),
            "organization_id": get_attr(main_message, "organizationId"),
            "organization_name": get_attr(main_message, "organizationName"),
            "organization_no": get_attr(main_message, "organizationNo"),
            "plant_id": get_attr(main_message, "plantId"),
            "plant_name": get_attr(main_message, "plantName"),
            "plant_no": get_attr(main_message, "plantNo"),

            "shift_start_time": get_attr(main_message, "shiftStartTime"),
            "shift_finish_time": get_attr(main_message, "shiftFinishTime"),
            "shift_start_text": str(int(get_attr(main_message, "shiftStartTime").timestamp() * 1000)) if get_attr(main_message, "shiftStartTime") else None,
            "shift_finish_text": str(int(get_attr(main_message, "shiftFinishTime").timestamp() * 1000)) if get_attr(main_message, "shiftFinishTime") else None,

            "work_center_id": get_attr(main_message, "workCenterId"),
            "work_center_name": get_attr(main_message, "workCenterName"),
            "work_center_no": get_attr(main_message, "workCenterNo"),
            "work_station_id": get_attr(main_message, "workStationId"),
            "work_station_name": get_attr(main_message, "workStationName"),
            "work_station_no": get_attr(main_message, "workStationNo"),

            "customer": customer,
            "message_key": get_attr(main_message, "key"),
            "month_year": month_year,
            "year": year,
            "workstation_state": get_attr(main_message, "productionState"),

            "produced_stock_id": pr_stk_id,
            "produced_stock_name": pr_stk_name,
            "produced_stock_no": pr_stk_num,
        }


        clean_values = {k: v for k, v in values.items() if v is not None}


        try:
            instance = cls.create(**clean_values)
            return instance
        except Exception as e:
            raise


